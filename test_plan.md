# Automated Test Plan — Tahmin Platformu

**Purpose:** Stress test the entire ML platform with simulated concurrent users, edge-case data, GPU operations, and security checks. Fully automated via `test.py`.

---

## Architecture

```
test.py
│
├── Starts server.py as subprocess (or connects to running instance)
├── Waits for healthcheck (GET / returns 200)
├── Creates test users via API
├── Runs test suites sequentially (each suite may run tests concurrently)
├── Asserts no crashes, no 500s, no resource leaks
├── Kills server, reports results
│
├── test_data/           ← Bundled test fixtures
│   ├── titanic.csv            (891 rows, binary classification: Survived)
│   ├── house_prices.csv       (100 rows, regression: SalePrice)
│   ├── daily_sales.csv        (90 rows, time series: date + sales columns)
│   ├── reviews.csv            (50 rows, text column + sentiment label)
│   ├── edge_bom.csv           (BOM-prefixed Excel export)
│   ├── edge_dupes.csv         (duplicate column names: age, age, income)
│   ├── edge_empty.csv         (headers only, 0 data rows)
│   ├── edge_one_row.csv       (1 data row)
│   ├── edge_one_col.csv       (1 column only — the target)
│   ├── edge_wide.csv          (200 columns × 50 rows)
│   ├── edge_unicode.csv       (Turkish column names: müşteri_adı, işlem_tutarı)
│   ├── edge_all_nan.csv       (target column is all NaN)
│   ├── edge_single_val.csv    (target column all same value)
│   ├── edge_mixed_types.csv   (numeric column with string "error" values)
│   ├── edge_infinity.csv      (columns with inf, -inf values)
│   ├── sample_audio.wav       (10-second Turkish speech clip)
│   └── sample_audio_empty.wav (0-byte file)
```

---

## Configuration

```python
# test.py top-level config
SERVER_URL = os.environ.get("TEST_SERVER_URL", "http://localhost:8080")
ADMIN_USER = "admin"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin123!")
START_SERVER = True  # False if connecting to already-running instance
GPU_TESTS = os.environ.get("GPU_TESTS", "false").lower() == "true"
TIMEOUT_TRAINING = 600  # Max seconds to wait for training completion
TIMEOUT_AUDIO = 300     # Max seconds for audio evaluation
TIMEOUT_PREDICT = 30    # Max seconds for single prediction
NUM_CONCURRENT_USERS = 3
```

---

## Test Fixtures Setup

```python
def setup():
    """Run once before all tests."""
    # 1. Start server.py as subprocess (if START_SERVER=True)
    #    server_proc = subprocess.Popen(["python3.11", "server.py"], ...)
    #    Wait for GET / to return 200 (poll every 1s, timeout 120s)

    # 2. Login as admin
    #    POST /api/auth/login {username: "admin", password: ADMIN_PASSWORD}
    #    Store admin_token

    # 3. Create test users
    #    POST /api/auth/register {username: "testuser1", password: "Test123!", ...}
    #    POST /api/auth/register {username: "testuser2", password: "Test123!", ...}
    #    POST /api/auth/register {username: "testuser3", password: "Test123!", ...}
    #    POST /api/admin/approve-user {username: "testuser1"}  (as admin)
    #    POST /api/admin/approve-user {username: "testuser2"}  (as admin)
    #    POST /api/admin/approve-user {username: "testuser3"}  (as admin)

    # 4. Login each test user, store session tokens
    #    sessions = {
    #        "admin": requests.Session(),   # with admin token
    #        "user1": requests.Session(),   # with testuser1 token
    #        "user2": requests.Session(),   # with testuser2 token
    #        "user3": requests.Session(),   # with testuser3 token
    #    }

    # 5. Pre-upload CSVs for tests that need existing models
    #    Upload titanic.csv via user1 → get temp_id_titanic
    #    Train classification model → wait for completion → get model_id_titanic
    #    Set visibility to "public" for cross-user tests
```

---

## Suite 1: Sequential Smoke Tests (13 tests)

Each test runs alone, verifies basic functionality works.

### S1: Upload CSV
```
Action:  POST /api/upload-csv (multipart, titanic.csv)
Expect:  200, response has temp_id, columns list, preview rows
Assert:  len(columns) > 1, preview has <= 5 rows
```

### S2: Train Tabular Classification
```
Action:  POST /api/train {temp_id, target_column: "Survived", task_type: "classification", preset: "medium_quality"}
Expect:  200, response has job_id
Action:  Poll GET /api/training/{job_id}/status every 3s
Expect:  Eventually status="done", model_id returned
Assert:  GET /api/models/{model_id} returns meta with submodels, best_score > 0
Time:    < 300s
```

### S3: Train Tabular Regression
```
Action:  Upload house_prices.csv, train with target="SalePrice", task_type="regression"
Expect:  status="done", best_score > 0
Assert:  meta.problem_type == "regression"
```

### S4: Train Time Series
```
Action:  Upload daily_sales.csv, train with target="sales", task_type="timeseries", timestamp_column="date"
Expect:  status="done"
Assert:  meta.task_type == "timeseries", submodels exist
CRITICAL: Watch server stdout for SIGABRT — this is the Ray/CUDA test
```

### S5: Train Text/Sentiment Model
```
Action:  Upload reviews.csv, train with target="sentiment", task_type="classification"
Expect:  status="done"
Assert:  meta.text_columns is not empty, meta.embedding_model exists, meta.embedding_dim exists
```

### S6: Single Prediction (Classification)
```
Action:  POST /api/models/{titanic_model_id}/predict/WeightedEnsemble_L2
         Body: {features: {Pclass: 3, Sex: "male", Age: 25, Fare: 7.25, ...}}
Expect:  200, response has prediction (0 or 1), probabilities
Assert:  probabilities sum to ~1.0, prediction is 0 or 1
```

### S7: Single Prediction (Regression)
```
Action:  POST /api/models/{house_model_id}/predict/{best_submodel}
         Body: {features: {LotArea: 8450, OverallQual: 7, ...}}
Expect:  200, response has prediction (numeric)
Assert:  prediction is a finite number
```

### S8: Time Series Prediction
```
Action:  POST /api/models/{ts_model_id}/predict/{best_submodel}
         Body: {history: [{date: "2024-01-01", sales: 100}, ...30 rows]}
Expect:  200, response has forecast array
Assert:  len(forecast) == prediction_length
```

### S9: Batch Prediction
```
Action:  POST /api/models/{titanic_model_id}/predict-batch/WeightedEnsemble_L2
         Body: multipart with test CSV (100 rows)
Expect:  200, Content-Type contains text/csv
Assert:  Downloaded CSV has 100 rows + header, has predicted column
```

### S10: Audio Evaluation (if Whisper available)
```
Action:  POST /api/audio-evaluate (multipart with sample_audio.wav, schema, prompt)
Expect:  200, job_id returned
Action:  Poll status every 3s
Expect:  status="done"
Assert:  Results contain transcript, predicted values
```

### S11: Explainability Analysis
```
Action:  GET /api/models/{titanic_model_id}/explain/WeightedEnsemble_L2
Expect:  200, response has feature_importance, correlations, data_profile
Assert:  feature_importance is non-empty, all values are finite numbers
```

### S12: Export Airflow DAG
```
Action:  GET /api/models/{titanic_model_id}/export/WeightedEnsemble_L2/airflow
Expect:  200, Content-Type is application/zip
Assert:  ZIP contains .py file with valid Python syntax
```

### S13: Export MSSQL SQL
```
Action:  GET /api/models/{titanic_model_id}/export/WeightedEnsemble_L2/mssql
Expect:  200, response contains SQL text
Assert:  SQL contains SELECT, CASE WHEN, no unescaped brackets
```

---

## Suite 2: Concurrent Stress Tests (7 tests)

Each test spawns multiple threads simulating concurrent users.

### C1: Three Users Train Simultaneously
```
Threads: 3
  user1: Upload + train titanic (classification)
  user2: Upload + train house_prices (regression)
  user3: Upload + train daily_sales (timeseries)
Assert:
  - All 3 eventually complete (queued, not rejected)
  - Server process still alive (no SIGABRT)
  - FairJobQueue served them round-robin
  - All user_action_tracker registrations cleared
```

### C2: Train + Predict Simultaneously
```
Threads: 3
  user1: Starts training new model (long-running)
  user2: Predicts with existing titanic model (5 rapid predictions)
  user3: Predicts with same titanic model (5 rapid predictions)
Assert:
  - Training completes
  - All 10 predictions return 200 (not 503 "busy")
  - Prediction semaphore didn't deadlock
  - model_ref_counter back to 0 for all models
```

### C3: Audio Eval + Training + Prediction
```
Threads: 3
  user1: Audio evaluation (Whisper + LLM)
  user2: Training tabular model
  user3: Predicting with existing model
Assert:
  - All complete (some may be queued)
  - Whisper used GPU (check logs for "GPU (ct2)")
  - No OOM errors (check for 503 responses)
  - resource_manager reservations back to 0
```

### C4: Five Users Hit Dashboard
```
Threads: 5
  All: GET /api/dashboard simultaneously
Assert:
  - All return 200
  - Response time < 5s for each
  - No deadlocks
```

### C5: Same Model Concurrent Predictions
```
Threads: 5
  All: Predict with same titanic model simultaneously
Assert:
  - All return 200 (semaphore allows 3 concurrent, others wait)
  - No model_ref_counter leaks
  - No 500 errors
```

### C6: Rapid-Fire Predictions
```
Threads: 1 (but rapid sequential)
  10 predictions in 2 seconds (same model)
Assert:
  - All return 200 or 503 (rate limit)
  - No 500 errors
  - Server stable after burst
```

### C7: User Submits Training Then Navigates Away
```
Threads: 1
  user1: Start training → immediately start ANOTHER training
Assert:
  - Second training returns 429 "already active"
  - First training completes
  - user_action_tracker correctly cleared after completion
  - User can start new training after first completes
```

---

## Suite 3: Edge Case Data Tests (15 tests)

Each test uploads a specially crafted CSV and verifies graceful handling.

### E1: Empty CSV (headers only)
```
Upload edge_empty.csv, attempt training
Expect: Training fails with clear error about insufficient rows (< 20)
Assert: HTTP 400 or training status="error" with Turkish message
```

### E2: Single Row CSV
```
Upload edge_one_row.csv, attempt training
Expect: Same as E1 — insufficient rows
```

### E3: Single Column CSV (only target)
```
Upload edge_one_col.csv, attempt training
Expect: Error "CSV'de en az bir özellik sütunu olmalı"
```

### E4: BOM-Prefixed CSV
```
Upload edge_bom.csv, attempt training
Expect: Training succeeds (BOM stripped by utf-8-sig encoding)
Assert: First column name does NOT start with \ufeff
```

### E5: Duplicate Column Names
```
Upload edge_dupes.csv, attempt training
Expect: Training succeeds (pandas auto-renames), warning logged
Assert: Log contains "Duplicate columns detected"
```

### E6: All-NaN Target
```
Upload edge_all_nan.csv, attempt training
Expect: Error about insufficient rows after NaN removal
```

### E7: Single-Value Target
```
Upload edge_single_val.csv, attempt training
Expect: Error "Hedef sütunda en az 2 farklı değer olmalı"
```

### E8: Unicode/Turkish Column Names
```
Upload edge_unicode.csv, attempt training
Expect: Training succeeds, model can predict
Assert: Feature columns include Turkish names in meta.json
```

### E9: Wide Dataset (200 columns)
```
Upload edge_wide.csv, attempt training
Expect: Training succeeds (may be slow)
Assert: meta.feature_columns has ~199 entries
```

### E10: Mixed Types in Column
```
Upload edge_mixed_types.csv, attempt training
Expect: Training succeeds (infinity strings replaced with NaN)
```

### E11: Infinity Values
```
Upload edge_infinity.csv, attempt training
Expect: Training succeeds (infinities replaced with NaN)
Assert: No Infinity in API response JSON
```

### E12: Zero-Length Audio File
```
Upload sample_audio_empty.wav for audio evaluation
Expect: Per-file error "Ses dosyası boş veya bulunamadı"
Assert: Job does NOT crash, returns error for that file
```

### E13: Time Series with prediction_length > data
```
Upload daily_sales.csv (90 rows), set prediction_length=80
Expect: Error "Tahmin uzunluğu veri uzunluğunun yarısından küçük olmalı"
```

### E14: Time Series with Constant Target
```
Upload CSV where all target values are 42.0
Expect: Error "Hedef sütundaki tüm değerler aynı"
```

### E15: Batch Prediction Column Mismatch
```
Train model on titanic.csv, then batch predict with house_prices.csv
Expect: Error about missing feature columns or degraded prediction
Assert: No 500 error
```

---

## Suite 4: Security Tests (8 tests)

### SEC1: Unauthenticated Access
```
Action: GET /api/dashboard (no auth header)
Expect: 401
```

### SEC2: Private Model Access by Other User
```
Setup: user1 trains model, sets visibility="private"
Action: user2 tries GET /api/models/{model_id}
Expect: 403
```

### SEC3: Job Status IDOR
```
Setup: user1 starts training → get job_id
Action: user2 tries GET /api/training/{job_id}/status
Expect: 403
```

### SEC4: Admin Endpoint as Regular User
```
Action: user1 tries POST /api/admin/approve-user
Expect: 403
```

### SEC5: Rate Limiting
```
Action: 20 rapid POST /api/auth/login with wrong password
Expect: Eventually returns 429
```

### SEC6: Malformed JSON
```
Action: POST /api/train with body "not json"
Expect: 400 "Geçersiz JSON verisi"
```

### SEC7: Malformed Multipart
```
Action: POST /api/upload-csv with invalid Content-Type
Expect: 400
```

### SEC8: Very Large Upload
```
Action: POST /api/upload-csv with 250MB payload
Expect: 413 "too large"
```

---

## Suite 5: GPU Stress Tests (4 tests, --gpu flag required)

### GPU1: Serial GPU Cycle
```
Train model → predict → train another → predict
Assert: VRAM measured values stored in meta.json
Assert: No OOM errors across the cycle
```

### GPU2: Whisper During Training
```
Start tabular training (CPU, num_gpus=0)
Simultaneously run audio transcription (GPU)
Assert: Whisper uses GPU (logs show "GPU (ct2)")
Assert: Training completes on CPU
Assert: No VRAM conflict
```

### GPU3: VRAM Measurement Accuracy
```
Predict with model → check meta.json for measured_vram_peak_mb
Assert: Value is a positive integer
Assert: Value is reasonable (50-5000 MB range)
Predict again → check value hasn't changed (measurement is one-time)
```

### GPU4: VRAM Exhaustion Recovery
```
Load 5+ different models for prediction in rapid sequence
Assert: Some return 503 "Sunucu meşgul" (VRAM gating works)
Assert: After waiting, prediction succeeds (VRAM freed by eviction)
Assert: Server still alive (no OOM crash)
```

---

## Assertions Framework

Every test validates these universal assertions:

```python
def assert_healthy(test_name, response, server_proc):
    """Universal post-test checks."""
    # 1. Server process still alive
    assert server_proc.poll() is None, f"{test_name}: Server crashed (SIGABRT?)"

    # 2. No 500 Internal Server Error (unless explicitly expected)
    assert response.status_code != 500, f"{test_name}: Got 500: {response.text[:200]}"

    # 3. Response is valid JSON (for API endpoints)
    if "application/json" in response.headers.get("Content-Type", ""):
        data = response.json()  # Should not raise
        # No raw Python tracebacks in error messages
        if "error" in data:
            assert "Traceback" not in data["error"], f"{test_name}: Traceback leaked"
            assert "File \"/" not in data["error"], f"{test_name}: File path leaked"

    # 4. Response time reasonable
    assert response.elapsed.total_seconds() < 30, f"{test_name}: Response took {response.elapsed}s"
```

Post-suite checks:

```python
def assert_no_resource_leaks():
    """Check after all tests complete."""
    status = admin_session.get("/api/server/status").json()

    # All VRAM reservations released
    assert status.get("vram_reserved_mb", 0) == 0, "VRAM reservation leak"

    # All RAM reservations released
    assert status.get("ram_reserved_mb", 0) == 0, "RAM reservation leak"

    # No orphan temp files
    temp_dir = DATA_DIR / "temp"
    if temp_dir.exists():
        orphans = list(temp_dir.iterdir())
        assert len(orphans) == 0, f"Orphan temp dirs: {orphans}"
```

---

## Running

```bash
# Full test suite (CPU only, starts server locally):
python test.py

# Against running instance (e.g., Vast.ai):
TEST_SERVER_URL=https://nocodeml.xyz ADMIN_PASSWORD=YourPassword python test.py

# GPU tests included:
GPU_TESTS=true python test.py

# Single suite:
python test.py --suite smoke
python test.py --suite concurrent
python test.py --suite edge
python test.py --suite security
python test.py --suite gpu

# Verbose (print each test result):
python test.py -v

# Stop on first failure:
python test.py --failfast
```

---

## Expected Output

```
=== Tahmin Platformu Test Suite ===
Server: http://localhost:8080
GPU tests: disabled

[Setup] Server started (PID 12345)
[Setup] Admin login OK
[Setup] Created 3 test users

Suite 1: Smoke Tests (13 tests)
  S1  Upload CSV ........................ PASS (0.3s)
  S2  Train Classification ............. PASS (45.2s)
  S3  Train Regression ................. PASS (38.1s)
  S4  Train Time Series ................ PASS (62.4s)  ← Ray didn't crash!
  S5  Train Sentiment .................. PASS (51.3s)
  S6  Predict Classification ........... PASS (0.2s)
  S7  Predict Regression ............... PASS (0.1s)
  S8  Predict Time Series .............. PASS (0.4s)
  S9  Batch Predict .................... PASS (1.1s)
  S10 Audio Evaluation ................. PASS (28.3s)
  S11 Explainability ................... PASS (3.2s)
  S12 Export Airflow ................... PASS (0.5s)
  S13 Export MSSQL ..................... PASS (0.3s)

Suite 2: Concurrent Tests (7 tests)
  C1  Three simultaneous trains ........ PASS (95.2s)
  C2  Train + predict .................. PASS (48.1s)
  C3  Audio + train + predict .......... PASS (62.0s)
  C4  Five dashboard requests .......... PASS (0.8s)
  C5  Same model concurrent predict .... PASS (1.2s)
  C6  Rapid-fire predictions ........... PASS (2.1s)
  C7  Double training rejection ........ PASS (3.5s)

Suite 3: Edge Cases (15 tests)
  E1-E15 ................................ 15/15 PASS

Suite 4: Security (8 tests)
  SEC1-SEC8 ............................. 8/8 PASS

[Post-check] No resource leaks ......... PASS
[Post-check] No orphan temp files ...... PASS
[Teardown] Server stopped

=== RESULTS: 43/43 PASSED, 0 FAILED ===
Total time: 312.4s
```

---

## Test Data Generation

For fixtures that need specific formats, generate them programmatically:

```python
def generate_test_data():
    """Create test CSV files in test_data/ directory."""
    import pandas as pd
    import numpy as np

    os.makedirs("test_data", exist_ok=True)

    # daily_sales.csv (time series)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    sales = np.random.normal(1000, 200, 90).clip(100)
    pd.DataFrame({"date": dates, "sales": sales.round(2)}).to_csv("test_data/daily_sales.csv", index=False)

    # reviews.csv (text + sentiment)
    reviews = [
        {"text": "Harika ürün, çok memnunum", "sentiment": "positive"},
        {"text": "Berbat kalite, para çöpe gitti", "sentiment": "negative"},
        # ... 48 more rows
    ]
    pd.DataFrame(reviews).to_csv("test_data/reviews.csv", index=False)

    # edge_bom.csv (BOM-prefixed)
    with open("test_data/edge_bom.csv", "w", encoding="utf-8-sig") as f:
        f.write("age,income,target\n25,50000,1\n30,60000,0\n")  # ... more rows

    # edge_dupes.csv
    pd.DataFrame({"age": [25,30,35], "age": [1,2,3], "target": [0,1,0]}).to_csv(...)

    # edge_single_val.csv
    pd.DataFrame({"x": range(30), "target": [1]*30}).to_csv(...)

    # edge_unicode.csv
    pd.DataFrame({"müşteri_adı": ["Ali","Veli"], "işlem_tutarı": [100,200], "sonuç": [1,0]}).to_csv(...)

    # ... etc for all fixtures
```

---

## Notes for Implementation

1. **Training takes time.** Use `medium_quality` preset and small datasets to keep training under 60s per model. Total suite should complete in ~5-10 minutes.

2. **Polling pattern.** Every training/audio test needs a poll loop:
   ```python
   def wait_for_job(session, url, timeout=300):
       deadline = time.time() + timeout
       while time.time() < deadline:
           res = session.get(url).json()
           if res.get("status") == "done": return res
           if res.get("status") == "error": raise AssertionError(f"Job failed: {res.get('error')}")
           time.sleep(3)
       raise TimeoutError(f"Job didn't complete within {timeout}s")
   ```

3. **Server output capture.** Capture server stdout/stderr to check for SIGABRT, CUDA errors, and warnings:
   ```python
   server_proc = subprocess.Popen(
       ["python3.11", "server.py"],
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       env={**os.environ, "ADMIN_PASSWORD": ADMIN_PASSWORD}
   )
   ```

4. **Cleanup between suites.** Delete all test models after each suite to reset VRAM state:
   ```python
   for model in get_all_models(admin_session):
       admin_session.delete(f"/api/models/{model['id']}")
   ```

5. **GPU detection.** Only run GPU suite if torch.cuda.is_available():
   ```python
   if GPU_TESTS:
       try:
           import torch
           assert torch.cuda.is_available(), "GPU_TESTS=true but no CUDA GPU detected"
       except (ImportError, AssertionError) as e:
           print(f"Skipping GPU tests: {e}")
   ```
