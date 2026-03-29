#!/usr/bin/env python3
"""
Tahmin Platformu: AutoGluon destekli tablo tabanlı tahmin uygulaması.
Yerleşik http.server modülü kullanan backend sunucu.
Kimlik doğrulama, yetkilendirme, model görünürlüğü ve onay sistemi içerir.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import uuid
import csv
import io
import time
import shutil
import re
import math
import threading
import traceback
import mimetypes
import hashlib
import secrets
import hmac
import signal
import atexit
import gc
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TahminPlatformu")

# ── In-memory ring buffer for /api/admin/logs endpoint ────────────
# Captures ALL output: log.info(), print(), third-party library output, warnings, etc.
import collections as _collections
_LOG_RING_BUFFER = _collections.deque(maxlen=2000)
_LOG_RING_LOCK = threading.Lock()

class _TeeWriter:
    """Wraps a stream so every write() also appends complete lines to the ring buffer."""
    def __init__(self, original):
        self._original = original
        self._partial = ""
    def write(self, text):
        self._original.write(text)
        if not text:
            return
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                with _LOG_RING_LOCK:
                    _LOG_RING_BUFFER.append(line)
    def flush(self):
        self._original.flush()
    def fileno(self):
        return self._original.fileno()
    def isatty(self):
        return self._original.isatty()
    def __getattr__(self, name):
        return getattr(self._original, name)

sys.stdout = _TeeWriter(sys.stdout)
sys.stderr = _TeeWriter(sys.stderr)


# ══════════════════════════════════════════════════════════════════
#  NaN-SAFE JSON ENCODER & DATA CLEANING UTILITIES
# ══════════════════════════════════════════════════════════════════

class NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN, Infinity, -Infinity to null."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.ndarray,)):
                return sanitize_value(obj.tolist())
            if isinstance(obj, (np.void,)):
                return None
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(obj, (pd.Timestamp,)):
                return obj.isoformat()
            if pd.isna(obj):
                return None
        except (ImportError, TypeError, ValueError):
            pass
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

    def encode(self, obj):
        return super().encode(sanitize_value(obj))

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(sanitize_value(obj), _one_shot=_one_shot)


def sanitize_value(obj, _depth=0):
    """Recursively replace NaN, Infinity, -Infinity with None."""
    if _depth > 100:
        return None  # Prevent stack overflow on deeply nested / circular structures
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return [sanitize_value(x, _depth + 1) for x in obj.tolist()]
        if isinstance(obj, (np.void,)):
            return None
    except ImportError:
        pass
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {k: sanitize_value(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_value(x, _depth + 1) for x in obj]
    return obj


def safe_json_dumps(data, **kwargs):
    """JSON serialize with NaN/Infinity safety."""
    return json.dumps(sanitize_value(data), cls=NaNSafeEncoder, default=str, **kwargs)


def _read_csv_with_fallback(csv_path, **kwargs):
    """Read CSV with encoding fallback: try utf-8-sig first, then detect encoding."""
    # Reject binary files disguised as CSV (latin-1 accepts any byte sequence)
    with open(csv_path, "rb") as f:
        head = f.read(4096)
    if b'\x00' in head:
        raise ValueError("Dosya bir metin/CSV dosyası değil (ikili veri algılandı). Lütfen geçerli bir CSV yükleyin.")
    try:
        return pd.read_csv(csv_path, encoding='utf-8-sig', **kwargs)
    except UnicodeDecodeError:
        # Try common Turkish/European encodings before giving up
        for enc in ['windows-1254', 'latin-1', 'windows-1252']:
            try:
                return pd.read_csv(csv_path, encoding=enc, **kwargs)
            except (UnicodeDecodeError, Exception):
                continue
        # Last resort: lossy read — undecodable bytes become U+FFFD
        log.warning(f"[CSV] All encoding attempts failed for {csv_path} — using lossy UTF-8 (data may contain replacement characters)")
        return pd.read_csv(csv_path, encoding='utf-8', errors='replace', **kwargs)


def clean_dataframe(df, context="training", timestamp_column=None, item_id_column=None):
    """Clean a DataFrame for ML processing."""
    import pandas as pd
    import numpy as np

    df = df.copy()

    report = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "issues_found": [],
        "actions_taken": [],
    }

    # ── Time Series mode: do NOT drop rows — it breaks frequency ──
    if context == "timeseries":
        # Convert timestamp column
        if timestamp_column and timestamp_column in df.columns:
            try:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                report["actions_taken"].append(f"'{timestamp_column}' sütunu datetime'a dönüştürüldü")
                # Drop rows where timestamp became NaT (blank/null cells) — AutoGluon requires non-null timestamps
                nat_count = df[timestamp_column].isna().sum()
                if nat_count > 0:
                    df = df.dropna(subset=[timestamp_column]).reset_index(drop=True)
                    report["issues_found"].append(f"{nat_count} geçersiz/boş zaman damgası satırı")
                    report["actions_taken"].append(f"{nat_count} NaT zaman damgası satırı silindi")
            except Exception as e:
                report["issues_found"].append(f"'{timestamp_column}' datetime dönüşümü başarısız: {e}")

        # Drop rows where ALL non-timestamp columns are empty (e.g. future placeholder rows)
        non_ts_cols = [c for c in df.columns if c != timestamp_column]
        if non_ts_cols:
            all_empty_mask = df[non_ts_cols].isna().all(axis=1)
            empty_count = all_empty_mask.sum()
            if empty_count > 0:
                df = df[~all_empty_mask].reset_index(drop=True)
                report["issues_found"].append(f"{empty_count} tamamen boş satır (yalnızca tarih içeren)")
                report["actions_taken"].append(f"{empty_count} boş satır silindi")

        # Replace infinities in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                report["issues_found"].append(f"'{col}' sütununda {inf_count} sonsuz değer")
                report["actions_taken"].append(f"'{col}' sütunundaki sonsuz değerler NaN ile değiştirildi")

        # Forward-fill then backward-fill to maintain sequence integrity
        # Exclude ID and timestamp columns — ffill on item_id would silently mix series
        fill_exclude = set()
        if timestamp_column:
            fill_exclude.add(timestamp_column)
        if item_id_column and item_id_column in df.columns:
            fill_exclude.add(item_id_column)
        fill_cols = [c for c in df.columns if c not in fill_exclude]

        nan_before = df[fill_cols].isna().sum().sum() if fill_cols else 0
        total_cells = len(fill_cols) * len(df) if fill_cols else 1
        # Reject datasets with >80% missing values — ffill would fabricate most of the data
        pre_fill_ratio = nan_before / max(total_cells, 1)
        if pre_fill_ratio > 0.8:
            raise ValueError(
                f"Veri setinde çok fazla eksik değer var ({pre_fill_ratio:.0%}). "
                f"Zaman serisi analizi için daha yoğun veri gerekiyor."
            )
        # Sort chronologically before fill to prevent future→past leakage
        if timestamp_column and timestamp_column in df.columns:
            sort_cols = [item_id_column, timestamp_column] if (item_id_column and item_id_column in df.columns) else [timestamp_column]
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
        # Group by item_id to prevent cross-series contamination
        if item_id_column and item_id_column in df.columns:
            df[fill_cols] = df.groupby(item_id_column)[fill_cols].transform(
                lambda x: x.ffill().bfill())
        else:
            df[fill_cols] = df[fill_cols].ffill().bfill()
        # Convert any remaining object columns that became mixed-type after fill
        df[fill_cols] = df[fill_cols].infer_objects()
        # Fill any remaining NaN (e.g. entire series missing a column) with column median/0
        remaining_nan = df[fill_cols].isna().sum()
        still_nan_cols = remaining_nan[remaining_nan > 0].index.tolist()
        if still_nan_cols:
            numeric_fill = df[still_nan_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_fill:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
            report["actions_taken"].append(f"{len(still_nan_cols)} sütunda kalan NaN değerler medyan ile dolduruldu")
        nan_after = df[fill_cols].isna().sum().sum() if fill_cols else 0
        filled_count = nan_before - nan_after
        if filled_count > 0:
            fill_ratio = filled_count / max(total_cells, 1)
            report["issues_found"].append(f"Toplam {filled_count} eksik değer bulundu")
            report["actions_taken"].append(f"{filled_count} eksik değer ileri/geri doldurma ile tamamlandı")
            if fill_ratio > 0.3:
                report["issues_found"].append(
                    f"⚠ Yüksek eksik değer oranı: {fill_ratio:.0%} — model kalitesi düşük olabilir"
                )

        # Strip strings
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            if col != timestamp_column:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        report["cleaned_rows"] = len(df)
        report["cleaned_cols"] = len(df.columns)
        return df, report

    # ── Standard tabular cleaning ──
    # Only drop empty rows/columns during training — prediction must preserve 1:1 row alignment
    if context != "prediction":
        empty_rows = df.isna().all(axis=1).sum()
        if empty_rows > 0:
            df = df.dropna(how='all')
            report["issues_found"].append(f"{empty_rows} tamamen boş satır")
            report["actions_taken"].append(f"{empty_rows} boş satır silindi")

        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report["issues_found"].append(f"{len(empty_cols)} boş sütun: {empty_cols}")
            report["actions_taken"].append(f"{len(empty_cols)} boş sütun silindi")

    if context == "training":
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
            report["issues_found"].append(f"{dup_count} tekrarlanan satır")
            report["actions_taken"].append(f"{dup_count} tekrarlanan satır silindi")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            report["issues_found"].append(f"'{col}' sütununda {inf_count} sonsuz değer")
            report["actions_taken"].append(f"'{col}' sütunundaki sonsuz değerler NaN ile değiştirildi")

    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in str_cols:
        empty_str_count = (df[col] == '').sum()
        if empty_str_count > 0:
            df[col] = df[col].replace('', np.nan)
            report["issues_found"].append(f"'{col}' sütununda {empty_str_count} boş metin")
            report["actions_taken"].append(f"'{col}' sütunundaki boş metinler NaN ile değiştirildi")

    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if len(nan_cols) > 0 and len(df) > 0:
        for col, count in nan_cols.items():
            pct = round(count / len(df) * 100, 1)
            report["issues_found"].append(f"'{col}' sütununda {count} NaN değer (%{pct})")
        report["actions_taken"].append("NaN değerler AutoGluon'un işlemesi için korundu")

    report["cleaned_rows"] = len(df)
    report["cleaned_cols"] = len(df.columns)

    return df, report


def clean_prediction_input(features_dict, column_types):
    """Clean a single prediction input dictionary."""
    import numpy as np
    cleaned = {}
    for key, value in features_dict.items():
        if value is None or value == '' or value == 'null' or value == 'NaN' or value == 'nan':
            cleaned[key] = np.nan
            continue
        col_type = column_types.get(key, '').lower()
        if any(t in col_type for t in ['int', 'float', 'number']):
            try:
                val = float(value)
                if math.isnan(val) or math.isinf(val):
                    cleaned[key] = np.nan
                else:
                    cleaned[key] = val
            except (ValueError, TypeError):
                # Handle Turkish/European decimal format: "3,14" or "1.250,50"
                try:
                    s = str(value).strip()
                    if ',' in s:
                        # "1.250,50" → "1250.50" or "3,14" → "3.14"
                        s = s.replace('.', '').replace(',', '.')
                    val = float(s)
                    cleaned[key] = np.nan if (math.isnan(val) or math.isinf(val)) else val
                except (ValueError, TypeError):
                    cleaned[key] = np.nan
        else:
            cleaned[key] = value
    return cleaned


# ══════════════════════════════════════════════════════════════════
#  WINDOWS MULTIPROCESSING FIX: Disable DataLoader worker processes.
#  On Windows, PyTorch DataLoader workers use 'spawn' which re-imports
#  the entire module, causing heavy re-initialization and crashes
#  (wandb/pydantic/timm import chain). For single-row predictions
#  num_workers=0 is faster anyway (no IPC overhead).
# ══════════════════════════════════════════════════════════════════
os.environ["AG_AUTOMM_NUM_WORKERS"] = "0"
os.environ["AG_AUTOMM_NUM_WORKERS_EVALUATION"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("RAY_USE_MULTIPROCESSING_CPU_COUNT", "1")  # Fix Ray CPU detection in containers
# Cap low-level thread pools to prevent host CPU count from spawning excessive threads in containers
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")

# ── AutoGluon imports ──────────────────────────────────────────────
try:
    from autogluon.tabular import TabularPredictor, TabularDataset
    import pandas as pd
    import numpy as np
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("UYARI: AutoGluon yüklü değil. Yüklemek için: pip install autogluon")
    print("Sunucu başlayacak ama eğitim/tahmin çalışmayacak.")

AUTOGLUON_TS_AVAILABLE = False
try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_TS_AVAILABLE = True
except ImportError:
    print("BİLGİ: autogluon.timeseries yüklü değil. Zaman serisi tahmini devre dışı.")
    print("Yüklemek için: pip install autogluon.timeseries")

# NOTE: autogluon.multimodal is no longer required for text/NLP tasks.
# Text/NLP now uses sentence-transformers + TabularPredictor (faster, no GPU needed).
# This import is kept only for backward compatibility with any external integrations.
AUTOGLUON_TEXT_AVAILABLE = False
try:
    from autogluon.multimodal import MultiModalPredictor
    AUTOGLUON_TEXT_AVAILABLE = True
except ImportError:
    pass  # Not required — text/NLP uses sentence-transformers now

# ── Detect GPU / CUDA availability ─────────────────────────────────
CUDA_AVAILABLE = False
try:
    import torch
    # Optimize Tensor Core usage on GPUs that support it (Ampere+, RTX 30xx+)
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass  # Not supported on this GPU architecture
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"BİLGİ: GPU algılandı — {torch.cuda.get_device_name(0)}")
    else:
        print("BİLGİ: CUDA bulunamadı. MultiModal modelleri CPU üzerinde çalışacak.")
except (ImportError, RuntimeError):
    print("BİLGİ: PyTorch yüklü değil veya CUDA başlatılamadı.")

# ── CVE-2025-32434 Workaround ────────────────────────────────────
# HuggingFace transformers (used by AutoGluon MultiModal) added a
# version gate in check_torch_load_is_safe() that raises ValueError
# BEFORE torch.load is even called when torch < 2.6.  This kills
# both training and prediction for all text/sentiment models because
# AutoModel.from_pretrained() → load_state_dict() → check → crash.
#
# We neutralise that gate AND patch torch.load as a safety net.
# This is safe because:
#   • Pretrained weights come from the trusted HuggingFace Hub.
#   • Fine-tuned checkpoints are created by this application itself.
#   • No user-uploaded .pt/.pth files are ever loaded via torch.load.
#
# The proper long-term fix is:  pip install torch>=2.6.0
# ──────────────────────────────────────────────────────────────────
_cve_patched = False
try:
    # Patch 1: Neutralise the transformers pre-check that blocks torch.load
    import transformers.utils.import_utils as _tf_import_utils
    if hasattr(_tf_import_utils, "check_torch_load_is_safe"):
        _tf_import_utils.check_torch_load_is_safe = lambda: None
        _cve_patched = True
except (ImportError, AttributeError):
    pass
try:
    # The function is also imported directly in modeling_utils — patch that copy too
    import transformers.modeling_utils as _tf_modeling_utils
    if hasattr(_tf_modeling_utils, "check_torch_load_is_safe"):
        _tf_modeling_utils.check_torch_load_is_safe = lambda: None
        _cve_patched = True
except (ImportError, AttributeError):
    pass
try:
    # Patch 2: Safety net — catch any remaining torch.load CVE RuntimeErrors
    import torch as _torch_module
    _original_torch_load = _torch_module.load

    def _patched_torch_load(*args, **kwargs):
        try:
            return _original_torch_load(*args, **kwargs)
        except RuntimeError as e:
            if "CVE-2025-32434" in str(e):
                kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)
            raise

    _torch_module.load = _patched_torch_load
    _cve_patched = True
except ImportError:
    pass
if _cve_patched:
    print("BİLGİ: torch.load CVE-2025-32434 uyumluluğu etkinleştirildi "
          "(torch>=2.6.0'a yükseltme önerilir)")

# ── Windows NVML Fix ─────────────────────────────────────────────
# AutoGluon MultiModal calls pynvml.nvmlInit() during training to
# log GPU info.  On Windows, pynvml only looks for nvml.dll in
# "C:\Program Files\NVIDIA Corporation\NVSMI\" — but modern NVIDIA
# drivers install it to "C:\Windows\System32\" instead.  When the
# DLL is not found, pynvml raises NVMLError_LibraryNotFound, which
# is uncaught by AutoGluon and kills the entire training process
# for a purely informational log message.
#
# Fix: (1) help pynvml find the DLL from common Windows locations,
#      (2) if still not found, make nvmlInit a silent no-op so
#          training proceeds without GPU logging.
# ──────────────────────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        import pynvml as _pynvml
        _original_nvml_init = _pynvml.nvmlInit

        def _safe_nvml_init():
            """Try the original init; on failure, search alternate DLL paths."""
            import ctypes
            try:
                return _original_nvml_init()
            except _pynvml.NVMLError:
                pass  # DLL not found at default path — try alternatives

            # Search common Windows locations for nvml.dll
            _nvml_search_paths = [
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                             "System32", "nvml.dll"),
                os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                             "NVIDIA Corporation", "GDK", "nvml.dll"),
            ]
            for dll_path in _nvml_search_paths:
                if os.path.isfile(dll_path):
                    try:
                        _pynvml.nvmlLib = ctypes.CDLL(dll_path)
                        _pynvml.nvmlLib.nvmlInit_v2.restype = ctypes.c_int
                        ret = _pynvml.nvmlLib.nvmlInit_v2()
                        if ret == 0:  # NVML_SUCCESS
                            return
                    except (OSError, AttributeError, Exception):
                        continue
            # DLL genuinely not available — raise so callers know
            raise _pynvml.NVMLError(_pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)

        _pynvml.nvmlInit = _safe_nvml_init
        print("BİLGİ: pynvml NVML Windows uyumluluğu etkinleştirildi")
    except ImportError:
        pass  # pynvml not installed — nothing to patch

# ── Make AutoGluon's GPU logging fault-tolerant ──────────────────
# AutoGluon MultiModal calls log_gpu_info() → get_gpu_message() →
# pynvml.nvmlInit() during training.  If pynvml fails for ANY reason
# (missing DLL, driver mismatch, permissions), the unhandled exception
# kills the entire training pipeline — for a purely informational log.
# Wrap it so training always proceeds.
# ──────────────────────────────────────────────────────────────────
try:
    import autogluon.multimodal.utils.log as _ag_log_module
    if hasattr(_ag_log_module, "get_gpu_message"):
        _original_get_gpu_message = _ag_log_module.get_gpu_message

        def _safe_get_gpu_message(*args, **kwargs):
            try:
                return _original_get_gpu_message(*args, **kwargs)
            except Exception:
                return "GPU bilgisi alınamadı (pynvml hatası — eğitim etkilenmez)"

        _ag_log_module.get_gpu_message = _safe_get_gpu_message
except (ImportError, AttributeError):
    pass

# ── Faster-Whisper (speech-to-text) ────────────────────────────────
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    print("BİLGİ: faster-whisper yüklü değil. Ses transkripsiyonu devre dışı.")
    print("Yüklemek için: pip install faster-whisper")

# ── Requests (for llama.cpp API calls) ─────────────────────────────
REQUESTS_AVAILABLE = False
try:
    import requests as http_requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("BİLGİ: requests yüklü değil. LLM API çağrıları devre dışı.")
    print("Yüklemek için: pip install requests")

from concurrent.futures import ThreadPoolExecutor

# ── Sentence-Transformers (text embeddings for NLP without heavy transformer training) ──
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("BİLGİ: sentence-transformers yüklü değil. Metin/NLP (embedding modu) devre dışı.")
    print("Yüklemek için: pip install sentence-transformers")

# Global sentence-transformer model cache (loaded once, reused across trainings)
_sentence_model_cache = {}  # {model_name: {"model": SentenceTransformer, "last_used": float, "measured_vram_mb": int}}
_sentence_model_lock = threading.Condition(threading.Lock())

# Prediction concurrency limiter (Solution 6: prevent concurrent VRAM explosion)
_prediction_semaphore = threading.Semaphore(3)  # Max 3 concurrent predictions
# Upload concurrency limiter — caps peak RAM from concurrent large body reads
_upload_semaphore = threading.Semaphore(5)  # Max 5 concurrent large uploads (5 × 200MB = 1GB peak)
_UPLOAD_SEMAPHORE_THRESHOLD = 10 * 1024 * 1024  # 10MB — only gate large bodies

# Default embedding model: lightweight, fast, 384 dimensions, good multilingual support
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDING_CACHE_TTL = int(os.environ.get("EMBEDDING_CACHE_TTL", "1800"))  # 30 min default

_sentence_model_loading = set()  # track models currently being loaded

def _get_sentence_model(model_name: str = None):
    """Load (or retrieve cached) sentence-transformer model. Thread-safe."""
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    with _sentence_model_lock:
        if model_name in _sentence_model_cache:
            _sentence_model_cache[model_name]["last_used"] = time.time()
            return _sentence_model_cache[model_name]["model"]
        # Prevent double-load: wait if another thread is already loading this model
        _wait_start = time.time()
        while model_name in _sentence_model_loading:
            # Condition.wait() atomically releases and re-acquires the lock
            remaining = 120 - (time.time() - _wait_start)
            if remaining <= 0:
                raise RuntimeError(f"Embedding model '{model_name}' yükleme zaman aşımı (120s)")
            _sentence_model_lock.wait(timeout=min(remaining, 0.5))
            if model_name in _sentence_model_cache:
                _sentence_model_cache[model_name]["last_used"] = time.time()
                return _sentence_model_cache[model_name]["model"]
        _sentence_model_loading.add(model_name)
    # Load outside lock to avoid blocking other threads
    try:
        print(f"[Embedding] Loading sentence-transformer model: {model_name}")
        # Measure VRAM delta during load (Solution 5)
        _emb_before = resource_manager.get_actual_free_vram_mb()
        model = SentenceTransformer(model_name)
        _emb_after = resource_manager.get_actual_free_vram_mb()
        _emb_measured = max(0, _emb_before - _emb_after)
        with _sentence_model_lock:
            _sentence_model_cache[model_name] = {
                "model": model, "last_used": time.time(),
                "measured_vram_mb": _emb_measured,
            }
            _sentence_model_loading.discard(model_name)
            _sentence_model_lock.notify_all()
        print(f"[Embedding] Model loaded: {model_name} (dim={model.get_sentence_embedding_dimension()}, VRAM: {_emb_measured}MB)")
        return model
    except Exception:
        with _sentence_model_lock:
            _sentence_model_loading.discard(model_name)
            _sentence_model_lock.notify_all()
        raise


def _embedding_cache_evictor():
    """Background thread to evict idle sentence-transformer models."""
    while True:
        time.sleep(60)
        now = time.time()
        with _sentence_model_lock:
            to_evict = [name for name, entry in _sentence_model_cache.items()
                       if (now - entry.get("last_used", now)) > EMBEDDING_CACHE_TTL]
            for name in to_evict:
                del _sentence_model_cache[name]
                log.info(f"[Embedding] Evicted idle model: {name}")
        if to_evict:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass

threading.Thread(target=_embedding_cache_evictor, daemon=True, name="embedding-evictor").start()


def _embed_text_columns(df, text_columns, model_name=None, batch_size=None, show_progress=True):
    if batch_size is None:
        # Scale batch size to available RAM (each batch ~100MB for 384-dim model)
        batch_size = min(256, max(16, resource_manager.safe_ram_mb // 100))
    """Convert text columns into dense embedding features using a sentence-transformer.

    For each text column, generates N embedding dimensions (e.g. 384 for MiniLM).
    Returns a new DataFrame with embedding columns replacing the original text columns.
    """
    import numpy as np
    model = _get_sentence_model(model_name)
    embed_dim = model.get_sentence_embedding_dimension()

    result_df = df.copy()

    for col in text_columns:
        texts = result_df[col].fillna("").astype(str).tolist()
        if show_progress:
            print(f"[Embedding] Encoding {len(texts)} texts from column '{col}' (dim={embed_dim})...")

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Create embedding columns: text_col_emb_0, text_col_emb_1, ...
        emb_col_names = [f"{col}_emb_{i}" for i in range(embed_dim)]
        emb_df = pd.DataFrame(embeddings, columns=emb_col_names, index=result_df.index)
        result_df = pd.concat([result_df, emb_df], axis=1)

        # Drop the original text column (tree models can't use raw text)
        result_df = result_df.drop(columns=[col])

    return result_df


def _save_text_pipeline_config(model_id, text_columns, embedding_model_name):
    """Save the text preprocessing pipeline configuration alongside the model."""
    config = {
        "pipeline_type": "sentence_embeddings",
        "text_columns": text_columns,
        "embedding_model": embedding_model_name,
    }
    config_path = MODELS_DIR / model_id / "text_pipeline.json"
    _atomic_write_json(config_path, config)
    print(f"[Embedding] Pipeline config saved: {config_path}")


def _load_text_pipeline_config(model_id):
    """Load the text preprocessing pipeline configuration."""
    config_path = MODELS_DIR / model_id / "text_pipeline.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning(f"[TextPipeline] Corrupted config: {config_path}")
        return None


# ── Configuration ──────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
MODELS_DIR = DATA_DIR / "models"
ACTIVITY_FILE = DATA_DIR / "activity.json"
USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"

# Critical files that must NEVER be silently replaced with empty defaults
_CRITICAL_JSON_FILES = {USERS_FILE, SESSIONS_FILE}

# Ensure directories exist
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "temp").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "stt").mkdir(parents=True, exist_ok=True)
except PermissionError:
    uid = getattr(os, 'getuid', lambda: 'unknown')()
    log.error(f"Cannot create data directories at {DATA_DIR}. "
              f"Check that the current user (uid={uid}) has write permissions.")
    sys.exit(1)

# ── Request size limits ───────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_AUDIO_FILE_SIZE_MB = int(os.environ.get("MAX_AUDIO_FILE_SIZE_MB", "200"))
MAX_AUDIO_FILE_SIZE_BYTES = MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "100000"))
MAX_PREDICTION_LENGTH = int(os.environ.get("MAX_PREDICTION_LENGTH", "500"))

# ── Session configuration ────────────────────────────────────────
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(8 * 3600)))
SESSION_IDLE_TIMEOUT = int(os.environ.get("SESSION_IDLE_TIMEOUT", str(2 * 3600)))

# ── Per-user model quota ─────────────────────────────────────────
MAX_MODELS_PER_USER = int(os.environ.get("MAX_MODELS_PER_USER", "50"))

# ── Trusted reverse-proxy IPs (for X-Forwarded-For) ──────────────
# Only trust X-Forwarded-For when the direct peer IP is in this set.
# Set to comma-separated IPs, e.g. "127.0.0.1,10.0.0.1,172.17.0.1"
_TRUSTED_PROXY_IPS_RAW = os.environ.get("TRUSTED_PROXY_IPS", "")
_TRUSTED_PROXY_IPS = (
    set(ip.strip() for ip in _TRUSTED_PROXY_IPS_RAW.split(",") if ip.strip())
    if _TRUSTED_PROXY_IPS_RAW else set()
)

# ══════════════════════════════════════════════════════════════════
#  THREAD-SAFE FILE I/O
# ══════════════════════════════════════════════════════════════════

# Per-file locks to prevent concurrent read-modify-write corruption
# Using RLock so callers can hold the lock across a full load→modify→save cycle
# while the inner load/save functions also acquire it.
_file_locks = {
    "users": threading.RLock(),
    "sessions": threading.RLock(),
    "activity": threading.RLock(),
    "model_meta": threading.RLock(),
}

_per_model_locks = {}
_per_model_locks_guard = threading.Lock()

def _get_model_lock(model_id: str) -> threading.RLock:
    """Get or create a per-model RLock. Returns a throwaway lock for non-existent models
    to prevent unbounded dict growth from requests to invalid model IDs."""
    with _per_model_locks_guard:
        if model_id in _per_model_locks:
            return _per_model_locks[model_id]
        if not (MODELS_DIR / model_id).is_dir():
            return threading.RLock()  # throwaway, not cached
        _per_model_locks[model_id] = threading.RLock()
        return _per_model_locks[model_id]


def _atomic_write_json(filepath: Path, data, use_safe_json=False):
    """Write JSON atomically: write to .tmp then os.replace.
    Prevents corruption if the process is killed mid-write."""
    tmp_path = filepath.with_suffix(".json.tmp")
    try:
        content = safe_json_dumps(data, indent=2) if use_safe_json else json.dumps(data, indent=2)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(filepath))
    except Exception:
        # Clean up temp file on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_read_json(filepath: Path, default=None):
    """Read JSON with error recovery. Returns default if file missing or corrupt.
    On corruption of CRITICAL files (users, sessions), raises instead of returning
    empty default to prevent overwriting the database with empty data.
    On corruption of non-critical files, creates backup and returns default."""
    if not filepath.exists():
        return default if default is not None else []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Create a backup of the corrupt file
        try:
            backup_name = f"{filepath.stem}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}{filepath.suffix}"
            backup_path = filepath.parent / backup_name
            shutil.copy2(str(filepath), str(backup_path))
            log.error(f"JSON CORRUPT: {filepath} — backup saved to {backup_path}. Error: {e}")
        except OSError as backup_err:
            log.error(f"JSON CORRUPT: {filepath} — backup failed ({backup_err}). Error: {e}")
        # For critical state files, refuse to return empty — prevents data wipeout
        if _CRITICAL_JSON_FILES and filepath in _CRITICAL_JSON_FILES:
            raise RuntimeError(
                f"Critical file corrupt: {filepath}. Refusing to return empty default "
                f"to prevent data loss. Backup saved. Manual recovery required."
            ) from e
        return default if default is not None else []


# ══════════════════════════════════════════════════════════════════
#  RESOURCE MANAGER — GPU/CPU/RAM tracking for safe concurrency
# ══════════════════════════════════════════════════════════════════

class ResourceManager:
    """Singleton that tracks hardware resources and gates concurrent tasks.

    Auto-detects GPU VRAM, system RAM, and CPU count on init.
    Applies a safety margin (default 10%) to prevent OOM.
    Tasks must acquire resources before starting and release when done.
    Scales automatically — same code works on 4GB and 24GB GPU.
    """

    def __init__(self, vram_safety_margin: float = 0.10, ram_safety_margin: float = 0.15):
        self._lock = threading.Lock()
        self._vram_safety = vram_safety_margin
        self._ram_safety = ram_safety_margin

        # ── Detect hardware ──
        self.total_vram_mb = 0
        self.gpu_name = "No GPU"
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                # PyTorch 2.4: total_global_mem, PyTorch 2.6+: total_memory
                raw_vram = getattr(props, 'total_global_mem',
                                   getattr(props, 'total_memory', 0))
                self.total_vram_mb = raw_vram // (1024 * 1024) if raw_vram else 0
                self.gpu_name = props.name
        except (ImportError, Exception) as e:
            log.info(f"GPU detection skipped: {e}")

        self.total_ram_mb = 0
        try:
            import psutil
            self.total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            self.total_ram_mb = int(line.split()[1]) // 1024
                            break
            except (OSError, ValueError):
                self.total_ram_mb = 8192

        # Apply Docker/cgroup memory limit (psutil and /proc/meminfo return host RAM)
        _cgroup_ram_applied = False
        try:
            # cgroup v2
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                if val != "max":
                    cgroup_ram = int(val) // (1024 * 1024)
                    if cgroup_ram > 0:
                        self.total_ram_mb = min(self.total_ram_mb, cgroup_ram)
                        _cgroup_ram_applied = True
        except (OSError, ValueError):
            try:
                # cgroup v1
                with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                    limit = int(f.read().strip())
                    if limit < 9 * 10**18:  # not "unlimited" sentinel
                        cgroup_ram = limit // (1024 * 1024)
                        if cgroup_ram > 0:
                            self.total_ram_mb = min(self.total_ram_mb, cgroup_ram)
                            _cgroup_ram_applied = True
            except (OSError, ValueError):
                pass

        # Detect effective CPU count (respects Docker/cgroup limits)
        _host_cpu_count = os.cpu_count() or 4
        self.cpu_count = _host_cpu_count
        _cgroup_cpu_applied = False
        try:
            # cgroup v2
            with open("/sys/fs/cgroup/cpu.max") as f:
                parts = f.read().strip().split()
                if parts[0] != "max":
                    quota, period = int(parts[0]), int(parts[1])
                    cgroup_cpus = max(1, quota // period)
                    self.cpu_count = min(self.cpu_count, cgroup_cpus)
                    _cgroup_cpu_applied = True
        except (OSError, ValueError, IndexError):
            try:
                # cgroup v1
                with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, \
                     open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
                    quota = int(fq.read().strip())
                    period = int(fp.read().strip())
                    if quota > 0:
                        cgroup_cpus = max(1, quota // period)
                        self.cpu_count = min(self.cpu_count, cgroup_cpus)
                        _cgroup_cpu_applied = True
            except (OSError, ValueError):
                pass

        if not _cgroup_cpu_applied and _host_cpu_count > 16:
            self.cpu_count = min(_host_cpu_count, 16)  # Hard cap when cgroup detection fails
            log.warning(f"[ResourceManager] Could not read cgroup CPU limits — capping to {self.cpu_count} "
                        f"(host has {_host_cpu_count}). Set TRAINING_CPU_COUNT env var to override.")
        # Allow manual override via env var (essential for vast.ai where cgroups may lie)
        _env_ram = os.environ.get("MAX_RAM_MB")
        if _env_ram:
            self.total_ram_mb = int(_env_ram)
            log.info(f"[ResourceManager] RAM override from MAX_RAM_MB env: {self.total_ram_mb}MB")
        elif not _cgroup_ram_applied and self.total_ram_mb > 32000:
            log.warning(f"[ResourceManager] Could not read cgroup memory limits — using host RAM ({self.total_ram_mb}MB). "
                        f"Set MAX_RAM_MB env var to override.")

        # Reserve cores for HTTP server + system — training gets the rest
        self.training_cpu_count = max(1, self.cpu_count - 2)

        # ── Compute safe budgets ──
        self.safe_vram_mb = int(self.total_vram_mb * (1.0 - self._vram_safety))
        self.safe_ram_mb = int(self.total_ram_mb * (1.0 - self._ram_safety))

        # ── Current reservations ──
        self._vram_reserved_mb = 0
        self._ram_reserved_mb = 0
        self._active_tasks = {}  # task_id -> {vram_mb, ram_mb, type, username, started_at}

        # ── Known resource profiles (approximate, in MB) ──
        self.PROFILES = {
            "whisper_gpu": {"vram_mb": 1500, "ram_mb": 500},
            "whisper_cpu": {"vram_mb": 0, "ram_mb": 2000},
            "llm_external": {"vram_mb": 2500, "ram_mb": 500},  # llama.cpp with CUDA (32 layers full model ≈ 2.5GB)
            "training_tabular": {"vram_mb": 0, "ram_mb": 4000},  # num_gpus=0 → CPU only; no VRAM needed
            "training_neural": {"vram_mb": 3000, "ram_mb": 4000},
            "prediction_light": {"vram_mb": 0, "ram_mb": 500},
            "prediction_model_load": {"vram_mb": 500, "ram_mb": 1000},
            # Audio pipeline per-job overhead (whisper VRAM managed separately by _get_whisper_model)
            "audio_pipeline": {"vram_mb": 0, "ram_mb": 800},
        }

        if self.total_vram_mb > 0:
            log.info(f"[ResourceManager] GPU: {self.gpu_name} ({self.total_vram_mb}MB total, "
                     f"{self.safe_vram_mb}MB safe budget)")
        else:
            log.info(f"[ResourceManager] GPU: not detected by torch — VRAM tracking disabled. "
                     f"(Whisper/ctranslate2 may still use GPU independently)")
        log.info(f"[ResourceManager] RAM: {self.total_ram_mb}MB total, "
                 f"{self.safe_ram_mb}MB safe budget, CPUs: {self.cpu_count}")

    @staticmethod
    def get_actual_free_vram_mb() -> int:
        """Get actual free GPU VRAM in MB using CUDA driver-level query.
        Accounts for ALL consumers: PyTorch, CTranslate2, llama.cpp subprocess."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                return free // (1024 * 1024)
        except Exception:
            pass
        return 0

    def can_acquire(self, vram_mb: int = 0, ram_mb: int = 0) -> bool:
        """Check if resources are available without actually reserving them."""
        with self._lock:
            vram_ok = True if self.total_vram_mb == 0 else \
                      (self._vram_reserved_mb + vram_mb) <= self.safe_vram_mb
            ram_ok = (self._ram_reserved_mb + ram_mb) <= self.safe_ram_mb
            return vram_ok and ram_ok

    def try_acquire(self, task_id: str, task_type: str, username: str = "",
                    vram_mb: int = 0, ram_mb: int = 0) -> bool:
        """Try to reserve resources. Returns True if successful, False if insufficient.

        Two-layer check:
        1. Bookkeeping: Does the reservation budget allow it?
        2. Reality: Does the GPU actually have enough free VRAM?
        Both must pass. This prevents OOM even if profiles underestimate usage.
        """
        with self._lock:
            effective_vram = vram_mb if self.total_vram_mb > 0 else 0

            # Layer 1: Bookkeeping check (reservation budget)
            if effective_vram > 0 and (self._vram_reserved_mb + effective_vram) > self.safe_vram_mb:
                log.warning(f"[ResourceManager] VRAM budget insufficient for {task_type}: "
                           f"need {effective_vram}MB, budget available {self.safe_vram_mb - self._vram_reserved_mb}MB")
                return False

            # Layer 2: Reality check (actual GPU free memory)
            if effective_vram > 0:
                actual_free = self.get_actual_free_vram_mb()
                # Require at least the requested VRAM + 500MB safety buffer
                if actual_free > 0 and actual_free < (effective_vram + 500):
                    log.warning(f"[ResourceManager] Actual GPU VRAM insufficient for {task_type}: "
                               f"need {effective_vram}+500MB safety, actual free {actual_free}MB")
                    return False
            if (self._ram_reserved_mb + ram_mb) > self.safe_ram_mb:
                log.warning(f"[ResourceManager] RAM insufficient for {task_type}: "
                           f"need {ram_mb}MB, available {self.safe_ram_mb - self._ram_reserved_mb}MB")
                return False

            self._vram_reserved_mb += effective_vram
            self._ram_reserved_mb += ram_mb
            self._active_tasks[task_id] = {
                "vram_mb": effective_vram, "ram_mb": ram_mb,
                "type": task_type, "username": username,
                "started_at": time.time(),
            }
            log.info(f"[ResourceManager] Acquired: {task_type} ({effective_vram}MB VRAM, {ram_mb}MB RAM) "
                     f"by {username or 'system'}. Total reserved: {self._vram_reserved_mb}MB VRAM, "
                     f"{self._ram_reserved_mb}MB RAM"
                     + (f" [VRAM tracking disabled — no GPU detected]" if self.total_vram_mb == 0 and vram_mb > 0 else ""))
            return True

    def release(self, task_id: str):
        """Release resources held by a task."""
        with self._lock:
            task = self._active_tasks.pop(task_id, None)
            if task:
                self._vram_reserved_mb = max(0, self._vram_reserved_mb - task["vram_mb"])
                self._ram_reserved_mb = max(0, self._ram_reserved_mb - task["ram_mb"])
                elapsed = time.time() - task["started_at"]
                log.info(f"[ResourceManager] Released: {task['type']} ({task['vram_mb']}MB VRAM, "
                         f"{task['ram_mb']}MB RAM) after {elapsed:.1f}s. "
                         f"Remaining: {self._vram_reserved_mb}MB VRAM, {self._ram_reserved_mb}MB RAM")

    def get_status(self) -> dict:
        """Return current resource utilization for monitoring."""
        with self._lock:
            return {
                "gpu": self.gpu_name,
                "vram_total_mb": self.total_vram_mb,
                "vram_safe_mb": self.safe_vram_mb,
                "vram_reserved_mb": self._vram_reserved_mb,
                "vram_available_mb": max(0, self.safe_vram_mb - self._vram_reserved_mb),
                "ram_total_mb": self.total_ram_mb,
                "ram_safe_mb": self.safe_ram_mb,
                "ram_reserved_mb": self._ram_reserved_mb,
                "ram_available_mb": max(0, self.safe_ram_mb - self._ram_reserved_mb),
                "cpu_count": self.cpu_count,
                "active_tasks": {k: {**v, "elapsed_sec": round(time.time() - v["started_at"], 1)}
                                 for k, v in self._active_tasks.items()},
            }

    def get_profile(self, profile_name: str) -> dict:
        """Get resource requirements for a known task profile."""
        return self.PROFILES.get(profile_name, {"vram_mb": 0, "ram_mb": 500})


# Global singleton — initialized at module load
resource_manager = ResourceManager()


_vram_eviction_lock = threading.Lock()

def _ensure_vram_available(needed_mb: int) -> bool:
    """Solution 7: Pre-emptive eviction — free VRAM BEFORE an operation, not after it fails.
    Returns True if enough VRAM is available after eviction attempts.
    Serialized to prevent multiple threads from racing to evict and overcommit."""
    with _vram_eviction_lock:
        actual_free = resource_manager.get_actual_free_vram_mb()
        if actual_free <= 0 or actual_free >= needed_mb:
            return True  # Either no GPU detected or enough free

        log.info(f"[VRAM] Need {needed_mb}MB but only {actual_free}MB free — pre-emptively evicting...")

        # Priority 1: Evict idle cached prediction models (cheapest to reload)
        try:
            model_cache._evict_idle(min_idle_seconds=60)
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        actual_free = resource_manager.get_actual_free_vram_mb()
        if actual_free >= needed_mb:
            return True

        # Priority 2: Evict idle embedding models
        try:
            now = time.time()
            with _sentence_model_lock:
                to_evict = [n for n, e in _sentence_model_cache.items()
                           if (now - e.get("last_used", now)) > 30]
                for name in to_evict:
                    log.info(f"[VRAM] Pre-evicted embedding model: {name}")
                    del _sentence_model_cache[name]
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        actual_free = resource_manager.get_actual_free_vram_mb()
        if actual_free >= needed_mb:
            return True

        # Priority 3: Unload idle Whisper (frees most VRAM but expensive to reload)
        # NOTE: We check idle time and set a flag here, but call _unload_whisper_model()
        # OUTSIDE _vram_eviction_lock to avoid deadlock with _get_whisper_model() which
        # acquires _whisper_model_lock -> _vram_eviction_lock (opposite order).
        _should_evict_whisper = False
        try:
            if _whisper_model is not None:
                idle = time.time() - _whisper_last_used  # float read is atomic in CPython
                if idle > 10:
                    _should_evict_whisper = True
        except Exception:
            pass

    # Outside _vram_eviction_lock — safe to acquire _whisper_model_lock
    if _should_evict_whisper:
        try:
            log.info(f"[VRAM] Pre-evicting Whisper to free VRAM")
            _unload_whisper_model(min_idle_seconds=10)
        except Exception:
            pass

    with _vram_eviction_lock:
        actual_free = resource_manager.get_actual_free_vram_mb()
        return actual_free >= needed_mb


# ══════════════════════════════════════════════════════════════════
#  USER ACTION TRACKER — per-user concurrency limits
# ══════════════════════════════════════════════════════════════════

class UserActionTracker:
    """Tracks active jobs per user to enforce concurrency limits.

    Limits:
    - Max 1 training job per user at a time
    - Max 1 audio evaluation per user at a time
    - Max 1 audio prediction per user at a time
    - Predictions (tabular) are unlimited (lightweight)
    """

    def __init__(self):
        self._lock = threading.Lock()
        # {username: {"training": job_id, "audio_eval": job_id, "audio_predict": job_id}}
        self._active = {}

    def can_start(self, username: str, action_type: str) -> tuple:
        """Check if user can start an action. Returns (allowed: bool, reason: str)."""
        if action_type not in ("training", "audio_eval", "audio_predict"):
            return True, ""
        with self._lock:
            user_actions = self._active.get(username, {})
            if action_type in user_actions:
                existing_job = user_actions[action_type]
                labels = {
                    "training": "eğitim işi",
                    "audio_eval": "ses değerlendirmesi",
                    "audio_predict": "ses tahmini",
                }
                label = labels.get(action_type, action_type)
                return False, f"Zaten aktif bir {label} çalışıyor (iş: {existing_job[:8]}…). Lütfen tamamlanmasını bekleyin."
            return True, ""

    def try_register(self, username: str, action_type: str, job_id: str) -> tuple:
        """Atomically check + register. Returns (allowed: bool, reason: str).
        Prevents TOCTOU race where two requests both pass can_start before either registers."""
        if action_type not in ("training", "audio_eval", "audio_predict"):
            return True, ""
        with self._lock:
            user_actions = self._active.get(username, {})
            if action_type in user_actions:
                existing_job = user_actions[action_type]
                labels = {
                    "training": "eğitim işi",
                    "audio_eval": "ses değerlendirmesi",
                    "audio_predict": "ses tahmini",
                }
                label = labels.get(action_type, action_type)
                return False, f"Zaten aktif bir {label} çalışıyor (iş: {existing_job[:8]}…). Lütfen tamamlanmasını bekleyin."
            if username not in self._active:
                self._active[username] = {}
            self._active[username][action_type] = job_id
            log.info(f"[UserTracker] {username} started {action_type}: {job_id[:8]}…")
            return True, ""

    def register(self, username: str, action_type: str, job_id: str):
        """Register a new active job for a user. Prefer try_register() for atomic check+register."""
        with self._lock:
            if username not in self._active:
                self._active[username] = {}
            self._active[username][action_type] = job_id
            log.info(f"[UserTracker] {username} started {action_type}: {job_id[:8]}…")

    def unregister(self, username: str, action_type: str):
        """Remove an active job for a user."""
        with self._lock:
            if username in self._active:
                self._active[username].pop(action_type, None)
                if not self._active[username]:
                    del self._active[username]
                log.info(f"[UserTracker] {username} finished {action_type}")

    def get_active(self, username: str = None) -> dict:
        """Get active jobs, optionally filtered by user."""
        with self._lock:
            if username:
                return dict(self._active.get(username, {}))
            return {u: dict(a) for u, a in self._active.items()}


# Global singleton
user_action_tracker = UserActionTracker()


# ══════════════════════════════════════════════════════════════════
#  MODEL REFERENCE COUNTER — prevent deletion during use
# ══════════════════════════════════════════════════════════════════

class ModelRefCounter:
    """Track active references to models to prevent deletion during use.

    Ref count semantics:
      > 0  — number of active users (predictions, explainability, etc.)
        0  — idle (default; key absent from dict)
       -1  — marked for deletion; new acquire() calls will be rejected
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._refs = {}  # model_id -> int

    def acquire(self, model_id: str) -> bool:
        """Increment ref count. Returns False if model is being deleted."""
        with self._lock:
            current = self._refs.get(model_id, 0)
            if current < 0:
                return False
            self._refs[model_id] = current + 1
            return True

    def release(self, model_id: str):
        with self._lock:
            count = self._refs.get(model_id, 0)
            if count < 0:
                return  # deletion sentinel, don't touch
            count -= 1
            if count <= 0:
                self._refs.pop(model_id, None)
            else:
                self._refs[model_id] = count

    def is_busy(self, model_id: str) -> bool:
        with self._lock:
            return self._refs.get(model_id, 0) > 0

    def try_mark_for_deletion(self, model_id: str) -> bool:
        """Atomically check model is idle and mark it for deletion.
        Returns True if successfully marked, False if model is busy."""
        with self._lock:
            if self._refs.get(model_id, 0) > 0:
                return False
            self._refs[model_id] = -1
            return True

    def unmark_deletion(self, model_id: str):
        """Remove deletion sentinel (e.g. if deletion itself fails)."""
        with self._lock:
            if self._refs.get(model_id, 0) == -1:
                self._refs.pop(model_id, None)


model_ref_counter = ModelRefCounter()


# ══════════════════════════════════════════════════════════════════
#  JOB STORE — bounded, TTL-aware job tracking
# ══════════════════════════════════════════════════════════════════

class JobStore:
    """Thread-safe job status store with automatic TTL eviction.

    Supports dict-like access (job_store[id] = {...}, job_store.get(id))
    for backward compatibility while preventing unbounded memory growth.
    """

    def __init__(self, max_completed: int = 200, ttl_seconds: int = 7200):
        self._lock = threading.RLock()  # reentrant for nested access
        self._jobs = {}
        self._max_completed = max_completed
        self._ttl = ttl_seconds
        t = threading.Thread(target=self._cleanup_loop, daemon=True)
        t.start()

    def __setitem__(self, job_id, value):
        with self._lock:
            if isinstance(value, dict):
                value["_created_at"] = value.get("_created_at", time.time())
                value["_updated_at"] = time.time()
            self._jobs[job_id] = value

    def __getitem__(self, job_id):
        with self._lock:
            return self._jobs[job_id]

    def __contains__(self, job_id):
        with self._lock:
            return job_id in self._jobs

    def get(self, job_id, default=None):
        with self._lock:
            return self._jobs.get(job_id, default)

    def pop(self, job_id, *args):
        with self._lock:
            return self._jobs.pop(job_id, *args)

    def update_fields(self, job_id: str, **kwargs):
        """Safely update specific fields on a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.update(kwargs)
                job["_updated_at"] = time.time()

    def _cleanup_loop(self):
        """Periodically evict old completed/errored jobs."""
        while True:
            time.sleep(300)
            now = time.time()
            with self._lock:
                to_remove = [jid for jid, job in self._jobs.items()
                             if (now - job.get("_created_at", now)) > self._ttl
                             and job.get("status") in ("done", "error")]
                for jid in to_remove:
                    del self._jobs[jid]
                if to_remove:
                    log.info(f"[JobStore] Evicted {len(to_remove)} expired jobs")

                completed = [(jid, j) for jid, j in self._jobs.items()
                             if j.get("status") in ("done", "error")]
                if len(completed) > self._max_completed:
                    completed.sort(key=lambda x: x[1].get("_created_at", 0))
                    excess = len(completed) - self._max_completed
                    for jid, _ in completed[:excess]:
                        del self._jobs[jid]

                # Detect stale running jobs (no update in 12 minutes)
                HEARTBEAT_TIMEOUT = 720  # 12 minutes
                stale = [(jid, job) for jid, job in self._jobs.items()
                         if job.get("status") in ("training", "processing", "queued")
                         and (now - job.get("_updated_at", now)) > HEARTBEAT_TIMEOUT]
                for jid, job in stale:
                    log.warning(f"[JobStore] Heartbeat timeout: {jid[:8]}…")
                    # Unregister BEFORE marking error so concurrent submits
                    # don't see a stale registration and get rejected
                    uname = job.get("_username")
                    atype = job.get("_action_type")
                    if uname and atype:
                        user_action_tracker.unregister(uname, atype)
                    # Release resource reservations held by the stale job
                    _stale_res_id = job.get("_resource_task_id")
                    if _stale_res_id:
                        resource_manager.release(_stale_res_id)
                    _stale_model_id = job.get("_model_id")
                    if _stale_model_id:
                        model_ref_counter.release(_stale_model_id)
                    # Only release quota for jobs that actually reserved it
                    if uname and atype in ("training", "audio_eval"):
                        release_model_quota(uname)
                    job["status"] = "error"
                    job["error"] = "İş zaman aşımına uğradı (güncelleme alınamadı)"
                    job["_stale_released"] = True


# Global job stores (replace the old plain dicts)
training_jobs = JobStore(max_completed=100, ttl_seconds=7200)
audio_eval_jobs = JobStore(max_completed=100, ttl_seconds=7200)
audio_predict_jobs = JobStore(max_completed=200, ttl_seconds=3600)


# ══════════════════════════════════════════════════════════════════
#  TEMP FILE CLEANUP
# ══════════════════════════════════════════════════════════════════

def _startup_cleanup():
    """Clean up stale temp directories and orphan model directories
    from previous crashed sessions."""
    # ── Clean stale temp dirs (>24h old) ──
    temp_base = DATA_DIR / "temp"
    if temp_base.exists():
        now = time.time()
        cleaned = 0
        for d in temp_base.iterdir():
            if d.is_dir():
                try:
                    age_hours = (now - d.stat().st_mtime) / 3600
                    if age_hours > 24:
                        shutil.rmtree(d, ignore_errors=True)
                        cleaned += 1
                except OSError:
                    pass
        if cleaned:
            log.info(f"[Cleanup] Removed {cleaned} stale temp directories (>24h old)")

    # ── Clean orphan model dirs (have training_data.csv but no meta.json) ──
    if MODELS_DIR.exists():
        orphans = 0
        for d in MODELS_DIR.iterdir():
            if d.is_dir():
                meta_path = d / "meta.json"
                training_data_path = d / "training_data.csv"
                if not meta_path.exists() and training_data_path.exists():
                    try:
                        # Only clean if it's old enough (>1 hour, in case training is starting)
                        age_hours = (time.time() - d.stat().st_mtime) / 3600
                        if age_hours > 1:
                            shutil.rmtree(d, ignore_errors=True)
                            orphans += 1
                    except OSError:
                        pass
        if orphans:
            log.info(f"[Cleanup] Removed {orphans} orphan model directories (no meta.json)")


def _periodic_temp_cleanup():
    """Background thread: clean stale temp dirs every hour (survives segfaults)."""
    while True:
        time.sleep(3600)
        try:
            temp_base = DATA_DIR / "temp"
            if not temp_base.exists():
                continue
            now = time.time()
            cleaned = 0
            for d in temp_base.iterdir():
                if d.is_dir():
                    try:
                        if (now - d.stat().st_mtime) / 3600 > 2:
                            shutil.rmtree(d, ignore_errors=True)
                            cleaned += 1
                    except OSError:
                        pass
            if cleaned:
                log.info(f"[Cleanup] Periodic: removed {cleaned} stale temp dirs (>2h old)")
        except Exception:
            pass

threading.Thread(target=_periodic_temp_cleanup, daemon=True, name="temp-cleaner").start()


def _cleanup_temp_dir(temp_id: str):
    """Remove a specific temp directory after use."""
    try:
        temp_dir = DATA_DIR / "temp" / temp_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except OSError:
        pass

# ══════════════════════════════════════════════════════════════════
#  AUTHENTICATION & USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with PBKDF2-HMAC-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(32)
    h = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100_000)
    return h.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash."""
    computed, _ = _hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)


def _validate_password(password: str) -> str:
    """Validate password strength. Returns error message or empty string."""
    if len(password) > 128:
        return "Şifre en fazla 128 karakter olabilir"
    if len(password) < 8:
        return "Şifre en az 8 karakter olmalı"
    if not re.search(r'[A-Z]', password):
        return "Şifre en az 1 büyük harf içermeli"
    if not re.search(r'[a-z]', password):
        return "Şifre en az 1 küçük harf içermeli"
    if not re.search(r'[0-9]', password):
        return "Şifre en az 1 rakam içermeli"
    return ""


def load_users() -> list:
    """Load users from JSON (thread-safe)."""
    with _file_locks["users"]:
        return _safe_read_json(USERS_FILE, default=[])


def save_users(users: list):
    """Save users to JSON (thread-safe, atomic write)."""
    with _file_locks["users"]:
        _atomic_write_json(USERS_FILE, users)


def find_user(username: str) -> dict:
    """Find a user by username."""
    users = load_users()
    for u in users:
        if u["username"].lower() == username.lower():
            return u
    return None


class _SessionStore:
    """In-memory session store with periodic disk flush.
    Eliminates per-request disk I/O for session validation."""
    _FLUSH_INTERVAL = 30  # seconds between disk flushes
    _MAX_SESSIONS_PER_USER = 5

    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}  # {token: {username, created_at, last_activity}}
        self._dirty = False
        self._loaded = False

    def _ensure_loaded(self):
        """Load sessions from disk on first access."""
        if not self._loaded:
            self._data = _safe_read_json(SESSIONS_FILE, default={})
            self._loaded = True

    def _flush_to_disk(self):
        """Write current state to disk if changed."""
        if self._dirty:
            _atomic_write_json(SESSIONS_FILE, self._data)
            self._dirty = False

    def _clean_expired(self):
        """Remove expired sessions from memory."""
        now_dt = datetime.now()
        to_remove = []
        for token, v in self._data.items():
            try:
                created = datetime.fromisoformat(v.get("created_at", "")).replace(tzinfo=None)
                if (now_dt - created).total_seconds() > SESSION_TTL_SECONDS:
                    to_remove.append(token)
            except (ValueError, TypeError, KeyError):
                to_remove.append(token)
        for t in to_remove:
            del self._data[t]
        if to_remove:
            self._dirty = True

    def create(self, username: str) -> str:
        token = secrets.token_urlsafe(48)
        now = datetime.now().isoformat()
        with self._lock:
            self._ensure_loaded()
            self._clean_expired()
            # Limit sessions per user
            user_sessions = [(k, v) for k, v in self._data.items() if v.get("username") == username]
            if len(user_sessions) >= self._MAX_SESSIONS_PER_USER:
                user_sessions.sort(key=lambda x: x[1].get("created_at", ""))
                for old_token, _ in user_sessions[:len(user_sessions) - self._MAX_SESSIONS_PER_USER + 1]:
                    self._data.pop(old_token, None)
            self._data[token] = {"username": username, "created_at": now, "last_activity": now}
            self._dirty = True
            self._flush_to_disk()  # flush immediately on create (new session must survive crash)
        return token

    def get_user(self, token: str) -> dict:
        """Validate session and return user dict, or None."""
        if not token:
            return None
        with self._lock:
            self._ensure_loaded()
            session = self._data.get(token)
            if not session:
                return None
            try:
                now_dt = datetime.now()
                created = datetime.fromisoformat(session.get("created_at", "")).replace(tzinfo=None)
                if (now_dt - created).total_seconds() > SESSION_TTL_SECONDS:
                    del self._data[token]
                    self._dirty = True
                    return None
                last_activity_str = session.get("last_activity", session.get("created_at", ""))
                last_activity = datetime.fromisoformat(last_activity_str).replace(tzinfo=None)
                idle_seconds = (now_dt - last_activity).total_seconds()
                if idle_seconds > SESSION_IDLE_TIMEOUT:
                    del self._data[token]
                    self._dirty = True
                    return None
                if idle_seconds > 300:
                    session["last_activity"] = now_dt.isoformat()
                    self._dirty = True
                username = session["username"]
            except (ValueError, TypeError, KeyError):
                self._data.pop(token, None)
                self._dirty = True
                return None
        return find_user(username)

    def destroy(self, token: str):
        with self._lock:
            self._ensure_loaded()
            if self._data.pop(token, None) is not None:
                self._dirty = True
                self._flush_to_disk()

    def destroy_user_sessions(self, username: str, except_token: str = None):
        with self._lock:
            self._ensure_loaded()
            to_remove = [t for t, v in self._data.items()
                         if v.get("username") == username and t != except_token]
            for t in to_remove:
                del self._data[t]
            if to_remove:
                self._dirty = True
                self._flush_to_disk()
                log.info(f"[Auth] Destroyed {len(to_remove)} session(s) for user '{username}'")

    def periodic_flush(self):
        """Background thread: flush dirty state to disk periodically."""
        while True:
            time.sleep(self._FLUSH_INTERVAL)
            with self._lock:
                self._flush_to_disk()


_session_store = _SessionStore()
threading.Thread(target=_session_store.periodic_flush, daemon=True, name="session-flusher").start()


def load_sessions() -> dict:
    """Load sessions (for compatibility)."""
    with _session_store._lock:
        _session_store._ensure_loaded()
        return dict(_session_store._data)


def save_sessions(sessions: dict):
    """Save sessions (for compatibility)."""
    with _session_store._lock:
        _session_store._data = sessions
        _session_store._dirty = True
        _session_store._flush_to_disk()


def create_session(username: str) -> str:
    return _session_store.create(username)


def get_session_user(token: str) -> dict:
    return _session_store.get_user(token)


def destroy_session(token: str):
    _session_store.destroy(token)


def destroy_user_sessions(username: str, except_token: str = None):
    _session_store.destroy_user_sessions(username, except_token=except_token)


def init_master_admin():
    """Create master admin if no users exist.
    Reads ADMIN_USER and ADMIN_PASSWORD from environment.
    Falls back to admin / Admin123! for local development only."""
    users = load_users()
    if len(users) == 0:
        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_pass = os.environ.get("ADMIN_PASSWORD", "")
        if not admin_pass:
            admin_pass = "Admin123!"
            log.warning("ADMIN_PASSWORD ortam değişkeni ayarlanmamış — varsayılan şifre kullanılıyor!")
            log.warning("Üretim ortamında ADMIN_PASSWORD ayarlamayı unutmayın.")
        pwd_hash, salt = _hash_password(admin_pass)
        # SECURITY: display_name and email are immutable — no user-facing endpoint should modify them
        master = {
            "username": admin_user,
            "display_name": "Sistem Yöneticisi",
            "password_hash": pwd_hash,
            "salt": salt,
            "role": "master_admin",
            "created_at": datetime.now().isoformat(),
            "created_by": "system",
        }
        save_users([master])
        print("═══════════════════════════════════════════════════")
        print(f"  Ana yönetici hesabı oluşturuldu:")
        print(f"  Kullanıcı: {admin_user}")
        if admin_pass == "Admin123!":
            print("  Şifre: Admin123!")
            print("  ⚠  ÜRETİMDE ADMIN_PASSWORD ORTAM DEĞİŞKENİ AYARLAYIN!")
        else:
            print("  Şifre: (ADMIN_PASSWORD ortam değişkeninden alındı)")
        print("═══════════════════════════════════════════════════")


# init_master_admin() is called in main() to avoid side effects on import


# ── Login Rate Limiter (Tiered) ─────────────────────────────────
class LoginRateLimiter:
    """Tiered in-memory rate limiter for login attempts per IP/username.

    Tier 1: 5 attempts in 15 min window  -> block for 15 min
    Tier 2: 15 attempts in 60 min window -> block for 60 min
    """

    TIERS = [
        {"max_attempts": 5,  "window": 900,  "block": 900},   # 15 min
        {"max_attempts": 15, "window": 3600, "block": 3600},  # 60 min
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._attempts = {}  # key -> list of timestamps
        self._last_prune = time.time()

    def is_blocked(self, key: str) -> tuple:
        """Returns (blocked: bool, retry_after_seconds: int)."""
        with self._lock:
            now = time.time()
            attempts = self._attempts.get(key, [])
            # Prune entries older than the largest window (60 min)
            max_window = max(t["window"] for t in self.TIERS)
            attempts = [t for t in attempts if now - t < max_window]
            self._attempts[key] = attempts

            # Check tiers from strictest (highest) to lowest
            for tier in reversed(self.TIERS):
                window_attempts = [t for t in attempts if now - t < tier["window"]]
                if len(window_attempts) >= tier["max_attempts"]:
                    oldest_in_window = min(window_attempts)
                    retry_after = int(tier["block"] - (now - oldest_in_window))
                    if retry_after > 0:
                        return True, retry_after

            # Periodic prune of empty/stale keys (every 5 minutes)
            if now - self._last_prune > 300:
                self._last_prune = now
                max_window = max(t["window"] for t in self.TIERS)
                stale_keys = [k for k, v in self._attempts.items()
                              if not v or all(now - t > max_window for t in v)]
                for k in stale_keys:
                    del self._attempts[k]
                # Hard cap to prevent memory exhaustion under brute-force
                if len(self._attempts) > 10000:
                    oldest_keys = sorted(self._attempts.keys(),
                                          key=lambda k: max(self._attempts[k]) if self._attempts[k] else 0)
                    for k in oldest_keys[:len(self._attempts) - 10000]:
                        del self._attempts[k]

            return False, 0

    def record_attempt(self, key: str):
        with self._lock:
            now = time.time()
            if key not in self._attempts:
                self._attempts[key] = []
            self._attempts[key].append(now)

    def reset(self, key: str):
        with self._lock:
            self._attempts.pop(key, None)


_login_limiter = LoginRateLimiter()


class SimpleRateLimiter:
    """Simple sliding window rate limiter."""
    def __init__(self, max_attempts=3, window_seconds=3600):
        self._lock = threading.Lock()
        self._attempts = {}
        self._max = max_attempts
        self._window = window_seconds

    def is_blocked(self, key):
        with self._lock:
            now = time.time()
            attempts = [t for t in self._attempts.get(key, []) if now - t < self._window]
            self._attempts[key] = attempts
            return len(attempts) >= self._max

    def record_attempt(self, key):
        with self._lock:
            now = time.time()
            if key not in self._attempts:
                self._attempts[key] = []
            self._attempts[key].append(now)
            self._attempts[key] = [t for t in self._attempts[key] if now - t < self._window]
            if len(self._attempts) > 1000:
                empty = [k for k, v in self._attempts.items() if not v]
                for k in empty:
                    del self._attempts[k]

_registration_limiter = SimpleRateLimiter(max_attempts=3, window_seconds=3600)


class APIRateLimiter:
    """Per-user rate limiter: normal (60/min) and heavy (10/hour) operations."""
    def __init__(self, normal_max=60, normal_window=60, heavy_max=10, heavy_window=3600):
        self._lock = threading.Lock()
        self._normal = {}  # username -> [timestamps]
        self._heavy = {}   # username -> [timestamps]
        self._normal_max = normal_max
        self._normal_window = normal_window
        self._heavy_max = heavy_max
        self._heavy_window = heavy_window

    def check_normal(self, username):
        with self._lock:
            now = time.time()
            attempts = [t for t in self._normal.get(username, []) if now - t < self._normal_window]
            self._normal[username] = attempts
            if len(attempts) >= self._normal_max:
                return False
            attempts.append(now)
            self._normal[username] = attempts
            return True

    def check_heavy(self, username):
        with self._lock:
            now = time.time()
            attempts = [t for t in self._heavy.get(username, []) if now - t < self._heavy_window]
            self._heavy[username] = attempts
            if len(attempts) >= self._heavy_max:
                return False
            attempts.append(now)
            self._heavy[username] = attempts
            return True

_api_rate_limiter = APIRateLimiter()


# ══════════════════════════════════════════════════════════════════
#  MODEL COMPATIBILITY & SQL GENERATION
# ══════════════════════════════════════════════════════════════════

TREE_BASED_MODELS = {
    "XGBoost", "LightGBM", "RandomForest", "ExtraTrees",
    "LightGBMLarge", "LightGBMXT", "CatBoost",
    "RandomForestGini", "RandomForestEntr",
    "ExtraTreesGini", "ExtraTreesEntr",
    "XGBoost_BAG_L1", "XGBoost_BAG_L2",
    "LightGBM_BAG_L1", "LightGBM_BAG_L2",
    "LightGBMXT_BAG_L1", "LightGBMXT_BAG_L2",
    "CatBoost_BAG_L1", "CatBoost_BAG_L2",
    "RandomForest_BAG_L1", "RandomForest_BAG_L2",
    "ExtraTrees_BAG_L1", "ExtraTrees_BAG_L2",
}

LINEAR_MODELS = {
    "LinearModel", "LR", "LinearModel_BAG_L1", "LinearModel_BAG_L2",
}

NEURAL_MODELS = {
    "NeuralNetFastAI", "NeuralNetTorch", "TabularNeuralNetwork",
    "NeuralNetFastAI_BAG_L1", "NeuralNetTorch_BAG_L1",
    "NeuralNetFastAI_BAG_L2", "NeuralNetTorch_BAG_L2",
    "FASTAI", "NN_TORCH",
}

KNN_MODELS = {
    "KNeighbors", "KNeighborsDist", "KNeighborsUnif",
    "KNeighborsDist_BAG_L1", "KNeighborsUnif_BAG_L1",
    "KNeighborsDist_BAG_L2", "KNeighborsUnif_BAG_L2",
}

PRESETS = [
    {"value": "medium_quality", "label": "Orta Kalite (En Hızlı)"},
    {"value": "good_quality", "label": "İyi Kalite (Dengeli)"},
    {"value": "high_quality", "label": "Yüksek Kalite (Daha Yavaş)"},
    {"value": "best_quality", "label": "En İyi Kalite (En Yavaş)"},
]

# NOTE: training_jobs, audio_eval_jobs, audio_predict_jobs are now
# JobStore instances defined above with TTL eviction and bounded size.

# ── LLM / llama.cpp Configuration ─────────────────────────────────
LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://localhost:8081/v1/chat/completions")
LLAMA_CPP_MODEL = os.environ.get("LLAMA_CPP_MODEL", "local-model")

# ── Bundled llama.cpp server ─────────────────────────────────────
# LLAMA_BUNDLED="auto" (default): download & start llama-server automatically
# if nothing is already listening on LLAMA_PORT.  Binary + model are cached
# in DATA_DIR/llm/ (persists across restarts).
# Set "false" to disable, "true" to force start even if port is in use.
_llama_process = None
_llama_log_fh = None
_llama_lock = threading.Lock()
LLAMA_BUNDLED = os.environ.get("LLAMA_BUNDLED", "auto")
LLAMA_MODEL_REPO = os.environ.get("LLAMA_MODEL_REPO", "unsloth/Qwen3.5-4B-GGUF")
LLAMA_MODEL_FILE = os.environ.get("LLAMA_MODEL_FILE", "Qwen3.5-4B-Q4_K_M.gguf")
LLAMA_GPU_LAYERS = os.environ.get("LLAMA_GPU_LAYERS", "auto")  # "auto" = detect from VRAM; or set manually
LLAMA_CTX_SIZE = os.environ.get("LLAMA_CTX_SIZE", "8192")
LLAMA_BATCH_SIZE = os.environ.get("LLAMA_BATCH_SIZE", "128")
LLAMA_UBATCH_SIZE = os.environ.get("LLAMA_UBATCH_SIZE", "128")
LLAMA_PORT = os.environ.get("LLAMA_PORT", "8081")
LLAMA_DIR = DATA_DIR / "llm"


def _find_llama_server_bin():
    """Find llama-server binary: check LLAMA_DIR/release, then system PATH."""
    bin_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    release_dir = LLAMA_DIR / "release"
    if release_dir.exists():
        for root, _dirs, files in os.walk(str(release_dir)):
            if bin_name in files:
                p = os.path.join(root, bin_name)
                if os.access(p, os.X_OK) or sys.platform == "win32":
                    return p
    from shutil import which
    return which("llama-server")


def _download_llama_server():
    """Download llama-server binary from GitHub releases if not cached."""
    import urllib.request
    import tarfile
    import zipfile

    existing = _find_llama_server_bin()
    if existing:
        return existing

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("[LLM] llama-server not found — downloading from GitHub...")
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    req = urllib.request.Request(api_url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "TahminPlatformu"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        release = json.loads(resp.read())

    assets = release.get("assets", [])

    if sys.platform == "win32":
        patterns, ext = ["win-x64"], ".zip"
    elif sys.platform == "darwin":
        import platform as _plat
        arch = "arm64" if _plat.machine() == "arm64" else "x64"
        patterns, ext = [f"macos-{arch}"], ".tar.gz"
    else:
        # Download pre-compiled binary (CPU-only; no Linux CUDA binaries in releases since b8413+).
        patterns = ["ubuntu-x64"]
        ext = ".tar.gz"

    asset_url = asset_name = None
    for pattern in patterns:
        for asset in assets:
            name = asset["name"]
            if pattern in name and name.endswith(ext) and "vulkan" not in name:
                asset_url = asset["browser_download_url"]
                asset_name = name
                break
        if asset_url:
            break
    if not asset_url:
        raise RuntimeError(f"llama-server binary not found for: {patterns}")

    archive_path = LLAMA_DIR / asset_name
    log.info(f"[LLM] Downloading {asset_name}...")
    urllib.request.urlretrieve(asset_url, str(archive_path))

    extract_dir = LLAMA_DIR / "release"
    if extract_dir.exists():
        shutil.rmtree(str(extract_dir))
    log.info("[LLM] Extracting...")
    if asset_name.endswith(".tar.gz"):
        with tarfile.open(str(archive_path), "r:gz") as tf:
            # Validate tar members to prevent path traversal
            for member in tf.getmembers():
                member_path = os.path.normpath(os.path.join(str(extract_dir), member.name))
                if not member_path.startswith(str(extract_dir) + os.sep):
                    raise ValueError(f"Tar member '{member.name}' would extract outside target directory")
            tf.extractall(str(extract_dir))
    else:
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            # Validate zip members to prevent path traversal
            for member in zf.namelist():
                member_path = os.path.normpath(os.path.join(str(extract_dir), member))
                if not member_path.startswith(str(extract_dir) + os.sep):
                    raise ValueError(f"Zip member '{member}' would extract outside target directory")
            zf.extractall(str(extract_dir))
    archive_path.unlink(missing_ok=True)

    if sys.platform != "win32":
        for root, _dirs, files in os.walk(str(extract_dir)):
            for f in files:
                if f == "llama-server" or f.endswith((".so", ".dylib")):
                    os.chmod(os.path.join(root, f), 0o755)

    bin_path = _find_llama_server_bin()
    if not bin_path:
        raise RuntimeError(f"llama-server not found after extracting {asset_name}")
    log.info(f"[LLM] llama-server installed: {bin_path}")
    return bin_path


def _download_llama_model():
    """Download GGUF model from HuggingFace if not cached."""
    model_path = LLAMA_DIR / LLAMA_MODEL_FILE
    if model_path.exists():
        log.info(f"[LLM] Model cached: {model_path}")
        return str(model_path)

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"[LLM] Downloading {LLAMA_MODEL_REPO}/{LLAMA_MODEL_FILE}...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=LLAMA_MODEL_REPO,
            filename=LLAMA_MODEL_FILE,
            local_dir=str(LLAMA_DIR),
            local_dir_use_symlinks=False,
        )
        log.info(f"[LLM] Model ready: {downloaded}")
        return str(model_path) if model_path.exists() else downloaded
    except ImportError:
        pass

    import urllib.request
    url = f"https://huggingface.co/{LLAMA_MODEL_REPO}/resolve/main/{LLAMA_MODEL_FILE}"
    tmp_path = str(model_path) + ".tmp"
    urllib.request.urlretrieve(url, tmp_path)
    # Validate downloaded file is a GGUF model (not an HTML error page)
    with open(tmp_path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        os.remove(tmp_path)
        raise RuntimeError(f"Downloaded file is not a valid GGUF model (magic={magic!r}). "
                           f"HuggingFace may have returned an error page (403/rate limit).")
    os.replace(tmp_path, str(model_path))
    log.info(f"[LLM] Model downloaded: {model_path}")
    return str(model_path)


def _start_bundled_llama():
    """Download (if needed) and start the bundled llama-server."""
    global _llama_process, _llama_log_fh, LLAMA_CPP_URL
    import subprocess as _sp
    import socket as _sock

    if LLAMA_BUNDLED == "false":
        log.info("[LLM] Bundled server disabled (LLAMA_BUNDLED=false)")
        return

    port = int(LLAMA_PORT)

    # Kill any orphan llama-server from a previous SIGKILL'd run
    if sys.platform != "win32":
        try:
            import subprocess as _sp_kill
            result = _sp_kill.run(["pkill", "-f", f"llama-server.*--port.*{port}"],
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                log.info(f"[LLM] Killed orphan llama-server on port {port}")
                time.sleep(1)  # let GPU memory release
        except (FileNotFoundError, _sp_kill.TimeoutExpired, OSError):
            pass

    if LLAMA_BUNDLED == "auto":
        try:
            s = _sock.create_connection(("localhost", port), timeout=2)
            s.close()
            log.info(f"[LLM] Port {port} in use — using existing LLM server")
            return
        except (ConnectionRefusedError, OSError, _sock.timeout):
            pass

    try:
        bin_path = _download_llama_server()
        model_path = _download_llama_model()
    except Exception as e:
        log.warning(f"[LLM] Setup failed: {e}")
        log.warning("[LLM] Call analysis will be unavailable")
        return

    # LD_LIBRARY_PATH for shared libs (Linux)
    env = os.environ.copy()
    # Cap subprocess thread pools to container CPU count (not host)
    _llama_threads = str(min(8, resource_manager.cpu_count))
    env["OMP_NUM_THREADS"] = _llama_threads
    env["MKL_NUM_THREADS"] = _llama_threads
    env["OPENBLAS_NUM_THREADS"] = _llama_threads
    bin_dir = str(Path(bin_path).parent)
    lib_dir = str(Path(bin_path).parent.parent / "lib")
    if sys.platform != "win32" and os.path.isdir(lib_dir):
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{bin_dir}" + (f":{ld}" if ld else "")

    # Determine GPU layers based on available VRAM
    gpu_layers = LLAMA_GPU_LAYERS
    if gpu_layers == "auto":
        if resource_manager.total_vram_mb >= 6000:
            gpu_layers = "32"  # All 32 layers on GPU (≈ 2.5GB for Qwen3.5-4B Q4)
        elif resource_manager.total_vram_mb >= 4000:
            gpu_layers = "22"  # Partial offload (≈ 1.5GB)
        else:
            gpu_layers = "0"   # CPU only
        log.info(f"[LLM] Auto GPU layers: {gpu_layers} (VRAM: {resource_manager.total_vram_mb}MB)")

    cmd = [
        bin_path, "-m", model_path,
        "--n-gpu-layers", gpu_layers,
        "--ctx-size", LLAMA_CTX_SIZE,
        "--batch-size", LLAMA_BATCH_SIZE,
        "--ubatch-size", LLAMA_UBATCH_SIZE,
        "--threads", str(min(8, resource_manager.cpu_count)),
        "--port", str(port),
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]
    if CUDA_AVAILABLE:
        cmd.extend(["--flash-attn", "on"])

    llama_log = LLAMA_DIR / "llama-server.log"
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"[LLM] Starting llama-server on port {port} with {gpu_layers} GPU layers...")
    # Solution 3: Measure VRAM before llama-server starts
    _llama_vram_before = resource_manager.get_actual_free_vram_mb()
    try:
        with _llama_lock:
            _llama_log_fh = open(str(llama_log), "w")
            _llama_process = _sp.Popen(cmd, stdout=_llama_log_fh, stderr=_sp.STDOUT, env=env)
    except Exception as e:
        with _llama_lock:
            if _llama_log_fh:
                _llama_log_fh.close()
                _llama_log_fh = None
        log.error(f"[LLM] Failed to start llama-server: {e}")
        return

    for i in range(120):
        time.sleep(1)
        with _llama_lock:
            if _llama_process is None:
                return  # Shutdown handler already cleaned up
            if _llama_process.poll() is not None:
                exit_code = _llama_process.returncode
                _llama_log_fh.close()
                _llama_log_fh = None
                _llama_process = None
                tail = ""
                try:
                    with open(str(llama_log), "r", errors="replace") as _lf:
                        _lf.seek(0, 2)
                        _lf.seek(max(0, _lf.tell() - 1000))
                        tail = _lf.read()[-500:]
                except Exception:
                    pass
                log.error(f"[LLM] llama-server exited (code {exit_code})")
                if tail:
                    log.error(f"[LLM] Output:\n{tail}")
                return
        try:
            s = _sock.create_connection(("localhost", port), timeout=2)
            s.close()
            LLAMA_CPP_URL = f"http://localhost:{port}/v1/chat/completions"
            # Solution 3: Measure VRAM consumed by llama-server
            _llama_vram_after = resource_manager.get_actual_free_vram_mb()
            _llama_measured = max(0, _llama_vram_before - _llama_vram_after)
            if _llama_measured > 0:
                resource_manager.PROFILES["llm_external"]["vram_mb"] = _llama_measured
                log.info(f"[LLM] Measured VRAM: {_llama_measured}MB (profile updated)")
            # Release any previous reservation before re-acquiring (prevents doubling on crash-restart)
            resource_manager.release("llama_server")
            # Register llama-server VRAM/RAM with ResourceManager so bookkeeping is accurate
            _llm_vram = _llama_measured if _llama_measured > 0 else resource_manager.PROFILES["llm_external"]["vram_mb"]
            _llm_ram = resource_manager.PROFILES["llm_external"].get("ram_mb", 500)
            if not resource_manager.try_acquire("llama_server", "llm_external",
                                                 vram_mb=_llm_vram, ram_mb=_llm_ram):
                log.warning(f"[LLM] Could not register VRAM reservation ({_llm_vram}MB) — running untracked")
            log.info(f"[LLM] llama-server ready (took {i+1}s)")
            return
        except (ConnectionRefusedError, OSError, _sock.timeout):
            if i % 15 == 14:
                log.info(f"[LLM] Waiting for llama-server... ({i+1}s)")

    log.warning("[LLM] llama-server did not start within 120s")
    with _llama_lock:
        if _llama_process is not None:
            _llama_process.kill()
            _llama_process = None
        if _llama_log_fh:
            _llama_log_fh.close()
            _llama_log_fh = None


def _stop_bundled_llama():
    """Terminate the bundled llama-server subprocess."""
    global _llama_process, _llama_log_fh
    with _llama_lock:
        if _llama_process is not None:
            log.info("[LLM] Stopping llama-server...")
            _llama_process.terminate()
            try:
                _llama_process.wait(timeout=10)
            except Exception:
                _llama_process.kill()
            _llama_process = None
        if _llama_log_fh is not None:
            _llama_log_fh.close()
            _llama_log_fh = None
    # Release VRAM/RAM reservation
    resource_manager.release("llama_server")


def _warmup_models():
    """Pre-load ML models in background so first user request is fast."""
    def _do_warmup():
        if FASTER_WHISPER_AVAILABLE:
            try:
                log.info("[Warmup] Pre-loading Whisper model...")
                _get_whisper_model()
                log.info("[Warmup] Whisper ready")
            except Exception as e:
                log.warning(f"[Warmup] Whisper failed: {e}")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                log.info("[Warmup] Pre-loading embedding model...")
                _get_sentence_model()
                log.info("[Warmup] Embedding ready")
            except Exception as e:
                log.warning(f"[Warmup] Embedding failed: {e}")

    threading.Thread(target=_do_warmup, daemon=True, name="model-warmup").start()


WHISPER_MODEL_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", str(DATA_DIR / "stt"))
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")

def _get_whisper_compute_type():
    """Pick optimal compute type based on actual device."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "int8_float16"
    except Exception:
        pass
    return "int8"

WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE") or _get_whisper_compute_type()


# Thread pool for audio evaluation (limit concurrency — serialized)
_audio_eval_pool = ThreadPoolExecutor(max_workers=1)
# Thread pool for training jobs (limit to 1 concurrent training)
_training_pool = ThreadPoolExecutor(max_workers=1)


class FairJobQueue:
    """Fair job queue with position tracking. Round-robin across users."""
    def __init__(self, pool, job_store, running_status="training"):
        self._lock = threading.RLock()
        self._queue = []  # [(username, job_id, submit_time, callable, args, kwargs)]
        self._pool = pool
        self._job_store = job_store
        self._running = None  # (username, job_id) or None
        self._running_status = running_status

    def submit(self, username, job_id, fn, *args, **kwargs):
        """Enqueue a job. Returns queue position (0 = will run next)."""
        with self._lock:
            self._queue.append((username, job_id, time.time(), fn, args, kwargs))
            pos = len(self._queue) - 1
            if self._running is None:
                self._dispatch_next()
                return 0
            return pos

    def get_position(self, job_id):
        """Get current queue position for a job. Returns -1 if not queued (running or not found)."""
        with self._lock:
            for i, (_, jid, _, _, _, _) in enumerate(self._queue):
                if jid == job_id:
                    return i
            return -1

    def _dispatch_next(self):
        """Dispatch the next job from queue to the thread pool. Must hold self._lock."""
        if not self._queue:
            self._running = None
            return
        # Pick next job (FIFO — round-robin not needed with 1-per-user limit)
        username, job_id, _, fn, args, kwargs = self._queue.pop(0)
        self._running = (username, job_id)
        self._job_store.update_fields(job_id, status=self._running_status)

        def _wrapper():
            try:
                fn(*args, **kwargs)
            finally:
                with self._lock:
                    self._running = None
                    try:
                        self._dispatch_next()
                    except Exception as _dne:
                        log.error(f"[FairJobQueue] _dispatch_next failed: {_dne}")

        self._pool.submit(_wrapper)

    def queue_length(self):
        with self._lock:
            return len(self._queue)


_training_queue = FairJobQueue(_training_pool, training_jobs, running_status="training")
_audio_eval_queue = FairJobQueue(_audio_eval_pool, audio_eval_jobs, running_status="processing")


# ══════════════════════════════════════════════════════════════════
#  MODEL CACHE — lazy-load & auto-evict after 20 min idle
# ══════════════════════════════════════════════════════════════════

class ModelCache:
    """Thread-safe in-memory cache for loaded AutoGluon predictors.

    Models are evicted automatically if not used for ``ttl`` seconds.
    A single daemon thread performs periodic sweeps.
    """

    def __init__(self, ttl: int = 1200):
        self._cache: dict = {}          # key → {"predictor": obj, "last_used": float, "task_type": str}
        self._lock = threading.Lock()
        self._ttl = ttl                 # seconds (default 20 min)
        self._loading: set = set()      # keys currently being loaded (prevent double-load)
        # Start the evictor daemon
        t = threading.Thread(target=self._evictor_loop, daemon=True)
        t.start()

    # ── public API ────────────────────────────────────────────

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            entry = self._cache.get(model_id)
            return entry is not None

    def get(self, model_id: str):
        """Return cached predictor or None.  Touching updates last_used."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry is not None:
                entry["last_used"] = time.time()
                return entry["predictor"]
        return None

    def put(self, model_id: str, predictor, task_type: str = "tabular"):
        """Store a predictor in the cache."""
        with self._lock:
            self._cache[model_id] = {
                "predictor": predictor,
                "task_type": task_type,
                "last_used": time.time(),
            }
            self._loading.discard(model_id)
        print(f"[ModelCache] Cached model {model_id[:8]}… ({task_type})")

    def evict(self, model_id: str):
        """Manually remove a model from cache (e.g. on delete)."""
        with self._lock:
            entry = self._cache.pop(model_id, None)
        if entry:
            print(f"[ModelCache] Evicted model {model_id[:8]}…")
            del entry  # release reference for GC
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, Exception):
                pass

    def is_loading(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._loading

    def mark_loading(self, model_id: str):
        with self._lock:
            self._loading.add(model_id)

    def unmark_loading(self, model_id: str):
        with self._lock:
            self._loading.discard(model_id)

    def load_model(self, model_id: str, meta: dict):
        """Load a model from disk into cache.  Returns the predictor.

        Handles memory / VRAM errors gracefully by falling back to None.
        Thread-safe: if another thread is already loading the same model,
        waits for it to finish and returns the cached result.
        """
        # Fast path: already cached
        cached = self.get(model_id)
        if cached is not None:
            return cached

        # Wait if another thread is already loading this model (prevent double-load)
        _waited = 0
        while self.is_loading(model_id):
            time.sleep(0.2)
            _waited += 0.2
            if _waited > 120:
                log.warning(f"[ModelCache] Timeout waiting for model {model_id[:8]}… to load")
                return None
            cached = self.get(model_id)
            if cached is not None:
                return cached

        ag_path = str(MODELS_DIR / model_id / "agmodel")
        task_type = meta.get("task_type", "tabular")

        try:
            self.mark_loading(model_id)

            if task_type == "timeseries":
                if not AUTOGLUON_TS_AVAILABLE:
                    raise ImportError("autogluon.timeseries yüklü değil")
                predictor = TimeSeriesPredictor.load(ag_path)

            else:  # tabular (classification, regression — with or without text embeddings)
                predictor = TabularPredictor.load(ag_path)

            self.put(model_id, predictor, task_type)
            return predictor

        except (MemoryError, RuntimeError) as e:
            # RuntimeError covers CUDA OOM from PyTorch
            err_str = str(e).lower()
            is_oom = isinstance(e, MemoryError) or "out of memory" in err_str or "cuda" in err_str
            if is_oom:
                print(f"[ModelCache] OOM loading {model_id[:8]}… — attempting to free memory")
                self._emergency_evict()
                # Retry once
                try:
                    if task_type == "timeseries":
                        predictor = TimeSeriesPredictor.load(ag_path)
                    else:
                        predictor = TabularPredictor.load(ag_path)
                    self.put(model_id, predictor, task_type)
                    return predictor
                except Exception as e2:
                    log.warning(f"[ModelCache] Retry failed: {e2}")
                    return None
            raise
        except Exception:
            traceback.print_exc()
            return None
        finally:
            self.unmark_loading(model_id)

    def _evict_idle(self, min_idle_seconds: int = 60):
        """Evict cached models idle longer than min_idle_seconds."""
        now = time.time()
        with self._lock:
            to_evict = [k for k, v in self._cache.items()
                       if (now - v["last_used"]) > min_idle_seconds]
            for key in to_evict:
                self._cache.pop(key, None)
                log.info(f"[ModelCache] Pre-evicted idle model {key[:8]}…")

    # ── internal ──────────────────────────────────────────────

    def _emergency_evict(self):
        """Evict the oldest models to free memory."""
        with self._lock:
            if not self._cache:
                return
            # Sort by last_used ascending, evict oldest half (at least 1)
            # Skip models that are actively being used in predictions
            items = [(k, v) for k, v in sorted(self._cache.items(), key=lambda kv: kv[1]["last_used"])
                     if not model_ref_counter.is_busy(k)]
            evict_count = max(1, len(items) // 2)
            for key, _ in items[:evict_count]:
                entry = self._cache.pop(key, None)
                if entry:
                    print(f"[ModelCache] Emergency evicted {key[:8]}…")
                    del entry
        # Also evict idle embedding models to free more VRAM
        try:
            now = time.time()
            with _sentence_model_lock:
                to_evict = [name for name, entry in _sentence_model_cache.items()
                           if (now - entry.get("last_used", now)) > 60]  # Aggressive: 1 min
                for name in to_evict:
                    del _sentence_model_cache[name]
                    log.info(f"[ModelCache] Emergency evicted embedding model: {name}")
        except Exception:
            pass
        # Nudge garbage collector
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _evictor_loop(self):
        """Background loop that evicts stale models every 60 seconds."""
        while True:
            time.sleep(60)
            now = time.time()
            to_evict = []
            with self._lock:
                for mid, entry in list(self._cache.items()):
                    if now - entry["last_used"] > self._ttl:
                        # Don't evict models that are actively being used in predictions
                        if model_ref_counter.is_busy(mid):
                            continue
                        to_evict.append(mid)
                for mid in to_evict:
                    entry = self._cache.pop(mid, None)
                    if entry:
                        print(f"[ModelCache] TTL evicted {mid[:8]}… "
                              f"(idle {int(now - entry['last_used'])}s)")
                        del entry
            if to_evict:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass


# Global cache instance — 20 minute TTL
model_cache = ModelCache(ttl=1200)


# ══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def get_model_type_category(model_name: str) -> str:
    """Categorize a submodel by its base type."""
    name_upper = model_name.upper()
    for tree in TREE_BASED_MODELS:
        if tree.upper() in name_upper:
            return "tree"
    for lin in LINEAR_MODELS:
        if lin.upper() in name_upper:
            return "linear"
    for nn in NEURAL_MODELS:
        if nn.upper() in name_upper:
            return "neural"
    for knn in KNN_MODELS:
        if knn.upper() in name_upper:
            return "knn"
    # Fallback: broader keyword matching for unknown/future model names
    # Use word-boundary-aware checks to avoid false positives (e.g. "XT" in "TEXT")
    if "XGBOOST" in name_upper or name_upper.startswith("XGB"):
        return "tree"
    if "LIGHTGBM" in name_upper or name_upper.startswith("LGB"):
        return "tree"
    if "CATBOOST" in name_upper:
        return "tree"
    if "RANDOMFOREST" in name_upper:
        return "tree"
    if "EXTRATREE" in name_upper:
        return "tree"
    if "NEURAL" in name_upper or "FASTAI" in name_upper or "TORCH" in name_upper:
        return "neural"
    if name_upper.startswith("NN_") or name_upper.endswith("_NN"):
        return "neural"
    if "LINEARMODEL" in name_upper:
        return "linear"
    if "KNN" in name_upper or "KNEIGHBOR" in name_upper:
        return "knn"
    if "WEIGHTED" in name_upper or "ENSEMBLE" in name_upper:
        return "ensemble"
    return "other"


def get_sql_support(model_name: str) -> dict:
    """SQL export compatibility."""
    cat = get_model_type_category(model_name)
    if cat in ("tree", "linear"):
        return {
            "easy_sql": True,
            "sql_method": "native",
            "label": "Saatlik çalışabilir",
            "verified": False,
        }
    elif cat == "ensemble":
        return {
            "easy_sql": False,
            "sql_method": "none",
            "label": "Desteklenmiyor (Ensemble)",
            "verified": True,
        }
    else:
        return {
            "easy_sql": False,
            "sql_method": "none",
            "label": "Desteklenmiyor",
            "verified": True,
        }


def verify_sql_support(predictor, model_name: str, feature_columns: list, target_col: str) -> dict:
    """Test whether SQL generation works for this submodel."""
    cat = get_model_type_category(model_name)

    if cat == "ensemble":
        return {
            "easy_sql": False,
            "sql_method": "none", "label": "Desteklenmiyor (Ensemble)", "verified": True,
        }

    sql_result = None
    try:
        if cat == "tree":
            sql_result = generate_tree_sql(predictor, model_name, feature_columns, target_col)
        elif cat == "linear":
            sql_result = generate_linear_sql(predictor, model_name, feature_columns, target_col)
    except Exception as e:
        print(f"  [SQL doğrulama] {model_name}: üretim başarısız: {e}")

    if sql_result is not None:
        return {
            "easy_sql": True,
            "sql_method": "native", "label": "Saatlik çalışabilir", "verified": True,
        }
    else:
        return {
            "easy_sql": False,
            "sql_method": "none", "label": "SQL desteklenmiyor", "verified": True,
        }


def get_airflow_support(model_name: str) -> dict:
    """All non-ensemble models support Airflow (Python DAG)."""
    cat = get_model_type_category(model_name)
    if cat == "ensemble":
        return {"supported": False, "label": "Desteklenmiyor (Ensemble)"}
    return {"supported": True, "label": "Günlük çalışabilir"}


def load_model_meta(model_id: str) -> dict:
    """Load model metadata from JSON file (thread-safe, per-model lock)."""
    meta_path = MODELS_DIR / model_id / "meta.json"
    with _get_model_lock(model_id):
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to load meta for {model_id}: {e}")
            return None


def save_model_meta(model_id: str, meta: dict):
    """Save model metadata to JSON file (thread-safe, atomic, NaN-safe, per-model lock)."""
    meta_path = MODELS_DIR / model_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_model_lock(model_id):
        _atomic_write_json(meta_path, meta, use_safe_json=True)
    invalidate_models_cache()


def update_model_meta_fields(model_id: str, **updates) -> dict:
    """Atomically read-modify-write specific fields on model metadata.
    Returns the updated meta or None if model not found.
    Use this instead of load→modify→save to prevent race conditions."""
    with _get_model_lock(model_id):
        meta_path = MODELS_DIR / model_id / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        meta.update(updates)
        _atomic_write_json(meta_path, meta, use_safe_json=True)
    invalidate_models_cache()
    return meta


def _estimate_model_ram_mb(model_id: str, meta: dict) -> int:
    """Estimate RAM needed to load a model: use saved measurement, else 2.5× disk size."""
    saved = meta.get("estimated_ram_mb")
    if saved and saved > 0:
        return saved
    agmodel_dir = MODELS_DIR / model_id / "agmodel"
    if agmodel_dir.exists():
        try:
            disk_bytes = sum(f.stat().st_size for f in agmodel_dir.rglob('*') if f.is_file())
            disk_mb = disk_bytes / (1024 * 1024)
            estimated = max(1000, int(disk_mb * 2.5))
            # Cap: single prediction can't claim more than half total RAM
            estimated = min(estimated, resource_manager.safe_ram_mb // 2)
            # Save to metadata so we don't re-scan next time
            update_model_meta_fields(model_id, estimated_ram_mb=estimated)
            return estimated
        except OSError:
            pass
    return resource_manager.PROFILES["prediction_model_load"]["ram_mb"]


def increment_model_meta_counter(model_id: str, field: str, amount: int = 1) -> dict:
    """Atomically increment a numeric counter on model metadata."""
    with _get_model_lock(model_id):
        meta_path = MODELS_DIR / model_id / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        meta[field] = meta.get(field, 0) + amount
        _atomic_write_json(meta_path, meta, use_safe_json=True)
    invalidate_models_cache()
    return meta


def load_activity_log() -> list:
    with _file_locks["activity"]:
        return _safe_read_json(ACTIVITY_FILE, default=[])


def save_activity_log(log_data: list):
    with _file_locks["activity"]:
        _atomic_write_json(ACTIVITY_FILE, log_data, use_safe_json=True)


def add_activity(action: str, model_id: str = None, model_name: str = None, details: str = "", username: str = "", visibility: str = "admin_only"):
    """Add an activity entry. Atomic read-modify-write.

    visibility: 'public' = visible to all users, 'admin_only' = only admins see it.
    """
    with _file_locks["activity"]:
        log_data = _safe_read_json(ACTIVITY_FILE, default=[])
        log_data.insert(0, {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "model_id": model_id,
            "model_name": model_name,
            "details": details,
            "username": username,
            "visibility": visibility,
        })
        log_data = log_data[:100]
        _atomic_write_json(ACTIVITY_FILE, log_data, use_safe_json=True)


def get_filtered_activity(user: dict) -> list:
    """Return activity entries filtered by user role."""
    activity = load_activity_log()
    is_admin = user.get("role") in ("admin", "master_admin")
    if is_admin:
        return activity
    # Regular users: see public activities + their own + old entries without visibility field
    username = user.get("username", "")
    return [a for a in activity
            if a.get("visibility", "public") == "public" or a.get("username") == username]


_all_models_cache = {"data": None, "ts": 0, "gen": 0}
_all_models_cache_lock = threading.Lock()
_ALL_MODELS_CACHE_TTL = 5  # seconds

def get_all_models(force_refresh: bool = False) -> list:
    """Get list of all model metadata. Cached for 5 seconds to avoid O(n) disk I/O.
    Uses a generation counter to prevent stale data overwriting fresh invalidations."""
    now = time.time()
    with _all_models_cache_lock:
        if not force_refresh and _all_models_cache["data"] is not None and (now - _all_models_cache["ts"]) < _ALL_MODELS_CACHE_TTL:
            return _all_models_cache["data"]
        gen_at_start = _all_models_cache["gen"]
    # Build outside lock to avoid blocking readers during disk I/O
    models = []
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                meta = load_model_meta(model_dir.name)
                if meta:
                    models.append(meta)
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    with _all_models_cache_lock:
        # Only update if no invalidation happened while we were building
        if _all_models_cache["gen"] == gen_at_start:
            _all_models_cache["data"] = models
            _all_models_cache["ts"] = time.time()
    return models

def invalidate_models_cache():
    """Invalidate the models cache after create/delete/update operations."""
    with _all_models_cache_lock:
        _all_models_cache["data"] = None
        _all_models_cache["gen"] += 1


def get_public_models() -> list:
    """Get public models only."""
    return [m for m in get_all_models() if m.get("visibility", "public") == "public"]


_model_quota_lock = threading.Lock()
_model_quota_pending = {}  # username -> int (pending model creations not yet saved)

def count_user_models(username: str) -> int:
    """Count total models owned by a user."""
    return sum(1 for m in get_all_models() if m.get("owner") == username)

def check_and_reserve_model_quota(username: str) -> bool:
    """Atomically check quota and reserve a slot. Returns True if allowed.
    The caller MUST call release_model_quota() if the model creation fails."""
    with _model_quota_lock:
        pending = _model_quota_pending.get(username, 0)
        if count_user_models(username) + pending >= MAX_MODELS_PER_USER:
            return False
        _model_quota_pending[username] = pending + 1
        return True

def release_model_quota(username: str):
    """Release a pending model quota reservation (e.g., on training failure)."""
    with _model_quota_lock:
        pending = _model_quota_pending.get(username, 0)
        if pending > 0:
            _model_quota_pending[username] = pending - 1
        if _model_quota_pending.get(username, 0) <= 0:
            _model_quota_pending.pop(username, None)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    # Clamp p to [0, 1] — regression holdout scores can exceed 1.0 in fallback paths
    p = max(0.0, min(1.0, p))
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


def measure_inference_time(predictor, sample_data, model_name: str) -> float:
    try:
        import numpy as np
        clean_sample = sample_data.copy()
        numeric_cols = clean_sample.select_dtypes(include=[np.number]).columns
        clean_sample[numeric_cols] = clean_sample[numeric_cols].replace([np.inf, -np.inf], np.nan)
        predictor.predict(clean_sample.head(1), model=model_name)
        times = []
        for _ in range(3):
            start = time.time()
            predictor.predict(clean_sample, model=model_name)
            elapsed = time.time() - start
            times.append(elapsed)
        return round(min(times), 4)
    except Exception as e:
        print(f"  [Benchmark] {model_name} için ölçüm yapılamadı: {e}")
        return -1.0


# ══════════════════════════════════════════════════════════════════
#  SQL GENERATION (Tree, Linear, XGBoost, LightGBM, sklearn)
# ══════════════════════════════════════════════════════════════════

def _collect_inner_objects(predictor, model_name: str) -> list:
    candidates = []
    try:
        ag_model = predictor._trainer.load_model(model_name)
    except Exception as e:
        print(f"  [SQL] Model '{model_name}' yüklenemedi: {e}")
        return candidates

    ag_type = type(ag_model).__name__
    print(f"  [SQL] Model '{model_name}' → AutoGluon wrapper: {ag_type}")
    candidates.append(("ag_model", ag_model))

    for attr in ['model', '_model', 'model_', '_learner', 'estimator', '_clf', '_reg']:
        obj = getattr(ag_model, attr, None)
        if obj is not None:
            candidates.append((f"ag_model.{attr}", obj))
            for attr2 in ['model', '_model', 'booster_', '_Booster', 'estimator']:
                obj2 = getattr(obj, attr2, None)
                if obj2 is not None:
                    candidates.append((f"ag_model.{attr}.{attr2}", obj2))
    return candidates


def _sql_bracket(col: str) -> str:
    """Escape column name for MSSQL bracket notation. Doubles ] to prevent injection."""
    return f"[{str(col).replace(']', ']]')}]"


def _try_lightgbm_sql(obj, feature_columns: list) -> str:
    try:
        booster = getattr(obj, 'booster_', None)
        if booster is not None and hasattr(booster, 'dump_model'):
            model_dump = booster.dump_model()
            return _lgbm_dump_to_sql(model_dump, feature_columns)
        if hasattr(obj, 'dump_model') and callable(getattr(obj, 'dump_model')):
            model_dump = obj.dump_model()
            if 'tree_info' in model_dump:
                return _lgbm_dump_to_sql(model_dump, feature_columns)
    except Exception as e:
        log.debug(f"LightGBM SQL generation failed: {e}")
    return None


def _lgbm_dump_to_sql(model_dump: dict, feature_columns: list) -> str:
    _SQL_MAX_TREES = 30
    trees = model_dump.get('tree_info', [])
    if not trees:
        return None
    model_feature_names = model_dump.get('feature_names', [])
    sql_cols = model_feature_names if model_feature_names else feature_columns
    tree_sqls = []
    _trees_truncated = len(trees) > _SQL_MAX_TREES
    for tree_info in trees[:_SQL_MAX_TREES]:
        tree = tree_info.get('tree_structure', {})
        sql = _lgbm_node_to_sql(tree, sql_cols)
        tree_sqls.append(f"({sql})")
    if not tree_sqls:
        return None
    truncation_warning = ""
    if _trees_truncated:
        truncation_warning = f"/* UYARI: Model {len(trees)} ağaç içeriyor, SQL yalnızca ilk {_SQL_MAX_TREES} ağacı kullanıyor — sonuçlar yaklaşık olacaktır */\n"
    sum_expr = " + ".join(tree_sqls)
    objective = model_dump.get('objective', '')
    if 'binary' in objective:
        return f"{truncation_warning}1.0 / (1.0 + EXP(-({sum_expr})))"
    return f"{truncation_warning}{sum_expr}"


def _try_xgboost_sql(obj, feature_columns: list) -> str:
    try:
        booster = None
        if hasattr(obj, 'get_booster') and callable(getattr(obj, 'get_booster')):
            booster = obj.get_booster()
        elif hasattr(obj, 'get_dump') and callable(getattr(obj, 'get_dump')):
            booster = obj
        if booster is None:
            return None
        try:
            bfn = booster.feature_names
            sql_cols = bfn if bfn else feature_columns
        except Exception:
            sql_cols = feature_columns
        _SQL_MAX_TREES = 30
        trees = booster.get_dump(dump_format='json')
        if not trees:
            return None
        _trees_truncated = len(trees) > _SQL_MAX_TREES
        tree_sqls = []
        for tree_json_str in trees[:_SQL_MAX_TREES]:
            tree_data = json.loads(tree_json_str)
            sql = _xgb_node_to_sql(tree_data, sql_cols)
            tree_sqls.append(f"({sql})")
        if not tree_sqls:
            return None
        sum_expr = " + ".join(tree_sqls)
        objective = ''
        if hasattr(obj, 'objective'):
            objective = str(getattr(obj, 'objective', ''))
        elif hasattr(obj, 'get_dump'):
            try:
                config = json.loads(booster.save_config())
                objective = config.get('learner', {}).get('objective', {}).get('name', '')
            except Exception:
                pass
        truncation_warning = ""
        if _trees_truncated:
            truncation_warning = f"/* UYARI: Model {len(trees)} ağaç içeriyor, SQL yalnızca ilk {_SQL_MAX_TREES} ağacı kullanıyor — sonuçlar yaklaşık olacaktır */\n"
        if 'binary' in objective:
            return f"{truncation_warning}1.0 / (1.0 + EXP(-({sum_expr})))"
        return f"{truncation_warning}{sum_expr}"
    except Exception as e:
        log.debug(f"XGBoost SQL generation failed: {e}")
    return None


def _try_sklearn_ensemble_sql(obj, feature_columns: list) -> str:
    try:
        estimators = getattr(obj, 'estimators_', None)
        if estimators is None or not hasattr(obj, 'n_estimators'):
            return None
        if not hasattr(estimators[0], 'tree_'):
            return None
        return _sklearn_tree_ensemble_to_sql(obj, feature_columns)
    except Exception as e:
        log.debug(f"sklearn ensemble SQL generation failed: {e}")
    return None


def _try_linear_sql(obj, feature_columns: list, target_col: str, table_name: str) -> str:
    try:
        coefs = getattr(obj, 'coef_', None)
        intercept = getattr(obj, 'intercept_', None)
        if coefs is None or intercept is None:
            return None
        import numpy as np
        if len(coefs.shape) > 1:
            coefs = coefs[0]
        if hasattr(intercept, '__len__'):
            intercept = intercept[0]
        terms = [str(round(float(intercept), 8))]
        for i, coef in enumerate(coefs):
            if i < len(feature_columns):
                c = feature_columns[i]
                coef_val = round(float(coef), 8)
                terms.append(f"ISNULL({coef_val} * TRY_CAST({_sql_bracket(c)} AS FLOAT), 0)")
        expression = " + ".join(terms)
        col_list = ", ".join([_sql_bracket(c) for c in feature_columns])
        null_check = " OR ".join([f"{_sql_bracket(c)} IS NULL" for c in feature_columns])
        return f"""SELECT
    {col_list},
    ({expression}) AS {_sql_bracket(f"{target_col}_predicted")},
    CASE WHEN {null_check} THEN 0 ELSE 1 END AS {_sql_bracket(f"{target_col}_prediction_valid")}
FROM {table_name};"""
    except Exception:
        pass
    return None


def generate_tree_sql(predictor, model_name: str, feature_columns: list, target_col: str,
                      table_name: str = "ExampleDB.dbo.Table_Containing_Rows_To_Process") -> str:
    try:
        candidates = _collect_inner_objects(predictor, model_name)
        if not candidates:
            return None

        converters = [
            ("LightGBM", _try_lightgbm_sql),
            ("XGBoost", _try_xgboost_sql),
            ("sklearn-ensemble", _try_sklearn_ensemble_sql),
        ]

        sql_expression = None
        winning_converter = None

        for conv_name, conv_fn in converters:
            for desc, obj in candidates:
                try:
                    result = conv_fn(obj, feature_columns)
                    if result is not None:
                        sql_expression = result
                        winning_converter = f"{conv_name} via {desc}"
                        print(f"  [SQL] ✓ BAŞARILI: {model_name} → {winning_converter}")
                        break
                except Exception:
                    pass
            if sql_expression is not None:
                break

        if sql_expression is None:
            print(f"  [SQL] ✗ Tüm dönüştürücüler başarısız: {model_name}")
            return None

        clean_col_defs = []
        for c in feature_columns:
            bc = _sql_bracket(c)
            clean_col_defs.append(
                f"        CASE\n"
                f"            WHEN {bc} IS NULL THEN NULL\n"
                f"            WHEN TRY_CAST({bc} AS FLOAT) IS NOT NULL\n"
                f"                 AND TRY_CAST({bc} AS FLOAT) <> TRY_CAST({bc} AS FLOAT)\n"
                f"                 THEN NULL\n"
                f"            ELSE {bc}\n"
                f"        END AS {bc}"
            )

        col_list = ", ".join([_sql_bracket(c) for c in feature_columns])
        clean_cols_sql = ",\n".join(clean_col_defs)

        sql = f"""-- ═══════════════════════════════════════════════════════════════
-- Otomatik üretilmiş MSSQL tahmin sorgusu
-- Model: {model_name}
-- Hedef sütun: {target_col}
-- Üretim zamanı: {datetime.now().isoformat()}
-- ═══════════════════════════════════════════════════════════════

WITH CleanedData AS (
    SELECT
{clean_cols_sql}
    FROM {table_name}
),
NullSafeData AS (
    SELECT
        {col_list}
    FROM CleanedData
)
SELECT
    {col_list},
    ({sql_expression}) AS {_sql_bracket(f"{target_col}_predicted")},
    CASE WHEN {" OR ".join([f"{_sql_bracket(c)} IS NULL" for c in feature_columns])}
         THEN 0 ELSE 1
    END AS {_sql_bracket(f"{target_col}_prediction_valid")}
FROM NullSafeData;
"""
        return sql
    except Exception as e:
        print(f"SQL üretim hatası {model_name}: {e}")
        traceback.print_exc()
        return None


def _tree_to_case_when(tree_dict: dict, feature_names: list, node_id: int = 0, _depth: int = 0) -> str:
    if _depth > 200:
        return "0"
    left = tree_dict["children_left"]
    right = tree_dict["children_right"]
    features = tree_dict["feature"]
    thresholds = tree_dict["threshold"]
    values = tree_dict["value"]

    if left[node_id] == -1:
        val = values[node_id]
        if hasattr(val, '__len__'):
            if len(val.shape) > 1:
                if val.shape[-1] == 1:
                    v = float(val.flat[0])
                else:
                    v = float(val.argmax())
            else:
                v = float(val[0])
        else:
            v = float(val)
        if math.isnan(v) or math.isinf(v):
            return "0"
        return str(round(v, 6))

    feat_idx = features[node_id]
    feat_name = _sql_bracket(feature_names[feat_idx]) if 0 <= feat_idx < len(feature_names) else _sql_bracket(f"f{feat_idx}")
    threshold = float(thresholds[node_id])
    if math.isnan(threshold) or math.isinf(threshold):
        return "0"
    left_expr = _tree_to_case_when(tree_dict, feature_names, left[node_id], _depth + 1)
    right_expr = _tree_to_case_when(tree_dict, feature_names, right[node_id], _depth + 1)
    return f"CASE WHEN {feat_name} <= {round(threshold, 10)} THEN {left_expr} ELSE {right_expr} END"


def _sklearn_tree_ensemble_to_sql(model, feature_columns: list) -> str:
    try:
        estimators = model.estimators_
        tree_sqls = []
        for est in estimators[:20]:
            tree = est.tree_
            tree_dict = {
                "children_left": tree.children_left,
                "children_right": tree.children_right,
                "feature": tree.feature,
                "threshold": tree.threshold,
                "value": tree.value,
            }
            sql = _tree_to_case_when(tree_dict, feature_columns)
            tree_sqls.append(f"({sql})")
        sum_expr = " + ".join(tree_sqls)
        truncation_warning = ""
        if len(estimators) > 20:
            truncation_warning = f"/* UYARI: Model {len(estimators)} ağaç içeriyor, SQL yalnızca ilk 20 ağacı kullanıyor — sonuçlar yaklaşık olacaktır */\n"
        return f"{truncation_warning}({sum_expr}) / {len(tree_sqls)}.0"
    except Exception as e:
        print(f"sklearn tree ensemble to SQL hatası: {e}")
        return None


def _xgb_node_to_sql(node: dict, feature_columns: list, _depth: int = 0) -> str:
    if _depth > 200:
        return "0"
    if 'leaf' in node:
        v = float(node['leaf'])
        if math.isnan(v) or math.isinf(v):
            return "0"
        return str(round(v, 8))
    split_feature = node.get('split', '')
    split_value = float(node.get('split_condition', 0))
    if math.isnan(split_value) or math.isinf(split_value):
        return "0"
    if split_feature.startswith('f') and split_feature[1:].isdigit():
        feat_idx = int(split_feature[1:])
        if feat_idx < len(feature_columns):
            feat_name = _sql_bracket(feature_columns[feat_idx])
        else:
            feat_name = _sql_bracket(split_feature)
    else:
        feat_name = _sql_bracket(split_feature)
    children = node.get('children', [])
    if len(children) >= 2:
        yes_child = children[0]
        no_child = children[1]
    else:
        return "0"
    yes_sql = _xgb_node_to_sql(yes_child, feature_columns, _depth + 1)
    no_sql = _xgb_node_to_sql(no_child, feature_columns, _depth + 1)
    # Handle XGBoost's learned missing-value direction
    missing_dir = node.get('missing')
    yes_id = node.get('yes')
    if missing_dir is not None and yes_id is not None:
        # missing_dir tells which child NULLs go to
        if missing_dir == yes_id:
            null_sql = yes_sql
        else:
            null_sql = no_sql
        return (f"CASE WHEN {feat_name} IS NULL THEN {null_sql} "
                f"WHEN {feat_name} < {round(split_value, 10)} THEN {yes_sql} "
                f"ELSE {no_sql} END")
    return f"CASE WHEN {feat_name} < {round(split_value, 10)} THEN {yes_sql} ELSE {no_sql} END"


def _lgbm_node_to_sql(node: dict, feature_columns: list, _depth: int = 0) -> str:
    if _depth > 200:
        return "0"
    if 'leaf_value' in node:
        v = float(node['leaf_value'])
        if math.isnan(v) or math.isinf(v):
            return "0"
        return str(round(v, 8))
    split_feature = node.get('split_feature', 0)
    threshold = float(node.get('threshold', 0))
    if math.isnan(threshold) or math.isinf(threshold):
        return "0"
    decision_type = node.get('decision_type', '<=')
    if isinstance(split_feature, int):
        if split_feature < len(feature_columns):
            feat_name = _sql_bracket(feature_columns[split_feature])
        else:
            feat_name = _sql_bracket(f"f{split_feature}")
    else:
        feat_name = _sql_bracket(split_feature)
    left_sql = _lgbm_node_to_sql(node.get('left_child', {'leaf_value': 0}), feature_columns, _depth + 1)
    right_sql = _lgbm_node_to_sql(node.get('right_child', {'leaf_value': 0}), feature_columns, _depth + 1)
    if decision_type == "==":
        return f"CASE WHEN {feat_name} = {round(threshold, 10)} THEN {left_sql} ELSE {right_sql} END"
    op = "<=" if decision_type == "<=" else "<"
    return f"CASE WHEN {feat_name} {op} {round(threshold, 10)} THEN {left_sql} ELSE {right_sql} END"


def generate_linear_sql(predictor, model_name: str, feature_columns: list, target_col: str,
                        table_name: str = "ExampleDB.dbo.Table_Containing_Rows_To_Process") -> str:
    try:
        candidates = _collect_inner_objects(predictor, model_name)
        for desc, obj in candidates:
            result = _try_linear_sql(obj, feature_columns, target_col, table_name)
            if result is not None:
                print(f"  [SQL] ✓ Lineer SQL via {desc}")
                return f"""-- ═══════════════════════════════════════════════════════════════
-- Otomatik üretilmiş MSSQL tahmin sorgusu (Lineer Model)
-- Model: {model_name}
-- Hedef: {target_col}
-- Üretim zamanı: {datetime.now().isoformat()}
-- ═══════════════════════════════════════════════════════════════

{result}"""
        print(f"  [SQL] {model_name} için lineer model bulunamadı")
    except Exception as e:
        print(f"  [SQL] Lineer SQL üretim hatası: {e}")
    return None


def generate_airflow_dag(model_id: str, model_name: str, submodel_name: str,
                         target_col: str, feature_columns: list, task_type: str,
                         text_columns: list = None, embedding_model: str = "") -> str:
    model_path = MODELS_DIR / model_id / "agmodel"
    safe_submodel = re.sub(r'[^a-zA-Z0-9._-]', '_', submodel_name).lower()

    # For text models, generate the embedding helper code
    text_columns = text_columns or []
    embedding_imports = ""
    embedding_constants = ""
    embedding_function = ""
    if text_columns:
        embedding_imports = "\nfrom sentence_transformers import SentenceTransformer"
        embedding_constants = f"""
EMBEDDING_MODEL = {json.dumps(embedding_model or DEFAULT_EMBEDDING_MODEL)}
"""
        embedding_function = """

_sentence_model = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        logger.info(f"Embedding modeli yukleniyor: {EMBEDDING_MODEL}")
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding modeli yuklendi (dim={_sentence_model.get_sentence_embedding_dimension()})")
    return _sentence_model


def embed_text_columns(df, text_columns):
    model = get_sentence_model()
    embed_dim = model.get_sentence_embedding_dimension()
    result_df = df.copy()
    for col in text_columns:
        texts = result_df[col].fillna("").astype(str).tolist()
        logger.info(f"Encoding {len(texts)} texts from column '{col}'...")
        embeddings = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        emb_col_names = [f"{col}_emb_{i}" for i in range(embed_dim)]
        import pandas as _pd
        emb_df = _pd.DataFrame(embeddings, columns=emb_col_names, index=result_df.index)
        result_df = _pd.concat([result_df, emb_df], axis=1)
        result_df = result_df.drop(columns=[col])
    return result_df
"""

    dag_code = f'''"""
Otomatik üretilmiş Airflow DAG dosyası.
Model ID: {model_id}
Alt model: {submodel_name}
Hedef: {target_col}
Üretim zamanı: {datetime.now().isoformat()}

Kurulum:
1. Bu dosyayı Airflow DAGs klasörüne kopyalayın (~airflow/dags/)
2. AutoGluon kurulu olduğundan emin olun: pip install autogluon.tabular
{('3. sentence-transformers kurulu olduğundan emin olun: pip install sentence-transformers' + chr(10)) if text_columns else ''}3. Model dizinini aşağıdaki MODEL_PATH konumuna kopyalayın
4. INPUT_CSV_PATH ve OUTPUT_CSV_PATH ayarlayın
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import os
import logging{embedding_imports}

logger = logging.getLogger(__name__)

MODEL_PATH = "/opt/airflow/models/{model_id}/agmodel"
INPUT_CSV_PATH = "/opt/airflow/data/input/prediction_input.csv"
OUTPUT_CSV_PATH = "/opt/airflow/data/output/predictions_{{ds}}.csv"
ERROR_LOG_PATH = "/opt/airflow/data/output/prediction_errors_{{ds}}.csv"
SUBMODEL_NAME = {json.dumps(submodel_name)}
TARGET_COLUMN = {json.dumps(target_col)}
TASK_TYPE = {json.dumps(task_type)}
FEATURE_COLUMNS = {json.dumps(feature_columns)}
TEXT_COLUMNS = {json.dumps(text_columns)}
{embedding_constants}
default_args = {{
    "owner": "tahmin-platformu",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}}

dag = DAG(
    dag_id="tahmin_{model_id[:8]}_{safe_submodel}",
    default_args=default_args,
    description={json.dumps(f"{target_col} tahmin pipeline")},
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "tahmin", "autogluon"],
)

{embedding_function}
def clean_dataframe(df):
    report = []
    original_count = len(df)
    empty_mask = df.isna().all(axis=1)
    empty_count = empty_mask.sum()
    if empty_count > 0:
        df = df[~empty_mask]
        report.append(f"{{empty_count}} bos satir silindi")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        inf_count = inf_mask.sum()
        if inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            report.append(f"'{{col}}': {{inf_count}} sonsuz deger NaN ile degistirildi")
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        empty_str_mask = df[col] == ""
        empty_str_count = empty_str_mask.sum()
        if empty_str_count > 0:
            df[col] = df[col].replace("", np.nan)
    feature_nan_pct = df[FEATURE_COLUMNS].isna().sum(axis=1) / len(FEATURE_COLUMNS)
    error_mask = feature_nan_pct > 0.5
    error_rows = df[error_mask].copy()
    error_rows["_error_reason"] = "Ozelliklerin %50den fazlasi NULL/NaN"
    df = df[~error_mask]
    report.append(f"Temizlendi: {{original_count}} -> {{len(df)}} satir")
    return df, error_rows, report


def validate_input(**kwargs):
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Girdi dosyasi bulunamadi: {{INPUT_CSV_PATH}}")
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sutunlar: {{missing}}")
    if len(df) == 0:
        raise ValueError("Girdi dosyasi bos")
    logger.info(f"Girdi dogrulandi: {{len(df)}} satir, {{len(df.columns)}} sutun")
    return True


def load_and_predict(**kwargs):
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(MODEL_PATH)
    input_df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')
    input_df, error_rows, cleaning_report = clean_dataframe(input_df)
    for msg in cleaning_report:
        logger.info(f"  [Temizlik] {{msg}}")
    if len(input_df) == 0:
        raise ValueError("Temizlik sonrasi gecerli satir kalmadi")
    # Keep human-readable copy before embedding
    original_input_df = input_df.copy()
    # NaN-safe: text/object sutunlardaki NaN degerleri bos string ile doldur
    if TEXT_COLUMNS:
        str_cols = input_df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            input_df[col] = input_df[col].fillna("")
        # Apply sentence embeddings to text columns
        text_cols_present = [c for c in TEXT_COLUMNS if c in input_df.columns]
        if text_cols_present:
            input_df = embed_text_columns(input_df, text_cols_present)
    predictions = predictor.predict(input_df, model=SUBMODEL_NAME)
    output_df = original_input_df.copy() if TEXT_COLUMNS else input_df.copy()
    output_df[f"{{TARGET_COLUMN}}_predicted"] = predictions
    null_feature_mask = original_input_df[FEATURE_COLUMNS].isna().any(axis=1) if TEXT_COLUMNS else input_df[FEATURE_COLUMNS].isna().any(axis=1)
    output_df[f"{{TARGET_COLUMN}}_has_null_features"] = null_feature_mask.astype(int)
    if TASK_TYPE == "classification":
        try:
            proba = predictor.predict_proba(input_df, model=SUBMODEL_NAME)
            for col in proba.columns:
                prob_series = proba[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
                output_df[f"{{TARGET_COLUMN}}_proba_{{col}}"] = prob_series.round(4)
        except Exception as e:
            logger.warning(f"Olasilik uretimi basarisiz: {{e}}")
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    output_path = OUTPUT_CSV_PATH.format(ds=kwargs.get("ds", "latest"))
    output_df.to_csv(output_path, index=False)
    logger.info(f"Tahminler kaydedildi: {{output_path}} ({{len(output_df)}} satir)")
    if len(error_rows) > 0:
        error_path = ERROR_LOG_PATH.format(ds=kwargs.get("ds", "latest"))
        os.makedirs(os.path.dirname(error_path), exist_ok=True)
        error_rows.to_csv(error_path, index=False)
    return output_path


validate_task = PythonOperator(
    task_id="girdi_dogrula",
    python_callable=validate_input,
    dag=dag,
)

predict_task = PythonOperator(
    task_id="tahmin_calistir",
    python_callable=load_and_predict,
    dag=dag,
)

validate_task >> predict_task
'''
    return dag_code


def generate_timeseries_airflow_dag(model_id: str, model_name: str, submodel_name: str,
                                     target_col: str, timestamp_column: str,
                                     item_id_column: str, prediction_length: int) -> str:
    """Generate an Airflow DAG for time series forecasting."""
    safe_submodel = re.sub(r'[^a-zA-Z0-9._-]', '_', submodel_name).lower()

    dag_code = f'''"""
Otomatik üretilmiş Airflow DAG dosyası — Zaman Serisi Tahmini.
Model ID: {model_id}
Alt model: {submodel_name}
Hedef: {target_col}
Tahmin uzunluğu: {prediction_length} adım
Üretim zamanı: {datetime.now().isoformat()}

Kurulum:
1. Bu dosyayı Airflow DAGs klasörüne kopyalayın (~airflow/dags/)
2. AutoGluon kurulu olduğundan emin olun: pip install autogluon.timeseries
3. Model dizinini aşağıdaki MODEL_PATH konumuna kopyalayın
4. INPUT_CSV_PATH = geçmiş veri (zaman damgası + hedef sütun içermeli)
5. OUTPUT_CSV_PATH = tahmin çıktısı
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "/opt/airflow/models/{model_id}/agmodel"
INPUT_CSV_PATH = "/opt/airflow/data/input/history_input.csv"
OUTPUT_CSV_PATH = "/opt/airflow/data/output/forecast_{{ds}}.csv"
SUBMODEL_NAME = {json.dumps(submodel_name)}
TARGET_COLUMN = {json.dumps(target_col)}
TIMESTAMP_COLUMN = {json.dumps(timestamp_column)}
ITEM_ID_COLUMN = {json.dumps(item_id_column)}
PREDICTION_LENGTH = {prediction_length}

default_args = {{
    "owner": "tahmin-platformu",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}}

dag = DAG(
    dag_id="ts_tahmin_{model_id[:8]}_{safe_submodel}",
    default_args=default_args,
    description={json.dumps(f"{target_col} zaman serisi tahmin pipeline")},
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "tahmin", "autogluon", "zaman-serisi"],
)


def clean_timeseries(df):
    """Zaman serisi verisini temizle — satir silmeden, ileri/geri doldurma."""
    report = []
    original_count = len(df)

    # Zaman damgasini parse et
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    report.append(f"Zaman damgasi parse edildi: {{TIMESTAMP_COLUMN}}")

    # Sonsuz degerleri NaN yap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        inf_count = inf_mask.sum()
        if inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            report.append(f"'{{col}}': {{inf_count}} sonsuz deger NaN ile degistirildi")

    # Bos metinleri temizle
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        if col != TIMESTAMP_COLUMN:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace("", np.nan)

    # Ileri/geri doldurma (satir silME — zaman serisini bozar)
    nan_before = df.isna().sum().sum()
    df = df.ffill().bfill()
    nan_after = df.isna().sum().sum()
    filled = nan_before - nan_after
    if filled > 0:
        report.append(f"{{filled}} eksik deger ileri/geri doldurma ile tamamlandi")

    report.append(f"Temizlendi: {{original_count}} satir korundu (satir silinmedi)")
    return df, report


def validate_input(**kwargs):
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Girdi dosyasi bulunamadi: {{INPUT_CSV_PATH}}")
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Zaman damgasi sutunu '{{TIMESTAMP_COLUMN}}' bulunamadi")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Hedef sutun '{{TARGET_COLUMN}}' bulunamadi")
    if len(df) == 0:
        raise ValueError("Girdi dosyasi bos")

    logger.info(f"Girdi dogrulandi: {{len(df)}} satir, {{len(df.columns)}} sutun")
    return True


def load_and_forecast(**kwargs):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    logger.info(f"Model yukleniyor: {{MODEL_PATH}}")
    predictor = TimeSeriesPredictor.load(MODEL_PATH)

    # Girdiyi oku ve temizle
    input_df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig')
    input_df, cleaning_report = clean_timeseries(input_df)
    for msg in cleaning_report:
        logger.info(f"  [Temizlik] {{msg}}")

    # Item ID sutunu yoksa dummy olustur
    id_col = ITEM_ID_COLUMN
    if id_col not in input_df.columns or id_col == "__item_id":
        id_col = "__item_id"
        input_df[id_col] = "series_0"
    else:
        input_df[id_col] = input_df[id_col].astype(str)

    # Sirala
    input_df = input_df.sort_values(by=[id_col, TIMESTAMP_COLUMN]).reset_index(drop=True)

    if len(input_df) == 0:
        raise ValueError("Temizlik sonrasi gecerli satir kalmadi")

    # TimeSeriesDataFrame olustur
    ts_df = TimeSeriesDataFrame.from_data_frame(
        input_df,
        id_column=id_col,
        timestamp_column=TIMESTAMP_COLUMN,
    )

    logger.info(f"Tahmin yapiliyor: {{PREDICTION_LENGTH}} adim, model={{SUBMODEL_NAME}}")
    predictions = predictor.predict(ts_df, model=SUBMODEL_NAME)

    # Ciktiyi kaydet
    output_df = predictions.reset_index()
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    output_path = OUTPUT_CSV_PATH.format(ds=kwargs.get("ds", "latest"))
    output_df.to_csv(output_path, index=False)

    logger.info(f"Tahminler kaydedildi: {{output_path}} ({{len(output_df)}} satir)")
    return output_path


validate_task = PythonOperator(
    task_id="girdi_dogrula",
    python_callable=validate_input,
    dag=dag,
)

forecast_task = PythonOperator(
    task_id="tahmin_calistir",
    python_callable=load_and_forecast,
    dag=dag,
)

validate_task >> forecast_task
'''
    return dag_code


def estimate_cost(inference_time_sec: float, num_rows: int, num_columns: int,
                  frequency: str, avg_row_bytes: int = 200) -> dict:
    # Guard: clamp all numeric inputs to non-negative to prevent negative costs
    inference_time_sec = max(float(inference_time_sec), 0.0)
    num_rows = max(int(num_rows), 0)
    num_columns = max(int(num_columns), 0)
    avg_row_bytes = max(int(avg_row_bytes), 1)

    freq_map = {
        "hourly": 720, "every_3_hours": 240, "twice_daily": 60,
        "daily": 30, "weekly": 4.3, "monthly": 1,
    }
    executions_per_month = freq_map.get(frequency, 30)
    sample_size = 100
    time_per_row = inference_time_sec / max(sample_size, 1)
    processing_time_sec = time_per_row * num_rows
    total_job_time_sec = processing_time_sec * 1.3
    cost_per_core_second = 0.000120
    cores_used = 2
    compute_cost = total_job_time_sec * cores_used * cost_per_core_second * executions_per_month
    model_size_gb = 0.5
    data_per_run_gb = (num_rows * num_columns * avg_row_bytes) / (1024**3)
    monthly_data_gb = data_per_run_gb * executions_per_month * 2
    storage_cost = (model_size_gb + monthly_data_gb) * 0.10
    total_monthly = compute_cost + storage_cost

    return {
        "compute_cost": round(compute_cost, 2),
        "storage_cost": round(storage_cost, 2),
        "total_monthly": round(total_monthly, 2),
        "executions_per_month": round(executions_per_month, 1),
        "estimated_job_duration_sec": round(total_job_time_sec, 2),
        "time_per_row_ms": round(time_per_row * 1000, 4),
        "assumptions": {
            "server": "8 çekirdek Xeon, 64GB RAM (mevcut altyapı)",
            "cost_per_core_second": "$0.000120",
            "storage_rate": "$0.10/GB/ay",
            "note": "Sadece hesaplama — donanım zaten mevcut",
        }
    }


# ══════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════

def train_model(job_id: str, model_id: str, csv_path: str, target_col: str,
                task_type: str, preset: str, model_name: str, username: str = "",
                visibility: str = "private", timestamp_column: str = None,
                item_id_column: str = None, prediction_length: int = 10):
    resource_task_id = f"training_{job_id}"
    _heartbeat_stop = threading.Event()
    if not model_ref_counter.acquire(model_id):
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = "Model siliniyor, eğitim başlatılamadı."
        user_action_tracker.unregister(username, "training")
        release_model_quota(username)
        return
    try:
        # Acquire resources
        profile = resource_manager.get_profile("training_tabular")
        if not resource_manager.try_acquire(resource_task_id, "training",
                                             username=username,
                                             vram_mb=profile["vram_mb"],
                                             ram_mb=profile["ram_mb"]):
            training_jobs[job_id]["status"] = "error"
            training_jobs[job_id]["error"] = "Yeterli kaynak yok. Devam eden işlerin bitmesini bekleyin."
            return  # finally block handles unregister

        training_jobs.update_fields(job_id, status="training")

        # Keep heartbeat alive during blocking fit() calls to prevent stale-job cleanup
        _heartbeat_stop = threading.Event()
        def _heartbeat_loop():
            while not _heartbeat_stop.wait(timeout=120):
                training_jobs.update_fields(job_id, _heartbeat="alive")
        _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name=f"hb-{job_id[:8]}")
        _heartbeat_thread.start()

        try:
            df = _read_csv_with_fallback(csv_path)
        except Exception as e:
            raise ValueError(f"CSV dosyası okunamadı. Dosya formatını kontrol edin: {str(e)[:100]}")

        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            raise ValueError(
                f"CSV dosyasında tekrarlanan sütun adları var: {', '.join(dupes[:5])}. "
                f"Lütfen sütun adlarını benzersiz yapın."
            )

        # ══════════════════════════════════════════════════════════
        #  TIME SERIES TRAINING PATH
        # ══════════════════════════════════════════════════════════
        if task_type == "timeseries":
            if not AUTOGLUON_TS_AVAILABLE:
                raise ValueError("Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin.")

            df, cleaning_report = clean_dataframe(df, context="timeseries",
                                                  timestamp_column=timestamp_column,
                                                  item_id_column=item_id_column)

            if target_col not in df.columns:
                raise ValueError(f"Hedef sütun '{target_col}' veride bulunamadı")
            if timestamp_column not in df.columns:
                raise ValueError(f"Zaman damgası sütunu '{timestamp_column}' veride bulunamadı")

            # Ensure datetime type
            try:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Zaman damgası sütunundaki tarihler ayrıştırılamadı: {str(e)[:100]}")

            # Validate target column is numeric
            if df[target_col].dtype.kind not in ("i", "f"):
                raise ValueError(
                    f"Zaman serisi hedef sütunu '{target_col}' sayısal olmalı. "
                    f"Bulunan tür: {df[target_col].dtype}"
                )

            # Reject if target column has too many NaN (ffill would fabricate targets)
            target_nan_ratio = df[target_col].isna().sum() / max(len(df), 1)
            if target_nan_ratio > 0.5:
                raise ValueError(
                    f"Hedef sütun '{target_col}' çok fazla eksik değer içeriyor "
                    f"(%{target_nan_ratio:.0%}). Zaman serisi için hedef verisi yoğun olmalı."
                )

            # Handle item_id: if user has single series, create a dummy ID
            if item_id_column and item_id_column in df.columns:
                id_col = item_id_column
                # Drop rows with NaN item_id — astype(str) would convert them to literal "nan",
                # silently grouping unrelated rows into a junk series
                _nan_id_count = df[id_col].isna().sum()
                if _nan_id_count > 0:
                    df = df.dropna(subset=[id_col]).reset_index(drop=True)
                    log.warning(f"[TS] Dropped {_nan_id_count} rows with missing item_id")
                df[id_col] = df[id_col].astype(str)
            else:
                id_col = "__item_id"
                df[id_col] = "series_0"
                item_id_column = id_col

            # Sort by item_id + timestamp for proper ordering
            df = df.sort_values(by=[id_col, timestamp_column]).reset_index(drop=True)

            if len(df) < 20:
                raise ValueError(f"Temizlik sonrası sadece {len(df)} satır kaldı. En az 20 satır gerekli.")

            # Build TimeSeriesDataFrame — NO random split, predictor handles chrono split
            try:
                ts_df = TimeSeriesDataFrame.from_data_frame(
                    df,
                    id_column=id_col,
                    timestamp_column=timestamp_column,
                )
            except (ValueError, KeyError) as e:
                raise ValueError(f"Zaman serisi verisi oluşturulamadı. Veri formatını kontrol edin: {str(e)[:100]}")

            # Validate per-item: each series must have enough rows for the prediction length
            _series_lengths = df.groupby(id_col).size()
            _min_series_len = int(_series_lengths.min())
            if prediction_length >= _min_series_len // 2:
                raise ValueError(
                    f"Tahmin uzunluğu ({prediction_length}) en kısa serinin yarısından "
                    f"({_min_series_len // 2}) küçük olmalı. "
                    f"En kısa seri {_min_series_len} satır içeriyor."
                )

            if df[target_col].nunique() < 2:
                raise ValueError("Hedef sütundaki tüm değerler aynı. Zaman serisi tahmini yapılamaz.")

            ag_model_path = str(MODELS_DIR / model_id / "agmodel")

            ts_predictor = TimeSeriesPredictor(
                target=target_col,
                prediction_length=prediction_length,
                path=ag_model_path,
                eval_metric="MASE",
            )

            ts_predictor.fit(
                train_data=ts_df,
                presets=preset if preset in ("fast_training", "medium_quality", "good_quality",
                                             "high_quality", "best_quality") else "medium_quality",
                time_limit=600,
                verbosity=1,
            )

            leaderboard = ts_predictor.leaderboard(silent=True)
            if len(leaderboard) == 0:
                raise ValueError("Hiçbir zaman serisi modeli eğitilemedi. Veri kalitesini kontrol edin.")

            feature_cols = [c for c in df.columns
                           if c not in (target_col, timestamp_column, id_col)]

            submodels = []
            for _, row in leaderboard.iterrows():
                sm_name = str(row["model"])
                # AutoGluon negates error metrics: score_val = -MASE
                # So -(-MASE) = positive MASE. Lower MASE = better.
                raw_score_val = float(row.get("score_val", 0))
                mase_value = abs(raw_score_val)  # raw MASE (lower = better)

                submodels.append({
                    "id": str(uuid.uuid4())[:8],
                    "name": sm_name,
                    "score": round(mase_value, 4),
                    "score_internal": round(mase_value, 4),
                    "ci_low": None,
                    "ci_high": None,
                    "n_test": prediction_length,
                    "inference_time_sec": -1.0,
                    "cpu_category": "Orta",
                    # Disable SQL for time series; Airflow IS supported
                    "sql_support": {
                        "easy_sql": False,
                        "sql_method": "none",
                        "label": "Desteklenmiyor (Zaman Serisi)",
                        "verified": True,
                    },
                    "airflow_support": {
                        "supported": True,
                        "label": "Günlük çalışabilir",
                    },
                    "model_type": "timeseries",
                })

            # MASE: lower is better → sort ascending
            submodels.sort(key=lambda x: x["score"], reverse=False)

            meta = {
                "id": model_id,
                "name": model_name,
                "created_at": datetime.now().isoformat(),
                "target_column": target_col,
                "task_type": "timeseries",
                "problem_type": "timeseries",
                "preset": preset,
                "eval_metric": "MASE",
                "score_lower_is_better": True,
                "num_rows": len(df),
                "num_train_rows": len(df),
                "num_test_rows": prediction_length,
                "test_split_pct": 0.0,
                "num_columns": len(df.columns),
                "feature_columns": feature_cols,
                "column_types": {col: str(df[col].dtype) for col in df.columns},
                "csv_filename": os.path.basename(csv_path),
                "submodels": submodels,
                "best_model": submodels[0]["name"] if submodels else None,
                "best_score": submodels[0]["score"] if submodels else 0,
                "total_predictions": 0,
                "avg_row_bytes": int(df.memory_usage(deep=True).sum() / max(len(df), 1)),
                "cleaning_report": cleaning_report,
                "owner": username,
                "visibility": visibility,
                "endorsed": False,
                "endorsed_by": None,
                "view_count": 0,
                # Time series specific metadata
                "timestamp_column": timestamp_column,
                "item_id_column": item_id_column,
                "prediction_length": prediction_length,
            }

            save_model_meta(model_id, meta)
            add_activity("trained", model_id, model_name,
                         f"bir zaman serisi modeli eğitti.",
                         username=username)

            # Cache the freshly trained model so predictions are instant
            model_cache.put(model_id, ts_predictor, "timeseries")

            training_jobs[job_id]["status"] = "done"
            training_jobs[job_id]["model_id"] = model_id
            return

        # ══════════════════════════════════════════════════════════
        #  UNIFIED TABULAR TRAINING PATH
        #  (classification / regression — with auto text embedding)
        # ══════════════════════════════════════════════════════════
        df, cleaning_report = clean_dataframe(df, context="training")

        if target_col not in df.columns:
            raise ValueError(f"Hedef sütun '{target_col}' temizlik sonrası veride bulunamadı")

        target_nan_count = df[target_col].isna().sum()
        if target_nan_count > 0:
            df = df.dropna(subset=[target_col])

        if len(df) < 20:
            raise ValueError(f"Temizlik sonrası sadece {len(df)} satır kaldı. En az 20 satır gerekli.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Also handle mixed-type columns that may contain infinity strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].replace(['inf', '-inf', 'Inf', '-Inf'], np.nan)

        # ── Auto-detect text columns and convert to embeddings ──
        # A column is "text" if it's object/string AND has average length > 30 chars
        # (short strings like "Yes"/"No" or "Male"/"Female" are just categoricals)
        text_columns = []
        embedding_model_name = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if col == target_col:
                    continue
                avg_len = df[col].fillna("").astype(str).str.len().mean()
                if avg_len > 30:
                    text_columns.append(col)

        # Save original feature names and types (human-readable) before embedding
        feature_cols_original = [c for c in df.columns if c != target_col]
        column_types_original = {col: str(df[col].dtype) for col in df.columns}

        if not feature_cols_original:
            raise ValueError("CSV'de en az bir özellik sütunu olmalı. Yalnızca hedef sütun tespit edildi.")

        if text_columns:
            embedding_model_name = DEFAULT_EMBEDDING_MODEL
            print(f"[Eğitim] Metin sütunları algılandı: {text_columns}")
            print(f"[Eğitim] Embedding modeli: {embedding_model_name}")
            for col in text_columns:
                df[col] = df[col].fillna("")
            df = _embed_text_columns(df, text_columns, embedding_model_name)
            _save_text_pipeline_config(model_id, text_columns, embedding_model_name)

        # Feature columns in the (possibly embedded) space
        feature_cols_embedded = [c for c in df.columns if c != target_col]

        from sklearn.model_selection import train_test_split

        test_size = 0.2

        # ── Infer problem type ──
        # Works for all cases: pure numeric targets → regression,
        # text/categorical targets (binary or multi-class) → classification
        target_unique = df[target_col].nunique()
        target_is_numeric = df[target_col].dtype.kind in ("i", "f")

        if target_unique < 2:
            raise ValueError(f"Hedef sütunda en az 2 farklı değer olmalı. Bulunan: {target_unique}")

        if task_type == "classification" or (not target_is_numeric) or (target_is_numeric and target_unique <= 20):
            # Classification: binary (2 classes) or multiclass (3+)
            problem_type = "binary" if target_unique == 2 else "multiclass"
            eval_metric = "accuracy"
            try:
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=42, stratify=df[target_col])
            except ValueError:
                train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        else:
            # Regression: continuous numeric target
            problem_type = "regression"
            eval_metric = "root_mean_squared_error"
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        if len(test_df) < 2:
            test_df = train_df.sample(n=min(5, len(train_df)), random_state=42)
        print(f"[Eğitim] Bölme: {len(train_df)} eğitim, {len(test_df)} test satırı")
        print(f"[Eğitim] Problem tipi: {problem_type}" +
              (f" | Metin embedding: {len(text_columns)} sütun" if text_columns else ""))

        train_data = TabularDataset(train_df)

        ag_model_path = str(MODELS_DIR / model_id / "agmodel")

        predictor = TabularPredictor(
            label=target_col, problem_type=problem_type,
            eval_metric=eval_metric, path=ag_model_path,
        )

        predictor.fit(train_data=train_data, presets=preset, time_limit=600, verbosity=1,
                      num_cpus=resource_manager.training_cpu_count,
                      num_gpus=0)  # GPU reserved for Whisper/LLM; tree models are CPU anyway

        training_jobs.update_fields(job_id, _phase="leaderboard")
        leaderboard = predictor.leaderboard(silent=True)
        if len(leaderboard) == 0:
            raise ValueError("Hiçbir model eğitilemedi. Veri kalitesini veya hedef sütunu kontrol edin.")

        sample_size = min(100, len(test_df))
        sample_features = test_df.drop(columns=[target_col]).head(sample_size)
        test_y = test_df[target_col]

        submodels = []
        for _sm_idx, row in leaderboard.iterrows():
            sm_name = row["model"]
            training_jobs.update_fields(job_id, _phase=f"eval_{_sm_idx}")
            internal_score = float(row.get("score_val", 0))

            try:
                test_predictions = predictor.predict(test_df.drop(columns=[target_col]), model=sm_name)
                if problem_type in ("binary", "multiclass"):
                    holdout_score = float((test_predictions == test_y).mean())
                else:
                    residuals = test_predictions - test_y
                    rmse = float(np.sqrt((residuals ** 2).mean()))
                    y_range = float(test_y.max() - test_y.min())
                    # Floor y_range to avoid misleadingly low scores for near-constant targets
                    y_range = max(y_range, abs(float(test_y.mean())) * 0.01, 1e-6)
                    holdout_score = max(0.0, min(1.0, 1.0 - (rmse / y_range)))
            except Exception as _eval_err:
                log.warning(f"[Training] Holdout evaluation failed for {sm_name}: {_eval_err}")
                # Conservative fallback: 0.0 (unknown quality) instead of misleading abs(internal_score)
                holdout_score = 0.0

            n_test = len(test_df)
            # Wilson CI is only meaningful for classification (proportion-based)
            if problem_type in ("binary", "multiclass"):
                ci_low, ci_high = _wilson_ci(holdout_score, n_test, z=1.96)
            else:
                ci_low, ci_high = None, None
            inf_time = measure_inference_time(predictor, sample_features, sm_name)
            sql_support = verify_sql_support(predictor, sm_name, feature_cols_embedded, target_col)
            airflow_support = get_airflow_support(sm_name)

            submodels.append({
                "id": str(uuid.uuid4())[:8],
                "name": sm_name,
                "score": round(holdout_score, 4),
                "score_internal": round(abs(internal_score), 4),
                "ci_low": round(ci_low, 4) if ci_low is not None else None,
                "ci_high": round(ci_high, 4) if ci_high is not None else None,
                "n_test": n_test,
                "inference_time_sec": inf_time,
                "cpu_category": "Düşük" if inf_time < 0.05 else ("Orta" if inf_time < 0.2 else "Yüksek"),
                "sql_support": sql_support,
                "airflow_support": airflow_support,
                "model_type": get_model_type_category(sm_name),
            })

        submodels.sort(key=lambda x: x["score"], reverse=True)

        # Filter: keep models that have at least one export path
        all_submodels = submodels
        submodels = [sm for sm in submodels if
                     sm["airflow_support"].get("supported") or
                     sm["sql_support"].get("easy_sql")]
        # Fall back to all submodels if filtering removes everything
        if not submodels and all_submodels:
            submodels = all_submodels
            log.info(f"[Training] No exportable submodels — keeping all {len(submodels)} models")
        hidden_count = len(all_submodels) - len(submodels)

        meta = {
            "id": model_id,
            "name": model_name,
            "created_at": datetime.now().isoformat(),
            "target_column": target_col,
            "task_type": task_type,
            "problem_type": problem_type,
            "preset": preset,
            "eval_metric": eval_metric,
            "num_rows": len(df),
            "num_train_rows": len(train_df),
            "num_test_rows": len(test_df),
            "test_split_pct": test_size,
            "num_columns": len(column_types_original),
            "feature_columns": feature_cols_original,
            "column_types": column_types_original,
            "csv_filename": os.path.basename(csv_path),
            "submodels": submodels,
            "best_model": submodels[0]["name"] if submodels else None,
            "best_score": submodels[0]["score"] if submodels else 0,
            "total_predictions": 0,
            "avg_row_bytes": int(df.memory_usage(deep=True).sum() / max(len(df), 1)),
            "cleaning_report": cleaning_report,
            "owner": username,
            "visibility": visibility,
            "endorsed": False,
            "endorsed_by": None,
            "view_count": 0,
        }

        # Add text embedding metadata if text columns were detected
        if text_columns:
            meta["text_columns"] = text_columns
            meta["embedding_model"] = embedding_model_name
            meta["feature_columns_embedded"] = feature_cols_embedded
            try:
                _emb_model = _get_sentence_model(embedding_model_name)
                meta["embedding_dim"] = _emb_model.get_sentence_embedding_dimension()
            except Exception as _dim_err:
                log.warning(f"[Training] Could not save embedding_dim: {_dim_err}")

        save_model_meta(model_id, meta)

        add_activity("trained", model_id, model_name,
                     f"bir tahmin modeli eğitti.",
                     username=username)

        # Cache the freshly trained model so predictions are instant
        model_cache.put(model_id, predictor, task_type)

        training_jobs[job_id]["status"] = "done"
        training_jobs[job_id]["model_id"] = model_id

    except MemoryError:
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["error"] = "Bellek yetersiz. Daha küçük veri seti veya daha düşük kalite ayarı ile tekrar deneyin."
    except Exception as e:
        traceback.print_exc()
        training_jobs[job_id]["status"] = "error"
        err_str = str(e).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            training_jobs[job_id]["error"] = "GPU belleği yetersiz. Lütfen tekrar deneyin."
        else:
            training_jobs[job_id]["error"] = _sanitize_error(e)
        # Clean up orphan model directory if training never completed (no meta.json)
        try:
            meta_path = MODELS_DIR / model_id / "meta.json"
            if not meta_path.exists():
                model_dir = MODELS_DIR / model_id
                if model_dir.exists():
                    shutil.rmtree(model_dir, ignore_errors=True)
                    log.info(f"[Training] Cleaned up orphan model dir {model_id[:8]}… after failed training")
        except OSError:
            pass
    finally:
        # Stop heartbeat thread
        _heartbeat_stop.set()
        # Free GPU memory before releasing resource tracking
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        # Skip releasing if stale-job detector already released these resources
        if not training_jobs.get(job_id, {}).get("_stale_released"):
            model_ref_counter.release(model_id)
            resource_manager.release(resource_task_id)
            user_action_tracker.unregister(username, "training")
        # Release pending quota reservation (model is now saved or failed)
        release_model_quota(username)


# ══════════════════════════════════════════════════════════════════
#  AUDIO EVALUATION (LLM-AS-A-JUDGE) PIPELINE
# ══════════════════════════════════════════════════════════════════

# ── Cached Whisper model (loaded on demand, unloaded after idle) ──
_whisper_model = None
_whisper_model_lock = threading.Lock()
_whisper_last_used = 0.0
_whisper_measured_vram_mb = 0  # Measured on first load — replaces profile guess
_WHISPER_IDLE_TIMEOUT = int(os.environ.get("WHISPER_IDLE_TIMEOUT", "300"))  # 5 min default

def _get_whisper_model():
    """Lazy-load and cache the Whisper model. Downloads to ./stt/ if missing.
    Registers resource reservation with ResourceManager."""
    global _whisper_model, _whisper_last_used, _whisper_measured_vram_mb
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                # Determine if whisper will actually use GPU
                # "auto" means faster-whisper/ctranslate2 decides at runtime
                whisper_uses_gpu = False
                if WHISPER_DEVICE == "cpu":
                    whisper_uses_gpu = False
                elif WHISPER_DEVICE == "cuda":
                    whisper_uses_gpu = True
                else:  # "auto" — check what ctranslate2 can actually see
                    try:
                        import ctranslate2
                        # Only trust ct2 GPU detection if we also confirmed VRAM exists
                        whisper_uses_gpu = (ctranslate2.get_cuda_device_count() > 0
                                            and resource_manager.total_vram_mb > 0)
                    except Exception:
                        whisper_uses_gpu = False

                profile_name = "whisper_gpu" if whisper_uses_gpu else "whisper_cpu"
                # Use measured VRAM if available, otherwise profile default
                vram_needed = _whisper_measured_vram_mb + 200 if _whisper_measured_vram_mb > 0 \
                    else resource_manager.PROFILES[profile_name]["vram_mb"]
                ram_needed = resource_manager.PROFILES[profile_name]["ram_mb"]

                # Solution 7: Pre-emptive eviction if VRAM is tight
                if whisper_uses_gpu and vram_needed > 0:
                    if not _ensure_vram_available(vram_needed):
                        log.warning(f"[VRAM] Pre-eviction insufficient: needed {vram_needed}MB")

                if not resource_manager.try_acquire("whisper_model", "whisper_load",
                                                     vram_mb=vram_needed,
                                                     ram_mb=ram_needed):
                    raise RuntimeError(
                        "Whisper modeli için yeterli kaynak yok. "
                        f"Gereken: {vram_needed}MB VRAM, {ram_needed}MB RAM. "
                        f"Cihaz: {profile_name}. "
                        "Devam eden işlerin bitmesini bekleyin."
                    )
                try:
                    # Solution 2: Measure VRAM delta during Whisper load
                    _w_before = resource_manager.get_actual_free_vram_mb()
                    if os.path.isfile(os.path.join(WHISPER_MODEL_DIR, "model.bin")):
                        log.info(f"  [Whisper] Loading local model from {WHISPER_MODEL_DIR}")
                        _whisper_model = WhisperModel(WHISPER_MODEL_DIR, device=WHISPER_DEVICE,
                                                       compute_type=WHISPER_COMPUTE_TYPE,
                                                       cpu_threads=resource_manager.cpu_count)
                    else:
                        log.info(f"  [Whisper] model.bin not found in {WHISPER_MODEL_DIR}, downloading {WHISPER_MODEL_REPO}...")
                        os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
                        _whisper_model = WhisperModel(WHISPER_MODEL_REPO, device=WHISPER_DEVICE,
                                                       compute_type=WHISPER_COMPUTE_TYPE,
                                                       download_root=WHISPER_MODEL_DIR,
                                                       cpu_threads=resource_manager.cpu_count)
                    _w_after = resource_manager.get_actual_free_vram_mb()
                    _whisper_measured_vram_mb = max(0, _w_before - _w_after)
                    log.info(f"  [Whisper] Model loaded. Measured VRAM: {_whisper_measured_vram_mb}MB")
                except Exception:
                    resource_manager.release("whisper_model")
                    raise
    with _whisper_model_lock:
        _whisper_last_used = time.time()
    return _whisper_model


def _unload_whisper_model(min_idle_seconds: float = 0):
    """Unload Whisper model to free GPU VRAM and RAM.
    If min_idle_seconds > 0, re-checks idle time under lock to prevent TOCTOU race."""
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is not None:
            # Re-check idle time under lock to avoid unloading a freshly-used model
            if min_idle_seconds > 0 and (time.time() - _whisper_last_used) < min_idle_seconds:
                return
            _whisper_model = None
            resource_manager.release("whisper_model")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            log.info("[Whisper] Model unloaded to free resources")


def _whisper_idle_monitor():
    """Background thread that unloads Whisper if idle too long.
    Never unloads while audio jobs are actively processing."""
    global _whisper_last_used
    while True:
        time.sleep(60)
        with _whisper_model_lock:
            model_loaded = _whisper_model is not None
            last_used = _whisper_last_used
        if model_loaded:
            # Don't unload if any audio jobs are actively processing
            active = user_action_tracker.get_active()
            has_active_audio = any(
                "audio_eval" in actions or "audio_predict" in actions
                for actions in active.values()
            )
            if has_active_audio:
                with _whisper_model_lock:
                    _whisper_last_used = time.time()  # Keep alive
                continue

            idle_time = time.time() - last_used
            if idle_time > _WHISPER_IDLE_TIMEOUT:
                log.info(f"[Whisper] Idle for {idle_time:.0f}s, unloading to free resources")
                _unload_whisper_model(min_idle_seconds=_WHISPER_IDLE_TIMEOUT)

# Start whisper idle monitor
threading.Thread(target=_whisper_idle_monitor, daemon=True).start()

def _transcribe_audio(audio_path: str, language: str) -> str:
    """Transcribe an audio file using faster-whisper."""
    global _whisper_last_used
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin.")
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise ValueError(f"Ses dosyası boş veya bulunamadı: {os.path.basename(audio_path)}")
    model = _get_whisper_model()
    lang_map = {
        "turkish": "tr", "english": "en", "german": "de",
        "french": "fr", "spanish": "es", "italian": "it",
        "dutch": "nl", "portuguese": "pt", "russian": "ru",
        "arabic": "ar", "chinese": "zh", "japanese": "ja",
        "korean": "ko",
    }
    lang_code = lang_map.get(language.lower(), language.lower()[:2])
    segments, info = model.transcribe(audio_path, language=lang_code, beam_size=1,
                                       temperature=0.0, condition_on_previous_text=True,
                                       vad_filter=True, without_timestamps=True)
    transcript = " ".join(seg.text.strip() for seg in segments)
    # Keep whisper alive during batch processing
    with _whisper_model_lock:
        _whisper_last_used = time.time()
    return transcript


_LLM_SYSTEM_TEMPLATE = """You are an expert call quality evaluator. You will receive a transcript and evaluation instructions.

CRITICAL RULES:
1. Respond with ONLY a valid JSON object — no markdown, no explanation, no extra text.
2. The JSON MUST match this exact schema: {json_template}
3. For classification fields: output a single concise label string.
4. For numeric fields: output a single number.
5. Include 'summary_reasoning' as the last key with a brief explanation of your evaluation."""

_LLM_USER_TEMPLATE = """=== EVALUATION INSTRUCTIONS ===
{user_prompt}

=== TRANSCRIPT ===
{transcript}

Respond with ONLY the JSON object matching the schema above."""


def _build_llm_prompt(user_prompt: str, transcript: str, schema: list) -> list:
    """Build the chat messages for the LLM call with strict JSON output instructions."""
    schema_desc_parts = []
    for var in schema:
        vtype = var.get("type", "classification")
        if vtype == "classification":
            schema_desc_parts.append(f'  "{var["name"]}": "<string: your classification label>"')
        else:
            schema_desc_parts.append(f'  "{var["name"]}": <number: your numeric estimate>')
    schema_desc_parts.append('  "summary_reasoning": "<string: brief reasoning for your decisions>"')
    json_template = "{\n" + ",\n".join(schema_desc_parts) + "\n}"

    system_msg = _LLM_SYSTEM_TEMPLATE.format(json_template=json_template)
    user_msg = _LLM_USER_TEMPLATE.format(user_prompt=user_prompt, transcript=transcript)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


_llama_restart_event = threading.Event()
_llama_restart_event.set()  # Initially "not restarting"
_llama_restarting = False
_llama_last_fail = 0.0
_LLAMA_RESTART_COOLDOWN = 300  # 5 minutes between restart attempts after failure

def _ensure_llama_running():
    """Check if llama-server subprocess is alive; restart if crashed.
    If another thread is restarting, waits for it to finish (up to 130s)."""
    global _llama_process, _llama_log_fh, _llama_restarting, _llama_last_fail
    if LLAMA_BUNDLED == "false":
        return
    i_am_restarter = False
    with _llama_lock:
        if _llama_restarting:
            # Another thread is already restarting — we just wait
            pass
        elif _llama_process is not None and _llama_process.poll() is not None:
            exit_code = _llama_process.returncode
            log.warning(f"[LLM] llama-server crashed (exit code {exit_code}) — restarting...")
            _llama_process = None
            if _llama_log_fh:
                _llama_log_fh.close()
                _llama_log_fh = None
            _llama_restarting = True
            i_am_restarter = True
            _llama_restart_event.clear()
        elif _llama_process is None and not _llama_restarting:
            if time.time() - _llama_last_fail < _LLAMA_RESTART_COOLDOWN:
                return  # Don't retry too soon after a failure
            _llama_restarting = True
            i_am_restarter = True
            _llama_restart_event.clear()
        else:
            return  # Process is running fine

    # Only the thread that set _llama_restarting=True performs the restart
    if i_am_restarter:
        try:
            # Check VRAM availability before restart to avoid GPU OOM race with Whisper
            llm_vram = resource_manager.PROFILES.get("llm_external", {}).get("vram_mb", 0)
            if llm_vram > 0:
                actual_free = resource_manager.get_actual_free_vram_mb()
                if actual_free < llm_vram:
                    log.warning(f"[LLM] Deferring restart — insufficient VRAM ({actual_free}MB free, need {llm_vram}MB)")
                    _llama_last_fail = time.time()
                    with _llama_lock:
                        _llama_restarting = False
                    _llama_restart_event.set()
                    return
            _start_bundled_llama()
            if _llama_process is None:
                _llama_last_fail = time.time()
        except Exception:
            _llama_last_fail = time.time()
            raise
        finally:
            with _llama_lock:
                _llama_restarting = False
            _llama_restart_event.set()
        return

    # Don't block HTTP threads waiting for restart — fail fast with 503
    if not _llama_restart_event.wait(timeout=5):
        raise RuntimeError("LLM yeniden başlatılıyor, lütfen birkaç saniye sonra tekrar deneyin.")


def _call_llm(messages: list, max_attempts: int = 2) -> dict:
    """Call the local llama.cpp API and parse the JSON response.

    Makes up to max_attempts total attempts (1 initial + N-1 retries) on:
    - Connection errors / timeouts
    - HTTP 5xx errors
    - Malformed JSON responses
    Uses exponential backoff: 2s, 4s between attempts.
    Worst-case: 120s timeout × 2 attempts + 6s backoff = ~246s.
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin.")

    _ensure_llama_running()

    payload = {
        "model": LLAMA_CPP_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = http_requests.post(LLAMA_CPP_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Guard against unexpected LLM API response structure
            try:
                raw_content = data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                raise json.JSONDecodeError(
                    f"LLM API yanıtı beklenmeyen formatta: {e}. Yanıt: {str(data)[:200]}",
                    str(data), 0
                )

            # Try to extract JSON from the response (handle markdown fences)
            json_str = raw_content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Try brace extraction as fallback — match balanced braces
            if not json_str.startswith("{"):
                start = json_str.find("{")
                if start != -1:
                    depth = 0
                    end = start
                    for _ci in range(start, len(json_str)):
                        if json_str[_ci] == '{': depth += 1
                        elif json_str[_ci] == '}': depth -= 1
                        if depth == 0:
                            end = _ci + 1
                            break
                    if end > start:
                        json_str = json_str[start:end]

            result = json.loads(json_str)
            if not isinstance(result, dict) or not result:
                raise json.JSONDecodeError("LLM boş veya geçersiz JSON döndü", json_str, 0)
            if attempt > 1:
                log.info(f"  [LLM] Succeeded on attempt {attempt}/{max_attempts}")
            return result

        except json.JSONDecodeError as e:
            last_error = e
            log.warning(f"  [LLM] Attempt {attempt}/{max_attempts}: malformed JSON — {e}")
            if attempt < max_attempts:
                # For JSON errors, slightly adjust temperature to get different output
                payload["temperature"] = min(0.3, payload["temperature"] + 0.1)
        except http_requests.exceptions.HTTPError as e:
            last_error = e
            status_code = e.response.status_code if e.response is not None else 0
            log.warning(f"  [LLM] Attempt {attempt}/{max_attempts}: HTTP {status_code} — {e}")
            if status_code < 500 and status_code != 429:
                raise  # Don't retry 4xx client errors (except 429 rate limit)
        except (http_requests.exceptions.Timeout,
                http_requests.exceptions.ReadTimeout) as e:
            last_error = e
            log.warning(f"  [LLM] Attempt {attempt}/{max_attempts}: timeout — {e}. Not retrying (server is overloaded).")
            break  # Don't retry on timeout — piling requests onto a slow LLM makes it worse
        except http_requests.exceptions.ConnectionError as e:
            last_error = e
            log.warning(f"  [LLM] Attempt {attempt}/{max_attempts}: connection error — {e}")
        except Exception as e:
            last_error = e
            log.warning(f"  [LLM] Attempt {attempt}/{max_attempts}: unexpected error — {e}")

        if attempt < max_attempts:
            backoff = 2 ** attempt  # 2s, 4s, 8s
            log.info(f"  [LLM] Retrying in {backoff}s...")
            time.sleep(backoff)

    # All retries exhausted
    raise RuntimeError(
        f"LLM çağrısı {max_attempts} denemeden sonra başarısız oldu. "
        f"Son hata: {type(last_error).__name__}: {last_error}"
    )


def _compute_evaluation_metrics(results: list, schema: list) -> dict:
    """Compute accuracy for classification and MAE/RMSE for regression variables."""
    metrics = {}
    for var in schema:
        var_name = var["name"]
        var_type = var.get("type", "classification")
        actuals = []
        predicted = []
        for r in results:
            actual_val = r.get("actuals", {}).get(var_name)
            pred_val = r.get("predicted", {}).get(var_name)
            if actual_val is not None and pred_val is not None:
                actuals.append(actual_val)
                predicted.append(pred_val)

        if len(actuals) == 0:
            metrics[var_name] = {"type": var_type, "n": 0, "error": "Veri yok"}
            continue

        if var_type == "classification":
            correct = sum(1 for a, p in zip(actuals, predicted) if str(a).strip().casefold() == str(p).strip().casefold())
            accuracy = correct / len(actuals) if actuals else 0
            metrics[var_name] = {
                "type": "classification",
                "n": len(actuals),
                "correct": correct,
                "accuracy": round(accuracy * 100, 2),
            }
        else:  # regression
            try:
                num_actuals = [float(a) for a in actuals]
                num_predicted = [float(p) for p in predicted]
                diffs = [(a - p) for a, p in zip(num_actuals, num_predicted)]
                mae = sum(abs(d) for d in diffs) / len(diffs)
                rmse = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs))
                metrics[var_name] = {
                    "type": "regression",
                    "n": len(num_actuals),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                }
            except (ValueError, TypeError) as e:
                metrics[var_name] = {"type": "regression", "n": len(actuals), "error": _sanitize_error(e)}

    return metrics


def audio_evaluate_pipeline(job_id: str, model_id: str, model_name: str,
                            audio_files: list, schema: list, prompt: str,
                            language: str, actuals_map: dict,
                            username: str = "", visibility: str = "private"):
    """
    Main pipeline for audio evaluation.
    audio_files: list of dicts {"path": str, "filename": str}
    schema: list of dicts {"name": str, "type": str}
    actuals_map: dict of { filename: { var_name: value } }
    """
    resource_task_id = f"audio_eval_{job_id}"
    if not model_ref_counter.acquire(model_id):
        audio_eval_jobs[job_id]["status"] = "error"
        audio_eval_jobs[job_id]["error"] = "Model siliniyor, işlem başlatılamadı."
        user_action_tracker.unregister(username, "audio_eval")
        release_model_quota(username)
        # Cleanup temp audio files before early return (finally block won't run)
        for af in audio_files:
            try:
                os.remove(af["path"])
            except OSError:
                pass
        if audio_files:
            try:
                shutil.rmtree(Path(audio_files[0]["path"]).parent, ignore_errors=True)
            except OSError:
                pass
        return
    try:
        # Reserve per-job processing overhead only; whisper VRAM is managed
        # independently by _get_whisper_model() / _unload_whisper_model()
        profile = resource_manager.get_profile("audio_pipeline")
        if not resource_manager.try_acquire(resource_task_id, "audio_eval",
                                             username=username,
                                             vram_mb=profile["vram_mb"],
                                             ram_mb=profile["ram_mb"]):
            audio_eval_jobs[job_id]["status"] = "error"
            audio_eval_jobs[job_id]["error"] = "Yeterli kaynak yok. Devam eden işlerin bitmesini bekleyin."
            return  # finally block handles unregister

        audio_eval_jobs[job_id]["status"] = "processing"
        audio_eval_jobs[job_id]["total"] = len(audio_files)
        audio_eval_jobs[job_id]["processed"] = 0

        row_results = []

        for i, af in enumerate(audio_files):
            fname = af["filename"]
            fpath = af["path"]
            row = {
                "filename": fname,
                "transcript": None,
                "predicted": {},
                "actuals": actuals_map.get(fname, {}),
                "summary_reasoning": None,
                "error": None,
            }

            try:
                # Step 1: Transcription
                print(f"  [AudioEval] Transcribing: {fname}")
                transcript = _transcribe_audio(fpath, language)
                row["transcript"] = transcript

                # Validate transcript — empty means silence/noise; skip LLM to avoid hallucinations
                if not transcript or not transcript.strip():
                    row["error"] = "Ses dosyasından metin çıkarılamadı (sessizlik veya gürültü)"
                    row_results.append(row)
                    audio_eval_jobs.update_fields(job_id, processed=i + 1)
                    continue

                # Step 2: Build prompt + call LLM
                print(f"  [AudioEval] Calling LLM for: {fname}")
                messages = _build_llm_prompt(prompt, transcript, schema)
                llm_result = _call_llm(messages)

                # Step 3: Extract predicted values
                for var in schema:
                    row["predicted"][var["name"]] = llm_result.get(var["name"])
                row["summary_reasoning"] = llm_result.get("summary_reasoning", "")

            except Exception as e:
                traceback.print_exc()
                row["error"] = _sanitize_error(e)

            row_results.append(row)
            audio_eval_jobs.update_fields(job_id, processed=i + 1)

        # Step 4: Compute metrics
        metrics = _compute_evaluation_metrics(row_results, schema)

        # Compute overall accuracy (classification) / avg MAE (regression)
        clf_scores = [m["accuracy"] for m in metrics.values() if m.get("type") == "classification" and "accuracy" in m]
        reg_scores = [m["mae"] for m in metrics.values() if m.get("type") == "regression" and "mae" in m]
        overall_accuracy = round(sum(clf_scores) / len(clf_scores), 2) if clf_scores else None
        overall_mae = round(sum(reg_scores) / len(reg_scores), 4) if reg_scores else None

        # Build a best_score that can appear on the dashboard
        if overall_accuracy is not None:
            best_score = overall_accuracy / 100.0  # normalize to 0-1
        elif overall_mae is not None:
            best_score = overall_mae  # lower is better, like timeseries
        else:
            best_score = 0

        # Step 5: Save as model metadata (appears on dashboard)
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "id": model_id,
            "name": model_name,
            "task_type": "call_analysis",
            "created_at": datetime.now().isoformat(),
            "target_column": ", ".join(v["name"] for v in schema),
            "num_rows": len(audio_files),
            "num_columns": len(schema),
            "best_score": best_score,
            "best_model": "LLM-as-a-Judge",
            "owner": username,
            "visibility": visibility,
            "endorsed": False,
            "endorsed_by": None,
            "view_count": 0,
            "total_predictions": len(audio_files),
            "call_analysis": {
                "schema": schema,
                "prompt": prompt,
                "language": language,
                "metrics": metrics,
                "overall_accuracy": overall_accuracy,
                "overall_mae": overall_mae,
                "row_results": row_results,
                "whisper_model": WHISPER_MODEL_REPO,
                "llm_endpoint": LLAMA_CPP_URL,
            },
            "submodels": [{
                "name": "LLM-as-a-Judge",
                "score": best_score,
                "inference_time": -1,
                "model_type": "llm_judge",
                "cpu_category": "N/A",
                "sql_support": {"easy_sql": False, "sql_method": "none", "label": "N/A"},
                "airflow_support": {"supported": False, "label": "N/A"},
            }],
        }

        save_model_meta(model_id, meta)
        add_activity("audio_evaluated", model_id, model_name,
                     f"{len(audio_files)} adet çağrı analiz etti.",
                     username=username)

        # Check if ALL files failed
        failed_count = sum(1 for r in row_results if r.get("error"))
        if failed_count == len(row_results) and len(row_results) > 0:
            audio_eval_jobs[job_id]["status"] = "error"
            audio_eval_jobs[job_id]["error"] = "Tüm ses dosyaları işlenemedi. GPU veya sunucu sorunu olabilir."
        else:
            audio_eval_jobs[job_id]["status"] = "done"
            audio_eval_jobs[job_id]["model_id"] = model_id

    except Exception as e:
        traceback.print_exc()
        audio_eval_jobs[job_id]["status"] = "error"
        audio_eval_jobs[job_id]["error"] = _sanitize_error(e)

    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        if not audio_eval_jobs.get(job_id, {}).get("_stale_released"):
            model_ref_counter.release(model_id)
            resource_manager.release(resource_task_id)
            user_action_tracker.unregister(username, "audio_eval")
            release_model_quota(username)
        # Cleanup temp audio files and directory
        temp_dir = None
        for af in audio_files:
            try:
                fpath = Path(af["path"])
                if temp_dir is None:
                    temp_dir = fpath.parent
                os.remove(af["path"])
            except OSError:
                pass
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass


def audio_predict_pipeline(job_id: str, model_id: str,
                           audio_files: list, schema: list, prompt: str,
                           language: str, username: str = ""):
    """
    Run prediction on new audio files using a saved call_analysis model's
    frozen prompt and schema.  Results are stored on the job dict for CSV download.
    """
    resource_task_id = f"audio_predict_{job_id}"
    if not model_ref_counter.acquire(model_id):
        audio_predict_jobs[job_id]["status"] = "error"
        audio_predict_jobs[job_id]["error"] = "Model siliniyor, işlem başlatılamadı."
        user_action_tracker.unregister(username, "audio_predict")
        # Cleanup temp audio files before early return (finally block won't run)
        for af in audio_files:
            try:
                os.remove(af["path"])
            except OSError:
                pass
        if audio_files:
            try:
                shutil.rmtree(Path(audio_files[0]["path"]).parent, ignore_errors=True)
            except OSError:
                pass
        return
    try:
        profile = resource_manager.get_profile("audio_pipeline")
        if not resource_manager.try_acquire(resource_task_id, "audio_predict",
                                             username=username,
                                             vram_mb=profile["vram_mb"],
                                             ram_mb=profile["ram_mb"]):
            audio_predict_jobs[job_id]["status"] = "error"
            audio_predict_jobs[job_id]["error"] = "Yeterli kaynak yok. Devam eden işlerin bitmesini bekleyin."
            return  # finally block handles unregister

        audio_predict_jobs[job_id]["status"] = "processing"
        audio_predict_jobs[job_id]["total"] = len(audio_files)
        audio_predict_jobs[job_id]["processed"] = 0

        row_results = []

        for i, af in enumerate(audio_files):
            fname = af["filename"]
            fpath = af["path"]
            row = {
                "filename": fname,
                "transcript": None,
                "predicted": {},
                "summary_reasoning": None,
                "error": None,
            }

            try:
                # Step 1: Transcription
                print(f"  [AudioPredict] Transcribing: {fname}")
                transcript = _transcribe_audio(fpath, language)
                row["transcript"] = transcript

                # Validate transcript — empty means silence/noise; skip LLM to avoid hallucinations
                if not transcript or not transcript.strip():
                    row["error"] = "Ses dosyasından metin çıkarılamadı (sessizlik veya gürültü)"
                    row_results.append(row)
                    audio_predict_jobs.update_fields(job_id, processed=i + 1)
                    continue

                # Step 2: Build prompt + call LLM
                print(f"  [AudioPredict] Calling LLM for: {fname}")
                messages = _build_llm_prompt(prompt, transcript, schema)
                llm_result = _call_llm(messages)

                # Step 3: Extract predicted values
                for var in schema:
                    row["predicted"][var["name"]] = llm_result.get(var["name"])
                row["summary_reasoning"] = llm_result.get("summary_reasoning", "")

            except Exception as e:
                traceback.print_exc()
                row["error"] = _sanitize_error(e)

            row_results.append(row)
            audio_predict_jobs.update_fields(job_id, processed=i + 1)

        # Build CSV content
        csv_buffer = io.StringIO()
        var_names = [v["name"] for v in schema]
        fieldnames = ["filename"] + var_names + ["llm_explanation"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for r in row_results:
            csv_row = {"filename": r["filename"]}
            for vn in var_names:
                csv_row[vn] = r["predicted"].get(vn, "")
            csv_row["llm_explanation"] = r.get("summary_reasoning") or r.get("error") or ""
            writer.writerow(csv_row)

        audio_predict_jobs[job_id]["csv_content"] = csv_buffer.getvalue()
        audio_predict_jobs[job_id]["row_results"] = row_results
        # Check if ALL files failed
        failed_count = sum(1 for r in row_results if r.get("error"))
        if failed_count == len(row_results) and len(row_results) > 0:
            audio_predict_jobs[job_id]["status"] = "error"
            audio_predict_jobs[job_id]["error"] = "Tüm ses dosyaları işlenemedi."
        else:
            audio_predict_jobs[job_id]["status"] = "done"

        # Update model prediction count atomically
        updated_meta = increment_model_meta_counter(model_id, "total_predictions", len(audio_files))
        model_display_name = updated_meta.get("name", model_id) if updated_meta else model_id

        add_activity("audio_predicted", model_id, model_display_name,
                     f"{len(audio_files)} ses dosyası ile tahmin yapıldı",
                     username=username)

    except Exception as e:
        traceback.print_exc()
        audio_predict_jobs[job_id]["status"] = "error"
        audio_predict_jobs[job_id]["error"] = _sanitize_error(e)

    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        if not audio_predict_jobs.get(job_id, {}).get("_stale_released"):
            model_ref_counter.release(model_id)
            resource_manager.release(resource_task_id)
            user_action_tracker.unregister(username, "audio_predict")
        # Cleanup temp audio files and directory
        temp_dir = None
        for af in audio_files:
            try:
                fpath = Path(af["path"])
                if temp_dir is None:
                    temp_dir = fpath.parent
                os.remove(af["path"])
            except OSError:
                pass
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass


def _sanitize_error(e) -> str:
    """Module-level error sanitizer. Strips file paths and truncates."""
    msg = str(e)
    # Windows paths: C:\..., c:\..., \\server\share\...
    msg = re.sub(r'[A-Za-z]:[\\\/][^\s"\']*', '[path]', msg)
    msg = re.sub(r'\\\\[^\s"\']+', '[path]', msg)
    # Unix paths: home directories, lib paths, and sensitive directories
    msg = re.sub(r'/(?:home|root|Users|opt|usr|var|tmp)/[^\s"\']*', '[path]', msg)
    msg = re.sub(r'/[^\s"\']*(/data/|/models/|/temp/|/site-packages/)[^\s"\']*', '[path]', msg)
    if len(msg) > 300:
        msg = msg[:300] + '…'
    return msg


# ══════════════════════════════════════════════════════════════════
#  HTTP REQUEST HANDLER
# ══════════════════════════════════════════════════════════════════

class PredictionAPIHandler(http.server.SimpleHTTPRequestHandler):
    timeout = 60  # Socket timeout — prevents Slowloris DoS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        if '/api/' in str(args[0]) if args else False:
            log.info(f"[API] {args[0]}" if args else "")

    def _safe_send_error(self, status: int, message: str):
        """Send error response, handling broken pipes gracefully."""
        try:
            self.send_json({"error": message}, status)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected — nothing we can do

    def _check_content_length(self, max_bytes: int = None) -> int:
        """Check Content-Length header and enforce size limit.
        Returns content length or -1 if rejected."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            self._safe_send_error(400, "Geçersiz Content-Length başlığı")
            return -1
        if content_length < 0:
            self._safe_send_error(400, "Geçersiz Content-Length başlığı")
            return -1
        limit = max_bytes or MAX_UPLOAD_SIZE_BYTES
        if content_length > limit:
            limit_mb = limit / (1024 * 1024)
            self._safe_send_error(413,
                f"Dosya boyutu çok büyük. Maksimum: {limit_mb:.0f}MB")
            return -1
        return content_length

    def _get_auth_token(self) -> str:
        """Extract auth token from cookie or Authorization header."""
        # Try cookie first
        cookie_header = self.headers.get("Cookie", "")
        for part in cookie_header.split(";"):
            part = part.strip()
            if part.startswith("session="):
                return part[len("session="):]
        # Try Authorization header
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None

    def _get_current_user(self) -> dict:
        """Get the current authenticated user."""
        token = self._get_auth_token()
        return get_session_user(token)

    def _require_auth(self) -> dict:
        """Require authentication, send 401 if not authenticated. Returns user or None."""
        user = self._get_current_user()
        if not user:
            self.send_json({"error": "Oturum açmanız gerekiyor"}, 401)
            return None
        if not _api_rate_limiter.check_normal(user["username"]):
            self.send_json({"error": "Çok fazla istek gönderdiniz. Lütfen biraz bekleyin."}, 429)
            return None
        return user

    def _get_client_ip(self):
        # Cloudflare Tunnel: always trust CF-Connecting-IP from localhost
        peer_ip = self.client_address[0]
        if peer_ip in ("127.0.0.1", "::1") or peer_ip in _TRUSTED_PROXY_IPS:
            cf_ip = self.headers.get("CF-Connecting-IP", "")
            if cf_ip:
                return cf_ip.strip()
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return peer_ip

    def _require_admin(self) -> dict:
        """Require admin role."""
        user = self._require_auth()
        if not user:
            return None
        if user["role"] not in ("admin", "master_admin"):
            self.send_json({"error": "Yönetici yetkisi gerekli"}, 403)
            return None
        return user

    def _check_model_access(self, meta: dict, user: dict) -> bool:
        """Check if user can access this model. Sends 403 if denied. Returns True if allowed."""
        if meta.get("visibility") == "private":
            if meta.get("owner") != user["username"] and user["role"] not in ("admin", "master_admin"):
                self.send_json({"error": "Bu model özel"}, 403)
                return False
        return True

    # Allowed CORS origins: comma-separated list.
    # Set via env: CORS_ORIGINS="https://myapp.com,https://admin.myapp.com"
    # Default: no CORS (same-origin only). Set CORS_ORIGINS="*" to allow all (not recommended).
    _cors_allowed = os.environ.get("CORS_ORIGINS", "").strip()

    def _cors_origin(self):
        """Return the requesting origin for CORS, or None."""
        origin = self.headers.get("Origin")
        if not origin:
            return None
        if not self._cors_allowed:
            return None  # No CORS configured — same-origin only
        if self._cors_allowed == "*":
            return origin  # Wildcard — reflect any origin (no credentials)
        allowed = [o.strip() for o in self._cors_allowed.split(",")]
        if origin in allowed:
            return origin
        return None  # Origin not allowed — no CORS header sent

    def send_json(self, data, status=200):
        try:
            body = safe_json_dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self._cors_allowed and self._cors_allowed != "*":
                    self.send_header("Access-Control-Allow-Credentials", "true")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """Sanitize filename for Content-Disposition to prevent response splitting."""
        return re.sub(r'[\r\n"\\\x00]', '_', filename)

    def send_file_download(self, content: str, filename: str, content_type: str = "text/plain"):
        try:
            filename = self._safe_filename(filename)
            encoded = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self._cors_allowed and self._cors_allowed != "*":
                    self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            if self._cors_allowed and self._cors_allowed != "*":
                self.send_header("Access-Control-Allow-Credentials", "true")
        self.end_headers()

    def _parse_json_body(self, body: bytes):
        """Parse JSON body and validate it's a dict. Returns (data, None) or (None, sent_error)."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self.send_json({"error": "Geçersiz JSON verisi"}, 400)
            return None, True
        if not isinstance(data, dict):
            self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
            return None, True
        return data, None

    @staticmethod
    def _safe_error_message(e: Exception) -> str:
        """Sanitize an exception message for user display."""
        return _sanitize_error(e)

    def do_GET(self):
        try:
            self._do_GET_inner()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected
        except Exception as e:
            log.error(f"Unhandled error in GET {self.path}: {e}", exc_info=True)
            self._safe_send_error(500, "Sunucu hatası oluştu. Lütfen tekrar deneyin.")

    def _do_GET_inner(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # ── Auth API ──
        if path == "/api/auth/me":
            return self.handle_auth_me()

        # ── Public API (no auth required) ──
        if path == "/api/presets":
            return self.send_json(PRESETS)

        # ── Authenticated API ──
        if path == "/api/dashboard":
            return self.handle_dashboard()

        elif path == "/api/server/status":
            return self.handle_server_status()

        elif path == "/api/training/active":
            return self.handle_active_training()

        elif path == "/api/models":
            return self.handle_list_models(params)

        elif path == "/api/users":
            return self.handle_list_users()

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)$", path)):
            return self.handle_get_model(m.group(1))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/export/([^/]+)/airflow$", path)):
            return self.handle_export_airflow(m.group(1), unquote(m.group(2)))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/export/([^/]+)/mssql$", path)):
            return self.handle_export_mssql(m.group(1), unquote(m.group(2)))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/columns$", path)):
            return self.handle_get_columns(m.group(1))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/explain/([^/]+)$", path)):
            return self.handle_explain(m.group(1), unquote(m.group(2)))

        elif (m := re.match(r"^/api/training/([a-f0-9-]+)/status$", path)):
            return self.handle_training_status(m.group(1))

        elif path == "/api/audio-evaluate/active":
            return self.handle_active_audio_eval()

        elif (m := re.match(r"^/api/audio-evaluate/([a-f0-9-]+)/status$", path)):
            return self.handle_audio_eval_status(m.group(1))

        elif (m := re.match(r"^/api/audio-predict/([a-f0-9-]+)/status$", path)):
            return self.handle_audio_predict_status(m.group(1))

        elif (m := re.match(r"^/api/audio-predict/([a-f0-9-]+)/download-csv$", path)):
            return self.handle_audio_predict_download(m.group(1))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/call-analysis/download-csv$", path)):
            return self.handle_call_analysis_download(m.group(1))

        elif path == "/api/admin/pending-users":
            return self.handle_pending_users()

        elif path == "/api/admin/logs":
            return self.handle_admin_logs(params)

        elif path == "/api/queue/status":
            return self.handle_queue_status()

        # ── Example datasets ──
        elif path == "/api/examples":
            return self.handle_examples_list()

        elif path == "/api/examples/audio-zip":
            return self.handle_examples_audio_zip()

        elif (m := re.match(r"^/api/examples/audio/([^/]+)$", path)):
            return self.handle_examples_audio(unquote(m.group(1)))

        elif (m := re.match(r"^/api/examples/csv/([^/]+)$", path)):
            return self.handle_examples_csv(unquote(m.group(1)))

        elif path.startswith("/api/"):
            return self.send_json({"error": "Bulunamadı"}, 404)

        # ── SPA: Serve index.html ──
        # Resolve the path and verify it stays within STATIC_DIR to prevent traversal
        try:
            file_path = (STATIC_DIR / path.lstrip("/")).resolve()
            if file_path.is_file() and str(file_path).startswith(str(STATIC_DIR.resolve()) + os.sep):
                return super().do_GET()
        except (OSError, ValueError):
            pass
        self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            self._do_POST_inner()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected
        except Exception as e:
            log.error(f"Unhandled error in POST {self.path}: {e}", exc_info=True)
            self._safe_send_error(500, "Sunucu hatası oluştu. Lütfen tekrar deneyin.")

    def _do_POST_inner(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ── Enforce request size limits ──
        content_length = self._check_content_length()
        if content_length < 0:
            return  # Already sent 413

        # ── Early auth check BEFORE reading large bodies to prevent RAM exhaustion ──
        # Auth routes have small bodies (<1KB); only read large bodies after auth verification
        _AUTH_ROUTES = ("/api/auth/login", "/api/auth/logout", "/api/auth/self-register")
        if path not in _AUTH_ROUTES and content_length > 1024 * 1024:  # >1MB needs auth first
            user = self._get_current_user()
            if not user:
                # Consume and discard the body to prevent HTTP/1.1 connection desync
                if content_length > 0:
                    try:
                        self.rfile.read(content_length)
                    except Exception:
                        pass
                self.send_json({"error": "Oturum açmanız gerekiyor"}, 401)
                return

        # Gate large body reads to prevent concurrent RAM exhaustion
        _large_upload = content_length > _UPLOAD_SEMAPHORE_THRESHOLD
        if _large_upload:
            if not _upload_semaphore.acquire(timeout=30):
                # Drain body to prevent HTTP/1.1 connection desync
                try:
                    self.rfile.read(content_length)
                except Exception:
                    pass
                self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyip tekrar deneyin."}, 503)
                return
        try:
            body = self.rfile.read(content_length) if content_length > 0 else b""
        except Exception:
            if _large_upload:
                _upload_semaphore.release()
            raise
        if _large_upload:
            _upload_semaphore.release()

        # ── Auth routes (no auth required) ──
        if path == "/api/auth/login":
            return self.handle_login(body)

        if path == "/api/auth/logout":
            return self.handle_logout()

        if path == "/api/auth/self-register":
            return self.handle_self_register(body)

        # ── Protected routes ──
        if path == "/api/auth/register":
            return self.handle_register(body)

        if path == "/api/auth/change-password":
            return self.handle_change_password(body)

        if path == "/api/upload-csv":
            return self.handle_upload_csv(body)

        elif path == "/api/train":
            return self.handle_train(body)

        elif path == "/api/audio-evaluate":
            return self.handle_audio_evaluate(body)

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/predict-audio$", path)):
            return self.handle_audio_predict(m.group(1), body)

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/predict/([^/]+)$", path)):
            return self.handle_predict(m.group(1), unquote(m.group(2)), body)

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/predict-batch/([^/]+)$", path)):
            return self.handle_predict_batch(m.group(1), unquote(m.group(2)), body)

        elif path == "/api/cost-estimate":
            return self.handle_cost_estimate(body)

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/delete$", path)):
            return self.handle_delete_model(m.group(1))

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/visibility$", path)):
            return self.handle_set_visibility(m.group(1), body)

        elif (m := re.match(r"^/api/models/([a-f0-9-]+)/endorse$", path)):
            return self.handle_endorse(m.group(1), body)

        elif path == "/api/users/update-role":
            return self.handle_update_role(body)

        elif path == "/api/admin/approve-user":
            return self.handle_approve_user(body)

        elif path == "/api/admin/reject-user":
            return self.handle_reject_user(body)

        elif path == "/api/admin/delete-user":
            return self.handle_delete_user(body)

        return self.send_json({"error": "Bulunamadı"}, 404)

    # ── Auth Handlers ───────────────────────────────────────

    def handle_login(self, body):
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return self.send_json({"error": "Kullanıcı adı ve şifre gerekli"}, 400)

        # Rate limiting: tiered blocking after repeated failed attempts
        # Key on both IP and username to prevent distributed brute-force
        client_ip = self._get_client_ip()
        rate_key = f"{client_ip}:{username.lower()}"
        # Also check per-IP limit (blocks all login attempts from one IP)
        ip_blocked, ip_retry = _login_limiter.is_blocked(f"ip:{client_ip}")
        if ip_blocked:
            minutes = math.ceil(ip_retry / 60)
            return self.send_json({"error": f"Çok fazla başarısız giriş denemesi. {minutes} dakika sonra tekrar deneyin."}, 429)
        blocked, retry_after = _login_limiter.is_blocked(rate_key)
        if blocked:
            minutes = math.ceil(retry_after / 60)
            return self.send_json({"error": f"Çok fazla başarısız giriş denemesi. {minutes} dakika sonra tekrar deneyin."}, 429)

        user = find_user(username)
        if not user:
            _login_limiter.record_attempt(rate_key)
            _login_limiter.record_attempt(f"ip:{client_ip}")
            return self.send_json({"error": "Kullanıcı adı veya şifre hatalı"}, 401)

        if user.get("status") == "pending":
            return self.send_json({"error": "Hesabınız henüz onaylanmadı. Lütfen yöneticinizle iletişime geçin."}, 403)
        if user.get("status") == "rejected":
            return self.send_json({"error": "Kayıt talebiniz reddedildi."}, 403)

        if not _verify_password(password, user["password_hash"], user["salt"]):
            _login_limiter.record_attempt(rate_key)
            _login_limiter.record_attempt(f"ip:{client_ip}")
            return self.send_json({"error": "Kullanıcı adı veya şifre hatalı"}, 401)

        # Successful login — reset per-user rate limiter only
        # (do NOT reset per-IP key: a legitimate login from a shared IP
        #  should not clear brute-force attempts against other usernames)
        _login_limiter.reset(rate_key)

        token = create_session(username)
        body_bytes = safe_json_dumps({
            "success": True,
            "user": {
                "username": user["username"],
                "display_name": user.get("display_name", user["username"]),
                "role": user["role"],
            }
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        cookie_flags = f"Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_TTL_SECONDS}"
        if os.environ.get("SECURE_COOKIES", "").lower() in ("true", "1", "yes"):
            cookie_flags += "; Secure"
        self.send_header("Set-Cookie", f"session={token}; {cookie_flags}")
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            if self._cors_allowed and self._cors_allowed != "*":
                self.send_header("Access-Control-Allow-Credentials", "true")
        self.end_headers()
        self.wfile.write(body_bytes)

    def handle_logout(self):
        token = self._get_auth_token()
        if token:
            destroy_session(token)
        body_bytes = b'{"success": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.send_header("Set-Cookie", "session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body_bytes)

    def handle_auth_me(self):
        user = self._get_current_user()
        if not user:
            return self.send_json({"authenticated": False}, 200)
        self.send_json({
            "authenticated": True,
            "user": {
                "username": user["username"],
                "display_name": user.get("display_name", user["username"]),
                "role": user["role"],
            }
        })

    def handle_register(self, body):
        admin = self._require_admin()
        if not admin:
            return

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        username = data.get("username", "").strip().lower()
        password = data.get("password", "")
        display_name = data.get("display_name", username).strip()
        role = data.get("role", "user")

        if not username or not password:
            return self.send_json({"error": "Kullanıcı adı ve şifre gerekli"}, 400)

        if len(username) < 3:
            return self.send_json({"error": "Kullanıcı adı en az 3 karakter olmalı"}, 400)

        if not re.match(r'^[a-z0-9_]+$', username):
            return self.send_json({"error": "Kullanıcı adı sadece küçük harf, rakam ve alt çizgi içerebilir"}, 400)

        pwd_error = _validate_password(password)
        if pwd_error:
            return self.send_json({"error": pwd_error}, 400)

        if role not in ("user", "admin"):
            return self.send_json({"error": "Geçersiz rol"}, 400)

        # Atomic check-and-create — single load to avoid TOCTOU
        with _file_locks["users"]:
            users = _safe_read_json(USERS_FILE, default=[])
            if any(u["username"].lower() == username.lower() for u in users):
                return self.send_json({"error": "Bu kullanıcı adı zaten kullanılıyor"}, 400)

            pwd_hash, salt = _hash_password(password)
            # SECURITY: display_name and email are immutable — no user-facing endpoint should modify them
            users.append({
                "username": username,
                "display_name": display_name,
                "password_hash": pwd_hash,
                "salt": salt,
                "role": role,
                "created_at": datetime.now().isoformat(),
                "created_by": admin["username"],
            })
            save_users(users)

        add_activity("user_created", details=f"Yeni kullanıcı: {display_name} ({role})", username=admin["username"])
        self.send_json({"success": True, "message": f"'{display_name}' kullanıcısı oluşturuldu"})

    def handle_self_register(self, body):
        """Public self-registration with admin approval."""
        # Rate limit by IP
        ip = self._get_client_ip()
        if _registration_limiter.is_blocked(ip):
            return self.send_json({"error": "Çok fazla kayıt denemesi. Lütfen daha sonra tekrar deneyin."}, 429)
        _registration_limiter.record_attempt(ip)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        username = data.get("username", "").strip().lower()
        password = data.get("password", "")
        display_name = data.get("display_name", "").strip()
        email = data.get("email", "").strip().lower()
        company = data.get("company", "").strip()

        if not username or not password or not display_name or not email:
            return self.send_json({"error": "Tüm alanları doldurun (kullanıcı adı, ad soyad, e-posta, şifre)"}, 400)
        if len(username) < 3:
            return self.send_json({"error": "Kullanıcı adı en az 3 karakter olmalı"}, 400)
        if not re.match(r'^[a-z0-9_]+$', username):
            return self.send_json({"error": "Kullanıcı adı sadece küçük harf, rakam ve alt çizgi içerebilir"}, 400)
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return self.send_json({"error": "Geçerli bir e-posta adresi girin"}, 400)
        if not email.endswith("@citizen.com"):
            return self.send_json({"error": "An error occured"}, 400)
        if len(display_name) > 100:
            return self.send_json({"error": "Ad soyad en fazla 100 karakter olabilir"}, 400)
        if len(company) > 100:
            return self.send_json({"error": "Firma adı en fazla 100 karakter olabilir"}, 400)

        pwd_error = _validate_password(password)
        if pwd_error:
            return self.send_json({"error": pwd_error}, 400)

        with _file_locks["users"]:
            users = _safe_read_json(USERS_FILE, default=[])
            if any(u["username"].lower() == username.lower() for u in users):
                return self.send_json({"error": "Bu kullanıcı adı zaten kullanılıyor"}, 400)
            # Check email uniqueness
            if any(u.get("email", "").lower() == email for u in users):
                return self.send_json({"error": "Bu e-posta adresi zaten kayıtlı"}, 400)

            pwd_hash, salt = _hash_password(password)
            # SECURITY: display_name and email are immutable — no user-facing endpoint should modify them
            users.append({
                "username": username,
                "display_name": display_name,
                "email": email,
                "company": company,
                "password_hash": pwd_hash,
                "salt": salt,
                "role": "user",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "created_by": "self",
            })
            save_users(users)

        add_activity("registration_request", details=f"{display_name} platforma katıldı.",
                     username=username, visibility="public")
        add_activity("registration_detail",
                     details=f"Yeni kayıt başvurusu: {display_name} ({email})",
                     username=username, visibility="admin_only")
        self.send_json({"success": True, "message": "Başvurunuz alındı. Yönetici onayından sonra giriş yapabilirsiniz."})

    def handle_change_password(self, body):
        user = self._require_auth()
        if not user:
            return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        old_password = data.get("old_password", "")
        new_password = data.get("new_password", "")

        if not _verify_password(old_password, user["password_hash"], user["salt"]):
            return self.send_json({"error": "Mevcut şifre hatalı"}, 400)

        pwd_error = _validate_password(new_password)
        if pwd_error:
            return self.send_json({"error": pwd_error}, 400)

        # Only modify password fields — display_name, email are immutable
        new_hash, new_salt = _hash_password(new_password)
        with _file_locks["users"]:
            users = load_users()
            for u in users:
                if u["username"] == user["username"]:
                    u["password_hash"] = new_hash
                    u["salt"] = new_salt
                    break
            save_users(users)
        # Invalidate all other sessions for this user (keep current session)
        current_token = self._get_auth_token()
        destroy_user_sessions(user["username"], except_token=current_token)
        self.send_json({"success": True, "message": "Şifre başarıyla değiştirildi"})

    def handle_list_users(self):
        admin = self._require_admin()
        if not admin:
            return
        users = load_users()
        safe_users = [{
            "username": u["username"],
            "display_name": u.get("display_name", u["username"]),
            "role": u["role"],
            "status": u.get("status", "active"),
            "email": u.get("email", ""),
            "company": u.get("company", ""),
            "created_at": u.get("created_at", ""),
            "created_by": u.get("created_by", ""),
        } for u in users]
        self.send_json({"users": safe_users})

    def handle_update_role(self, body):
        admin = self._require_admin()
        if not admin:
            return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        target_username = data.get("username", "").strip().lower()
        new_role = data.get("role", "")

        if new_role not in ("user", "admin"):
            return self.send_json({"error": "Geçersiz rol"}, 400)

        target_user = find_user(target_username)
        if not target_user:
            return self.send_json({"error": "Kullanıcı bulunamadı"}, 404)

        # Cannot change master_admin's role
        if target_user["role"] == "master_admin":
            return self.send_json({"error": "Ana yöneticinin rolü değiştirilemez"}, 403)

        # Cannot change own role
        if target_username == admin["username"].lower():
            return self.send_json({"error": "Kendi rolünüzü değiştiremezsiniz"}, 403)

        with _file_locks["users"]:
            users = load_users()
            found = False
            for u in users:
                if u["username"].lower() == target_username:
                    if u["role"] == "master_admin":
                        return self.send_json({"error": "Ana yöneticinin rolü değiştirilemez"}, 403)
                    u["role"] = new_role
                    found = True
                    break
            if not found:
                return self.send_json({"error": "Kullanıcı bulunamadı"}, 404)
            save_users(users)

        role_label = "Yönetici" if new_role == "admin" else "Kullanıcı"
        add_activity("role_changed",
                     details=f"{target_user.get('display_name', target_username)} → {role_label}",
                     username=admin["username"])
        self.send_json({"success": True, "message": f"Rol güncellendi: {role_label}"})

    def handle_approve_user(self, body):
        admin = self._require_admin()
        if not admin: return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
        username = data.get("username", "").strip().lower()
        if not username:
            return self.send_json({"error": "Kullanıcı adı gerekli"}, 400)
        with _file_locks["users"]:
            users = load_users()
            found = False
            for u in users:
                if u["username"] == username and u.get("status") == "pending":
                    u["status"] = "active"
                    found = True
                    break
            if not found:
                return self.send_json({"error": "Bekleyen kullanıcı bulunamadı"}, 404)
            save_users(users)
        add_activity("user_approved", details=f"Kullanıcı onaylandı: {username}", username=admin["username"])
        self.send_json({"success": True, "message": f"'{username}' kullanıcısı onaylandı"})

    def handle_reject_user(self, body):
        admin = self._require_admin()
        if not admin: return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
        username = data.get("username", "").strip().lower()
        if not username:
            return self.send_json({"error": "Kullanıcı adı gerekli"}, 400)
        with _file_locks["users"]:
            users = load_users()
            original_len = len(users)
            users = [u for u in users if not (u["username"] == username and u.get("status") == "pending")]
            if len(users) == original_len:
                return self.send_json({"error": "Bekleyen kullanıcı bulunamadı"}, 404)
            save_users(users)
        add_activity("user_rejected", details=f"Kayıt talebi reddedildi: {username}", username=admin["username"])
        self.send_json({"success": True, "message": f"'{username}' kullanıcısının başvurusu reddedildi"})

    def handle_delete_user(self, body):
        admin = self._require_admin()
        if not admin: return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
        username = data.get("username", "").strip().lower()
        if not username:
            return self.send_json({"error": "Kullanıcı adı gerekli"}, 400)
        if username == admin["username"]:
            return self.send_json({"error": "Kendi hesabınızı silemezsiniz"}, 400)
        with _file_locks["users"]:
            users = load_users()
            target = next((u for u in users if u["username"] == username), None)
            if not target:
                return self.send_json({"error": "Kullanıcı bulunamadı"}, 404)
            if target.get("role") == "master_admin":
                return self.send_json({"error": "Ana yönetici silinemez"}, 403)
            users = [u for u in users if u["username"] != username]
            save_users(users)
        add_activity("user_deleted", details=f"Kullanıcı silindi: {username}", username=admin["username"])
        self.send_json({"success": True, "message": f"'{username}' kullanıcısı silindi"})

    def handle_pending_users(self):
        admin = self._require_admin()
        if not admin: return
        users = load_users()
        pending = [{"username": u["username"], "display_name": u.get("display_name", ""),
                    "email": u.get("email", ""), "company": u.get("company", ""),
                    "created_at": u.get("created_at", "")}
                   for u in users if u.get("status") == "pending"]
        self.send_json({"pending": pending, "count": len(pending)})

    def handle_admin_logs(self, params):
        """Return recent server log lines. Admin only."""
        admin = self._require_admin()
        if not admin:
            return
        limit = min(int(params.get("limit", [2000])[0]), 2000)
        with _LOG_RING_LOCK:
            lines = list(_LOG_RING_BUFFER)
        # Return most recent `limit` lines
        lines = lines[-limit:]
        self.send_json({"lines": lines, "count": len(lines)})

    def handle_queue_status(self):
        user = self._require_auth()
        if not user: return
        self.send_json({
            "training_queue": _training_queue.queue_length(),
            "audio_eval_queue": _audio_eval_queue.queue_length(),
        })

    # ── Example datasets ────────────────────────────────────────

    @staticmethod
    def _nat_sort_key(name):
        """Sort key for natural ordering: 1_x, 2_x, 10_x instead of 1_x, 10_x, 2_x."""
        return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', name)]

    def handle_examples_list(self):
        """Return list of available example files from the example/ directory."""
        user = self._require_auth()
        if not user:
            return
        example_dir = BASE_DIR / "example"
        audio_files = []
        csv_files = []
        if example_dir.is_dir():
            for f in example_dir.iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext == ".mp3":
                    audio_files.append(f.name)
                elif ext == ".csv":
                    csv_files.append(f.name)
        audio_files.sort(key=self._nat_sort_key)
        csv_files.sort(key=self._nat_sort_key)
        self.send_json({"audio_files": audio_files, "csv_files": csv_files})

    def handle_examples_audio_zip(self):
        """Download all example MP3 files as a single zip."""
        import zipfile as _zipfile
        user = self._require_auth()
        if not user:
            return
        example_dir = BASE_DIR / "example"
        if not example_dir.is_dir():
            return self.send_json({"error": "Örnek dosya klasörü bulunamadı"}, 404)
        mp3_files = sorted([f for f in example_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp3"], key=lambda f: self._nat_sort_key(f.name))
        if not mp3_files:
            return self.send_json({"error": "Örnek ses dosyası bulunamadı"}, 404)
        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_STORED) as zf:
            for f in mp3_files:
                zf.write(f, f.name)
        zip_bytes = buf.getvalue()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(zip_bytes)))
            self.send_header("Content-Disposition", 'attachment; filename="ornek_ses_dosyalari.zip"')
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self._cors_allowed and self._cors_allowed != "*":
                    self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
            self.end_headers()
            self.wfile.write(zip_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_example_file(self, filename: str, allowed_ext: str, content_type: str, as_attachment: bool):
        """Serve a single file from the example/ directory with path-traversal protection."""
        user = self._require_auth()
        if not user:
            return
        if "/" in filename or "\\" in filename or ".." in filename:
            return self.send_json({"error": "Geçersiz dosya adı"}, 400)
        example_dir = (BASE_DIR / "example").resolve()
        file_path = (BASE_DIR / "example" / filename).resolve()
        if not str(file_path).startswith(str(example_dir) + os.sep):
            return self.send_json({"error": "Geçersiz dosya yolu"}, 400)
        if not file_path.is_file() or file_path.suffix.lower() != allowed_ext:
            return self.send_json({"error": "Dosya bulunamadı"}, 404)
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if as_attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{self._safe_filename(filename)}"')
                self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self._cors_allowed and self._cors_allowed != "*":
                    self.send_header("Access-Control-Allow-Credentials", "true")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def handle_examples_audio(self, filename: str):
        return self._serve_example_file(filename, ".mp3", "audio/mpeg", False)

    def handle_examples_csv(self, filename: str):
        return self._serve_example_file(filename, ".csv", "text/csv; charset=utf-8", True)

    # ── API Handlers ───────────────────────────────────────────

    def handle_server_status(self):
        """Return server resource status (admin only)."""
        user = self._require_auth()
        if not user:
            return
        # Anyone authenticated can see basic status; admins see full details
        status = resource_manager.get_status()
        if user["role"] in ("admin", "master_admin"):
            status["user_actions"] = user_action_tracker.get_active()
        else:
            status["user_actions"] = user_action_tracker.get_active(user["username"])
        self.send_json(status)

    def handle_dashboard(self):
        user = self._require_auth()
        if not user:
            return

        all_models = get_all_models()
        # Public + user's own
        visible_models = [m for m in all_models
                         if m.get("visibility", "public") == "public" or m.get("owner") == user["username"]]

        # Recently trained by others (public)
        recent_public = [m for m in all_models
                         if m.get("visibility", "public") == "public"][:10]

        # Popular models (by view_count)
        popular = sorted([m for m in all_models if m.get("visibility", "public") == "public"],
                        key=lambda x: x.get("view_count", 0), reverse=True)[:5]

        # Endorsed models
        endorsed = [m for m in all_models
                    if m.get("endorsed") and m.get("visibility", "public") == "public"]

        # My models
        my_models = [m for m in all_models if m.get("owner") == user["username"]]

        activity = get_filtered_activity(user)

        best_model = None
        if visible_models:
            # Separate tabular (higher=better) from timeseries (lower=better)
            non_audio = [m for m in visible_models if m.get("task_type") != "call_analysis"]
            tabular_models = [m for m in non_audio
                              if m.get("task_type") != "timeseries" and m.get("best_score", 0) > 0]
            ts_models = [m for m in non_audio
                         if m.get("task_type") == "timeseries" and m.get("best_score", 0) > 0]

            best = None
            if tabular_models:
                best = max(tabular_models, key=lambda m: m.get("best_score", 0))
            elif ts_models:
                best = min(ts_models, key=lambda m: m.get("best_score", float('inf')))
            elif non_audio:
                best = max(non_audio, key=lambda m: m.get("best_score", 0))

            if best:
                best_model = {
                    "id": best["id"], "name": best["name"],
                    "score": best["best_score"], "best_submodel": best.get("best_model"),
                    "task_type": best.get("task_type"), "owner": best.get("owner", ""),
                }

        self.send_json({
            "total_models": len(visible_models),
            "my_model_count": len(my_models),
            "total_users": len(load_users()),
            "recent_public": recent_public[:6],
            "popular": popular,
            "endorsed": endorsed,
            "my_models": my_models[:5],
            "best_model": best_model,
            "activity": [a for a in activity if a.get("action") in
                        ("trained", "audio_evaluated")][:10],
            "user": {
                "username": user["username"],
                "display_name": user.get("display_name", user["username"]),
                "role": user["role"],
            }
        })

    def handle_list_models(self, params):
        user = self._require_auth()
        if not user:
            return

        all_models = get_all_models()
        # Show public + own models
        models = [m for m in all_models
                  if m.get("visibility", "public") == "public" or m.get("owner") == user["username"]]

        search = params.get("search", [None])[0]
        if search:
            search = search.lower()
            models = [m for m in models if search in m["name"].lower()
                      or search in m.get("target_column", "").lower()
                      or search in m.get("task_type", "").lower()
                      or search in m.get("owner", "").lower()]

        task_filter = params.get("task_type", [None])[0]
        if task_filter:
            models = [m for m in models if m.get("task_type") == task_filter]

        self.send_json({"models": models, "total": len(models)})

    def handle_get_model(self, model_id):
        user = self._require_auth()
        if not user:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)

        # Check visibility
        if meta.get("visibility") == "private" and meta.get("owner") != user["username"]:
            if user["role"] not in ("admin", "master_admin"):
                return self.send_json({"error": "Bu model özel"}, 403)

        # Increment view count atomically
        updated = increment_model_meta_counter(model_id, "view_count", 1)
        if updated:
            meta = updated

        # Include cache status so the frontend can show load/predict button
        meta["model_loaded"] = model_cache.is_loaded(model_id)

        # Strip internal infrastructure details from response
        ca = meta.get("call_analysis", {})
        if isinstance(ca, dict):
            ca.pop("llm_endpoint", None)
        # Strip transcripts for non-owner/non-admin
        if meta.get("task_type") == "call_analysis" and meta.get("owner") != user["username"] and user["role"] not in ("admin", "master_admin"):
            if ca.get("row_results"):
                for row in ca["row_results"]:
                    row.pop("transcript", None)

        self.send_json(meta)

    def handle_get_columns(self, model_id):
        user = self._require_auth()
        if not user:
            return
        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        self.send_json({"columns": list(meta.get("column_types", {}).keys())})

    def handle_upload_csv(self, body):
        user = self._require_auth()
        if not user:
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            boundary = self._extract_boundary(content_type)
            if not boundary:
                return self.send_json({"error": "Geçersiz Content-Type başlığı"}, 400)
            parts = self._parse_multipart(body, boundary)
            csv_data = parts.get("file", {}).get("data")
            raw_filename = parts.get("file", {}).get("filename", "upload.csv")

            if not csv_data:
                return self.send_json({"error": "CSV dosyası bulunamadı"}, 400)

            # Sanitize filename to prevent path traversal
            filename = re.sub(r'[^\w\-. ]', '_', os.path.basename(raw_filename)).strip('. ')
            if not filename:
                filename = "upload"
            filename = filename[:200]  # limit length
            if not filename.lower().endswith('.csv'):
                filename = filename + '.csv'

            temp_id = str(uuid.uuid4())
            temp_dir = DATA_DIR / "temp" / temp_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            csv_path = temp_dir / filename

            try:
                with open(csv_path, "wb") as f:
                    f.write(csv_data)
            except OSError as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return self.send_json({"error": "Dosya yazılamadı. Sunucu diski dolu olabilir."}, 507)

            try:
                df_sample = _read_csv_with_fallback(csv_path, nrows=100)
                df_preview = df_sample.head(5).copy()
                columns = list(df_sample.columns)
                dtypes = {col: str(df_sample[col].dtype) for col in df_sample.columns}

                preview_records = []
                for _, row in df_preview.iterrows():
                    record = {}
                    for col in columns:
                        val = row[col]
                        try:
                            if pd.isna(val):
                                record[col] = None
                            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                                record[col] = None
                            else:
                                record[col] = val
                        except (TypeError, ValueError):
                            record[col] = val
                    preview_records.append(record)

                _, cleaning_report = clean_dataframe(df_sample.copy(), context="preview")

                # Count total rows for limit check (use csv.reader to handle quoted newlines)
                try:
                    import csv as _csv_mod
                    with open(str(csv_path), 'r', encoding='utf-8', errors='replace', newline='') as f:
                        reader = _csv_mod.reader(f)
                        total_rows = max(0, sum(1 for _ in reader) - 1)  # subtract header
                except Exception:
                    total_rows = 0
                if total_rows > MAX_BATCH_ROWS:
                    shutil.rmtree(str(temp_dir), ignore_errors=True)
                    return self.send_json({"error": f"CSV dosyası çok büyük ({total_rows:,} satır). Maksimum: {MAX_BATCH_ROWS:,} satır."}, 413)

                return self.send_json({
                    "temp_id": temp_id, "filename": filename,
                    "columns": columns, "dtypes": dtypes,
                    "preview": preview_records,
                    "cleaning_report": cleaning_report,
                })
            except Exception as e:
                traceback.print_exc()
                # Clean up temp directory on parse failure
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except OSError:
                    pass
                return self.send_json({"error": f"CSV ayrıştırma hatası: {self._safe_error_message(e)}"}, 400)
        else:
            return self.send_json({"error": "Multipart form data bekleniyor"}, 400)

    def handle_train(self, body):
        user = self._require_auth()
        if not user:
            return

        if not _api_rate_limiter.check_heavy(user["username"]):
            return self.send_json({"error": "Saatlik işlem limitinize ulaştınız. Lütfen daha sonra tekrar deneyin."}, 429)

        if not AUTOGLUON_AVAILABLE:
            return self.send_json({"error": "AutoGluon yüklü değil"}, 500)

        # Per-user model quota check (atomic to prevent TOCTOU bypass)
        if not check_and_reserve_model_quota(user["username"]):
            return self.send_json({"error": f"Analiz limitinize ulaştınız (maksimum {MAX_MODELS_PER_USER}). Eski analizleri silerek yer açabilirsiniz."}, 429)

        data, err = self._parse_json_body(body)
        if err:
            release_model_quota(user["username"])
            return

        temp_id = data.get("temp_id")
        target_col = data.get("target_column")
        task_type = data.get("task_type", "classification")
        preset = data.get("preset", "medium_quality")
        model_name = data.get("model_name", f"Model_{datetime.now().strftime('%Y%m%d_%H%M')}")
        if len(model_name) > 200:
            release_model_quota(user["username"])
            return self.send_json({"error": "Analiz adı en fazla 200 karakter olabilir"}, 400)
        visibility = data.get("visibility", "private")

        # Time series specific parameters
        timestamp_column = data.get("timestamp_column", None)
        item_id_column = data.get("item_id_column", None)
        try:
            prediction_length = int(data.get("prediction_length", 10))
            if prediction_length < 1:
                prediction_length = 10
            if prediction_length > MAX_PREDICTION_LENGTH:
                release_model_quota(user["username"])
                return self.send_json({
                    "error": f"Tahmin uzunluğu çok büyük. Maksimum: {MAX_PREDICTION_LENGTH}"
                }, 400)
        except (ValueError, TypeError):
            prediction_length = 10

        if task_type == "timeseries":
            if not AUTOGLUON_TS_AVAILABLE:
                release_model_quota(user["username"])
                return self.send_json({"error": "autogluon.timeseries yüklü değil"}, 500)
            if not timestamp_column:
                release_model_quota(user["username"])
                return self.send_json({"error": "Zaman serisi için timestamp_column gerekli"}, 400)

        if visibility not in ("public", "private"):
            visibility = "private"

        if not temp_id or not target_col:
            release_model_quota(user["username"])
            return self.send_json({"error": "temp_id veya target_column eksik"}, 400)

        # Sanitize temp_id to prevent path traversal
        temp_id = re.sub(r'[^a-f0-9\-]', '', temp_id)
        temp_dir = DATA_DIR / "temp" / temp_id
        if not temp_dir.exists():
            release_model_quota(user["username"])
            return self.send_json({"error": "Geçici dosya bulunamadı. Lütfen CSV'yi tekrar yükleyin."}, 400)
        csv_files = list(temp_dir.glob("*.csv"))
        if not csv_files:
            release_model_quota(user["username"])
            return self.send_json({"error": "CSV dosyası bulunamadı"}, 400)

        csv_path = str(csv_files[0])
        model_id = str(uuid.uuid4())
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        # Atomic copy: write to tmp then rename (crash-safe on network volumes)
        _tmp_csv = model_dir / ".training_data.csv.tmp"
        shutil.copy2(csv_path, _tmp_csv)
        durable_csv_path = str(model_dir / "training_data.csv")
        os.replace(str(_tmp_csv), durable_csv_path)

        # Clean up temp directory now that we have the durable copy
        _cleanup_temp_dir(temp_id)

        # ── Per-user concurrency check (atomic, AFTER all validation) ──
        job_id = str(uuid.uuid4())
        allowed, reason = user_action_tracker.try_register(user["username"], "training", job_id)
        if not allowed:
            shutil.rmtree(model_dir, ignore_errors=True)
            release_model_quota(user["username"])
            return self.send_json({"error": reason}, 429)

        training_jobs[job_id] = {"status": "queued", "model_id": model_id, "_username": user["username"], "_action_type": "training", "_resource_task_id": f"training_{job_id}", "_model_id": model_id}

        try:
            _training_queue.submit(
                user["username"], job_id,
                train_model,
                job_id, model_id, durable_csv_path, target_col, task_type, preset, model_name,
                user["username"], visibility,
                timestamp_column=timestamp_column,
                item_id_column=item_id_column,
                prediction_length=prediction_length,
            )
        except Exception as e:
            # Queue submission failed — unregister so user isn't permanently locked
            user_action_tracker.unregister(user["username"], "training")
            release_model_quota(user["username"])
            training_jobs[job_id]["status"] = "error"
            training_jobs[job_id]["error"] = "İş kuyruğa eklenemedi."
            shutil.rmtree(model_dir, ignore_errors=True)
            log.error(f"Training queue submit failed: {e}")
            return self.send_json({"error": "İş kuyruğa eklenemedi. Lütfen tekrar deneyin."}, 503)

        add_activity("started_training", model_id, model_name,
                     f"Eğitim başlatıldı: {preset}", username=user["username"])

        return self.send_json({"job_id": job_id, "model_id": model_id, "status": "queued"})

    def handle_active_training(self):
        """Check if current user has an active training job."""
        user = self._require_auth()
        if not user:
            return
        active = user_action_tracker.get_active(user["username"])
        training_job_id = active.get("training")
        if training_job_id:
            job = training_jobs.get(training_job_id)
            if job and job.get("status") not in ("done", "error"):
                return self.send_json({
                    "active": True,
                    "job_id": training_job_id,
                    "status": job.get("status"),
                    "model_id": job.get("model_id"),
                })
        self.send_json({"active": False})

    def handle_training_status(self, job_id):
        user = self._require_auth()
        if not user:
            return
        job = training_jobs.get(job_id)
        if not job:
            return self.send_json({"error": "İş bulunamadı"}, 404)
        if job.get("_username") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu işi görüntüleme yetkiniz yok"}, 403)
        response = {"status": job["status"], "model_id": job.get("model_id")}
        if job["status"] == "queued":
            response["position"] = _training_queue.get_position(job_id)
        if job["status"] == "done":
            meta = load_model_meta(job["model_id"])
            if meta:
                response["model"] = meta
        if job["status"] == "error":
            response["error"] = job.get("error", "Bilinmeyen hata")
        self.send_json(response)

    def handle_predict(self, model_id, submodel_name, body):
        user = self._require_auth()
        if not user:
            return

        if not AUTOGLUON_AVAILABLE:
            return self.send_json({"error": "AutoGluon yüklü değil"}, 500)

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        # Validate submodel exists
        if not any(sm["name"] == submodel_name for sm in meta.get("submodels", [])):
            return self.send_json({"error": f"Alt model bulunamadı: {submodel_name}"}, 404)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz JSON verisi"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        # ── Handle explicit load request from UI ──
        if data.get("_load_only"):
            if not model_ref_counter.acquire(model_id):
                return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)
            if not _prediction_semaphore.acquire(timeout=0):
                model_ref_counter.release(model_id)
                return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyin."}, 503)
            try:
                predictor = model_cache.load_model(model_id, meta)
                if predictor is None:
                    return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)
                return self.send_json({"loaded": True})
            except Exception as e:
                traceback.print_exc()
                return self.send_json({"error": f"Model yüklenirken hata: {self._safe_error_message(e)}"}, 500)
            finally:
                _prediction_semaphore.release()
                model_ref_counter.release(model_id)

        if not model_ref_counter.acquire(model_id):
            return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)

        # Solution 6: Limit concurrent predictions (non-blocking to avoid thread pile-up)
        if not _prediction_semaphore.acquire(timeout=0):
            model_ref_counter.release(model_id)
            return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyin."}, 503)

        # Solution 4: Use measured VRAM if available, else profile default
        _pred_resource_id = f"predict_{uuid.uuid4().hex[:8]}"
        _pred_resource_acquired = False
        try:
            _pred_vram = meta.get("measured_vram_peak_mb", resource_manager.PROFILES["prediction_model_load"]["vram_mb"])
            with _sentence_model_lock:
                _emb_entry = _sentence_model_cache.get(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
            if meta.get("text_columns") and _emb_entry:
                _pred_vram += _emb_entry.get("measured_vram_mb", 500)
            elif meta.get("text_columns"):
                _pred_vram += 500  # fallback for unmeasured embeddings

            # Solution 7: Pre-emptive eviction if VRAM tight
            if _pred_vram > 0:
                if not _ensure_vram_available(_pred_vram + 300):
                    log.warning(f"[VRAM] Pre-eviction insufficient: needed {_pred_vram + 300}MB")

            _pred_ram = _estimate_model_ram_mb(model_id, meta)
            if not resource_manager.try_acquire(_pred_resource_id, "prediction_model_load",
                                                vram_mb=_pred_vram,
                                                ram_mb=_pred_ram):
                return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyip tekrar deneyin."}, 503)
            _pred_resource_acquired = True
            # ── Time Series prediction path ──
            if meta.get("task_type") == "timeseries":
                if not AUTOGLUON_TS_AVAILABLE:
                    return self.send_json({"error": "autogluon.timeseries yüklü değil"}, 500)

                # Expect historical context as rows
                history_rows = data.get("history", [])
                if not history_rows:
                    return self.send_json({
                        "error": "Zaman serisi tahmini için 'history' (geçmiş veri satırları) gerekli"
                    }, 400)
                if len(history_rows) > MAX_BATCH_ROWS:
                    return self.send_json({
                        "error": f"Geçmiş veri çok büyük ({len(history_rows):,} satır). Maksimum: {MAX_BATCH_ROWS:,}"
                    }, 413)

                ts_col = meta.get("timestamp_column")
                id_col = meta.get("item_id_column", "__item_id")
                target_col = meta["target_column"]

                hist_df = pd.DataFrame(history_rows)
                if ts_col not in hist_df.columns:
                    return self.send_json({"error": f"Zaman damgası sütunu '{ts_col}' gönderilen veride bulunamadı"}, 400)
                if target_col not in hist_df.columns:
                    return self.send_json({"error": f"Hedef sütun '{target_col}' gönderilen veride bulunamadı"}, 400)

                # Clean prediction input (drop empty rows, fill gaps, handle infinities)
                try:
                    hist_df, _ = clean_dataframe(hist_df, context="timeseries",
                                                 timestamp_column=ts_col,
                                                 item_id_column=id_col if id_col in hist_df.columns else None)
                except ValueError as e:
                    return self.send_json({"error": self._safe_error_message(e)}, 400)

                try:
                    hist_df[ts_col] = pd.to_datetime(hist_df[ts_col])
                except (ValueError, TypeError) as e:
                    return self.send_json({"error": f"Zaman damgası ayrıştırılamadı: {self._safe_error_message(e)}"}, 400)

                if id_col not in hist_df.columns:
                    hist_df[id_col] = "series_0"
                else:
                    hist_df[id_col] = hist_df[id_col].astype(str)

                hist_df = hist_df.sort_values(by=[id_col, ts_col]).reset_index(drop=True)

                ts_df = TimeSeriesDataFrame.from_data_frame(
                    hist_df, id_column=id_col, timestamp_column=ts_col,
                )

                ts_predictor = model_cache.load_model(model_id, meta)
                if ts_predictor is None:
                    return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)
                predictions = ts_predictor.predict(ts_df, model=submodel_name)

                # Convert forecast to a list of dicts
                forecast_records = []
                pred_df = predictions.reset_index()
                if len(pred_df) == 0:
                    return self.send_json({"error": "Tahmin sonucu boş döndü"}, 400)
                for _, row in pred_df.iterrows():
                    rec = {}
                    for col in pred_df.columns:
                        val = row[col]
                        if hasattr(val, 'isoformat'):
                            rec[col] = val.isoformat()
                        else:
                            rec[col] = sanitize_value(val)
                    forecast_records.append(rec)

                increment_model_meta_counter(model_id, "total_predictions", 1)
                add_activity("predicted", model_id, meta["name"],
                            f"{submodel_name} ile zaman serisi tahmini", username=user["username"])

                return self.send_json({
                    "task_type": "timeseries",
                    "prediction_length": meta.get("prediction_length", 10),
                    "forecast": forecast_records,
                })

            # ── Unified prediction path (with auto text embedding) ──
            features = data.get("features", {})
            column_types = meta.get("column_types", {})
            # Validate required columns are present
            required_cols = set(meta.get("feature_columns", []))
            provided_cols = set(features.keys())
            missing = required_cols - provided_cols
            # Reject if >20% of columns are missing (strict but allows minor omissions)
            if missing and len(missing) > max(1, len(required_cols) * 0.2):
                return self.send_json({
                    "error": f"Eksik özellik sütunları: {', '.join(sorted(list(missing)[:10]))}"
                }, 400)
            cleaned_features = clean_prediction_input(features, column_types)
            predictor = model_cache.load_model(model_id, meta)
            if predictor is None:
                return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)
            input_df = pd.DataFrame([cleaned_features])

            numeric_cols = input_df.select_dtypes(include=[np.number]).columns
            input_df[numeric_cols] = input_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

            # Auto-detect text columns that need embedding
            _pred_text_columns = meta.get("text_columns", [])
            if not _pred_text_columns:
                pipeline_config = _load_text_pipeline_config(model_id)
                if pipeline_config:
                    _pred_text_columns = pipeline_config.get("text_columns", [])

            _pred_text_cols_present = [c for c in _pred_text_columns if c in input_df.columns]
            _pred_text_cols_missing = [c for c in _pred_text_columns if c not in input_df.columns]
            if _pred_text_cols_missing:
                return self.send_json({
                    "error": f"Eksik metin sütunları: {', '.join(_pred_text_cols_missing)}"
                }, 400)
            if _pred_text_cols_present:
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    return self.send_json({"error": "sentence-transformers yüklü değil"}, 500)
                embedding_model_name = meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                # Verify embedding dimension matches training
                _emb_dim_expected = meta.get("embedding_dim")
                if _emb_dim_expected:
                    _emb_model = _get_sentence_model(embedding_model_name)
                    if _emb_model.get_sentence_embedding_dimension() != _emb_dim_expected:
                        return self.send_json({"error": "Embedding model boyutu eğitim ile uyuşmuyor"}, 400)
                for col in _pred_text_cols_present:
                    input_df[col] = input_df[col].fillna("")
                input_df = _embed_text_columns(input_df, _pred_text_cols_present,
                                                embedding_model_name, show_progress=False)

            prediction = predictor.predict(input_df, model=submodel_name)
            if len(prediction) == 0:
                return self.send_json({"error": "Tahmin sonucu boş döndü"}, 400)
            result = {"prediction": sanitize_value(prediction.iloc[0])}

            if meta.get("problem_type") in ("binary", "multiclass") or meta.get("task_type") == "classification":
                try:
                    proba = predictor.predict_proba(input_df, model=submodel_name)
                    result["probabilities"] = {}
                    for k, v in proba.iloc[0].items():
                        try:
                            val = float(v)
                            if math.isnan(val) or math.isinf(val):
                                val = 0.0
                        except (ValueError, TypeError):
                            val = 0.0
                        result["probabilities"][str(k)] = round(val, 4)
                except Exception:
                    pass

            if hasattr(result["prediction"], 'item'):
                result["prediction"] = result["prediction"].item()

            # Solution 4: Measure peak VRAM on first prediction for this model
            if not meta.get("measured_vram_peak_mb"):
                try:
                    import torch as _torch_vram
                    _peak = _torch_vram.cuda.max_memory_allocated(0) // (1024 * 1024)
                    _torch_vram.cuda.reset_peak_memory_stats(0)
                    if _peak > 0:
                        update_model_meta_fields(model_id, measured_vram_peak_mb=int(_peak))
                        log.info(f"[Predict] Model {model_id[:8]} measured VRAM peak: {_peak}MB")
                except Exception:
                    pass

            increment_model_meta_counter(model_id, "total_predictions", 1)
            add_activity("predicted", model_id, meta["name"],
                        f"{submodel_name} ile tekli tahmin", username=user["username"])
            self.send_json(result)

        except (MemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if isinstance(e, MemoryError) or "out of memory" in err_str:
                model_cache.evict(model_id)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.send_json({"error": "Bellek yetersiz. Model bellekten kaldırıldı. Lütfen tekrar deneyin."}, 503)
            elif "cuda" in err_str or "device" in err_str:
                model_cache.evict(model_id)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.send_json({"error": "GPU hatası oluştu. Lütfen tekrar deneyin."}, 503)
            else:
                traceback.print_exc()
                self.send_json({"error": self._safe_error_message(e)}, 500)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": self._safe_error_message(e)}, 500)
        finally:
            if _pred_resource_acquired:
                resource_manager.release(_pred_resource_id)
            _prediction_semaphore.release()
            model_ref_counter.release(model_id)

    def handle_predict_batch(self, model_id, submodel_name, body):
        user = self._require_auth()
        if not user:
            return

        if not _api_rate_limiter.check_heavy(user["username"]):
            return self.send_json({"error": "Saatlik işlem limitinize ulaştınız. Lütfen daha sonra tekrar deneyin."}, 429)

        if not AUTOGLUON_AVAILABLE:
            return self.send_json({"error": "AutoGluon yüklü değil"}, 500)

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        # Validate submodel exists
        if not any(sm["name"] == submodel_name for sm in meta.get("submodels", [])):
            return self.send_json({"error": f"Alt model bulunamadı: {submodel_name}"}, 404)

        content_type = self.headers.get("Content-Type", "")

        if not model_ref_counter.acquire(model_id):
            return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)

        # Solution 6: Limit concurrent predictions (non-blocking to avoid thread pile-up)
        if not _prediction_semaphore.acquire(timeout=0):
            model_ref_counter.release(model_id)
            return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyin."}, 503)

        # Solution 4: Use measured VRAM if available
        _bpred_resource_id = f"batch_predict_{uuid.uuid4().hex[:8]}"
        _bp_resource_acquired = False
        try:
            _bp_vram = meta.get("measured_vram_peak_mb", resource_manager.PROFILES["prediction_model_load"]["vram_mb"])
            with _sentence_model_lock:
                _emb_entry_b = _sentence_model_cache.get(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
            if meta.get("text_columns") and _emb_entry_b:
                _bp_vram += _emb_entry_b.get("measured_vram_mb", 500)
            elif meta.get("text_columns"):
                _bp_vram += 500

            # Solution 7: Pre-emptive eviction
            if _bp_vram > 0:
                if not _ensure_vram_available(_bp_vram + 300):
                    log.warning(f"[VRAM] Pre-eviction insufficient: needed {_bp_vram + 300}MB")

            _bp_ram = _estimate_model_ram_mb(model_id, meta)
            if not resource_manager.try_acquire(_bpred_resource_id, "prediction_model_load",
                                                vram_mb=_bp_vram,
                                                ram_mb=_bp_ram):
                return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyip tekrar deneyin."}, 503)
            _bp_resource_acquired = True
            if "multipart/form-data" in content_type:
                boundary = self._extract_boundary(content_type)
                if not boundary:
                    return self.send_json({"error": "Geçersiz Content-Type başlığı"}, 400)
                parts = self._parse_multipart(body, boundary)
                csv_data = parts.get("file", {}).get("data")
                if not csv_data:
                    return self.send_json({"error": "CSV dosyası bulunamadı"}, 400)
                try:
                    input_df = pd.read_csv(io.BytesIO(csv_data), encoding='utf-8-sig')
                except UnicodeDecodeError:
                    input_df = None
                    for enc in ['windows-1254', 'latin-1', 'windows-1252']:
                        try:
                            input_df = pd.read_csv(io.BytesIO(csv_data), encoding=enc)
                            break
                        except (UnicodeDecodeError, Exception):
                            continue
                    if input_df is None:
                        input_df = pd.read_csv(io.BytesIO(csv_data), encoding='utf-8', errors='replace')
            else:
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    return self.send_json({"error": "Geçersiz JSON verisi"}, 400)
                if not isinstance(data, dict):
                    return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
                input_df = pd.DataFrame(data.get("rows", []))

            if len(input_df) > MAX_BATCH_ROWS:
                return self.send_json({
                    "error": f"Toplu tahmin satır limiti aşıldı. Maksimum: {MAX_BATCH_ROWS:,} satır, "
                             f"gönderilen: {len(input_df):,} satır."
                }, 413)

            # ── Time Series batch prediction path ──
            if meta.get("task_type") == "timeseries":
                if not AUTOGLUON_TS_AVAILABLE:
                    return self.send_json({"error": "autogluon.timeseries yüklü değil"}, 500)

                ts_col = meta.get("timestamp_column")
                id_col = meta.get("item_id_column", "__item_id")
                target_col = meta.get("target_column", "")

                if target_col and target_col not in input_df.columns:
                    return self.send_json({"error": f"Hedef sütun '{target_col}' gönderilen veride bulunamadı"}, 400)

                input_df, batch_report = clean_dataframe(
                    input_df, context="timeseries", timestamp_column=ts_col,
                    item_id_column=id_col if id_col in input_df.columns else None)

                try:
                    input_df[ts_col] = pd.to_datetime(input_df[ts_col])
                except (ValueError, TypeError, KeyError) as e:
                    return self.send_json({"error": f"Zaman damgası sütunu ayrıştırılamadı: {self._safe_error_message(e)}"}, 400)
                if id_col not in input_df.columns:
                    input_df[id_col] = "series_0"
                else:
                    input_df[id_col] = input_df[id_col].astype(str)

                input_df = input_df.sort_values(by=[id_col, ts_col]).reset_index(drop=True)

                ts_df = TimeSeriesDataFrame.from_data_frame(
                    input_df, id_column=id_col, timestamp_column=ts_col,
                )

                ts_predictor = model_cache.load_model(model_id, meta)
                if ts_predictor is None:
                    return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)
                predictions = ts_predictor.predict(ts_df, model=submodel_name)

                output_df = predictions.reset_index()
                if len(output_df) == 0:
                    return self.send_json({"error": "Zaman serisi toplu tahmin sonucu boş döndü"}, 400)
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()

                increment_model_meta_counter(model_id, "total_predictions", len(output_df))
                add_activity("batch_predicted", model_id, meta["name"],
                            f"Zaman serisi toplu tahmin ({submodel_name})", username=user["username"])

                return self.send_file_download(
                    csv_string, f"tahminler_ts_{model_id[:8]}_{submodel_name}.csv", "text/csv")

            # ── Tabular batch prediction path (with auto text embedding) ──
            input_df, batch_report = clean_dataframe(input_df, context="prediction")
            numeric_cols = input_df.select_dtypes(include=[np.number]).columns
            input_df[numeric_cols] = input_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

            str_cols = input_df.select_dtypes(include=['object']).columns
            for col in str_cols:
                input_df[col] = input_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

            if len(input_df) == 0:
                return self.send_json({"error": "Temizlik sonrası geçerli satır kalmadı."}, 400)

            # Auto-detect text columns that need embedding
            _batch_text_columns = meta.get("text_columns", [])
            if not _batch_text_columns:
                pipeline_config = _load_text_pipeline_config(model_id)
                if pipeline_config:
                    _batch_text_columns = pipeline_config.get("text_columns", [])

            # Keep human-readable copy before embedding (for output CSV)
            original_input_df = input_df.copy()
            _batch_text_cols_present = [c for c in _batch_text_columns if c in input_df.columns]

            if _batch_text_cols_present:
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    return self.send_json({"error": "sentence-transformers yüklü değil"}, 500)
                embedding_model_name = meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                # Fill NaN in text columns before embedding
                for col in _batch_text_cols_present:
                    input_df[col] = input_df[col].fillna("")
                input_df = _embed_text_columns(input_df, _batch_text_cols_present, embedding_model_name)
            else:
                # Standard tabular: replace empty strings with NaN for non-text columns
                for col in str_cols:
                    input_df[col] = input_df[col].replace('', np.nan)

            predictor = model_cache.load_model(model_id, meta)
            if predictor is None:
                return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)

            predictions = predictor.predict(input_df, model=submodel_name)
            # Use original (human-readable) DataFrame if we embedded text columns
            output_df = original_input_df.copy() if _batch_text_cols_present else input_df.copy()
            target_col = meta["target_column"]
            # Use .values for positional assignment to avoid pandas index misalignment
            output_df[f"{target_col}_predicted"] = predictions.values

            if meta.get("problem_type") in ("binary", "multiclass") or meta.get("task_type") == "classification":
                try:
                    proba = predictor.predict_proba(input_df, model=submodel_name)
                    for col in proba.columns:
                        prob_col = proba[col].fillna(0.0).replace([np.inf, -np.inf], 0.0).round(4)
                        output_df[f"{target_col}_proba_{col}"] = prob_col.values
                except Exception:
                    pass

            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()

            row_count = len(original_input_df) if _batch_text_cols_present else len(input_df)
            increment_model_meta_counter(model_id, "total_predictions", row_count)
            add_activity("batch_predicted", model_id, meta["name"],
                        f"{row_count} satırlık toplu tahmin ({submodel_name})", username=user["username"])

            self.send_file_download(csv_string, f"tahminler_{model_id[:8]}_{submodel_name}.csv", "text/csv")

        except (MemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if isinstance(e, MemoryError) or "out of memory" in err_str:
                model_cache.evict(model_id)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.send_json({"error": "Bellek yetersiz. Model bellekten kaldırıldı. Lütfen tekrar deneyin."}, 503)
            elif "cuda" in err_str or "device" in err_str:
                model_cache.evict(model_id)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.send_json({"error": "GPU hatası oluştu. Lütfen tekrar deneyin."}, 503)
            else:
                traceback.print_exc()
                self.send_json({"error": self._safe_error_message(e)}, 500)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": self._safe_error_message(e)}, 500)
        finally:
            if _bp_resource_acquired:
                resource_manager.release(_bpred_resource_id)
            _prediction_semaphore.release()
            model_ref_counter.release(model_id)

    def handle_export_airflow(self, model_id, submodel_name):
        user = self._require_auth()
        if not user:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        # Validate submodel exists
        if not any(sm["name"] == submodel_name for sm in meta.get("submodels", [])):
            return self.send_json({"error": f"Alt model bulunamadı: {submodel_name}"}, 404)

        if not model_ref_counter.acquire(model_id):
            return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)
        try:
            return self._do_export_airflow(model_id, submodel_name, meta, user)
        except Exception as e:
            traceback.print_exc()
            self._safe_send_error(500, f"Dışa aktarma hatası: {self._safe_error_message(e)}")
        finally:
            model_ref_counter.release(model_id)

    def _do_export_airflow(self, model_id, submodel_name, meta, user):
        import zipfile

        # Check model directory size before creating zip in memory
        MAX_EXPORT_SIZE_MB = int(os.environ.get("MAX_EXPORT_SIZE_MB", "2048"))
        ag_model_path = MODELS_DIR / model_id / "agmodel"
        if ag_model_path.exists():
            total_bytes = sum(f.stat().st_size for f in ag_model_path.rglob('*') if f.is_file())
            if total_bytes > MAX_EXPORT_SIZE_MB * 1024 * 1024:
                return self.send_json({
                    "error": f"Model dizini çok büyük ({total_bytes // (1024*1024)}MB). "
                             f"Maksimum dışa aktarma boyutu: {MAX_EXPORT_SIZE_MB}MB. "
                             f"Modeli sunucudan doğrudan kopyalayın."
                }, 413)

        # ── Time Series Airflow DAG ──
        if meta.get("task_type") == "timeseries":
            dag_code = generate_timeseries_airflow_dag(
                model_id=model_id, model_name=meta["name"],
                submodel_name=submodel_name, target_col=meta["target_column"],
                timestamp_column=meta.get("timestamp_column", "timestamp"),
                item_id_column=meta.get("item_id_column", "__item_id"),
                prediction_length=meta.get("prediction_length", 10),
            )

            safe_submodel = re.sub(r'[^a-zA-Z0-9._-]', '_', submodel_name).lower()

            readme = f"""# Airflow DAG Paketi — Zaman Serisi
## Model: {meta['name']}
## Alt Model: {submodel_name}
## Hedef: {meta['target_column']} (Zaman Serisi Tahmini)
## Tahmin Uzunluğu: {meta.get('prediction_length', 10)} adım

## Kurulum
1. model/ klasörünü sunucuya kopyalayın
2. pip install autogluon.timeseries
3. DAG dosyasını ~/airflow/dags/ klasörüne kopyalayın
4. INPUT_CSV_PATH: Geçmiş veri CSV dosyası (zaman damgası + hedef sütun)
5. OUTPUT_CSV_PATH: Tahmin çıktısı

## Girdi CSV Formatı
CSV dosyası en az şu sütunları içermelidir:
- {meta.get('timestamp_column', 'timestamp')}: Zaman damgası sütunu
- {meta['target_column']}: Hedef değer sütunu
{('- ' + meta.get('item_id_column', '') + ': Seri kimlik sütunu') if meta.get('item_id_column', '__item_id') != '__item_id' else '(Tekli seri — item_id otomatik atanır)'}

## Çıktı
{meta.get('prediction_length', 10)} adımlık tahmin, timestamp + tahmin değerleri içeren CSV.
"""

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"dag_ts_tahmin_{safe_submodel}.py", dag_code)
                zf.writestr("README.md", readme)
                ag_model_path = MODELS_DIR / model_id / "agmodel"
                if ag_model_path.exists():
                    for file_path in ag_model_path.rglob('*'):
                        if file_path.is_file() and not file_path.is_symlink():
                            arcname = "model/" + file_path.relative_to(ag_model_path).as_posix()
                            zf.write(file_path, arcname)

            zip_buffer.seek(0)
            zip_bytes = zip_buffer.getvalue()

            add_activity("exported_airflow", model_id, meta["name"],
                        f"{submodel_name} için Zaman Serisi Airflow paketi dışa aktarıldı",
                        username=user["username"])

            filename = self._safe_filename(f"{safe_submodel}_ts_airflow_paketi.zip")
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(zip_bytes)))
            origin = self._cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self._cors_allowed and self._cors_allowed != "*":
                    self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
            self.end_headers()
            self.wfile.write(zip_bytes)
            return

        # ── Tabular / Text Airflow DAG ──
        dag_code = generate_airflow_dag(
            model_id=model_id, model_name=meta["name"],
            submodel_name=submodel_name, target_col=meta["target_column"],
            feature_columns=meta["feature_columns"], task_type=meta["task_type"],
            text_columns=meta.get("text_columns", []),
            embedding_model=meta.get("embedding_model", ""),
        )

        safe_submodel = re.sub(r'[^a-zA-Z0-9._-]', '_', submodel_name).lower()

        if meta.get("text_columns"):
            readme = f"""# Airflow DAG Paketi — Metin Embedding Destekli
## Model: {meta['name']}
## Alt Model: {submodel_name}
## Hedef: {meta['target_column']} ({meta['task_type']})

## Gereksinimler
- autogluon.tabular (pip install autogluon.tabular)
- sentence-transformers (pip install sentence-transformers)
- GPU gerekmez (CPU yeterli)

## Embedding Modeli: {meta.get('embedding_model', DEFAULT_EMBEDDING_MODEL)}
Metin sütunları ({', '.join(meta['text_columns'])}) otomatik olarak embedding'e dönüştürülür.

## Kurulum
1. model/ klasörünü sunucuya kopyalayın
2. pip install autogluon.tabular sentence-transformers
3. DAG dosyasını ~/airflow/dags/ klasörüne kopyalayın
"""
        else:
            readme = f"""# Airflow DAG Paketi
## Model: {meta['name']}
## Alt Model: {submodel_name}
## Hedef: {meta['target_column']} ({meta['task_type']})

## Kurulum
1. model/ klasörünü sunucuya kopyalayın
2. pip install autogluon
3. DAG dosyasını ~/airflow/dags/ klasörüne kopyalayın
"""

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"dag_tahmin_{safe_submodel}.py", dag_code)
            zf.writestr("README.md", readme)
            ag_model_path = MODELS_DIR / model_id / "agmodel"
            if ag_model_path.exists():
                for file_path in ag_model_path.rglob('*'):
                    if file_path.is_file() and not file_path.is_symlink():
                        arcname = "model/" + file_path.relative_to(ag_model_path).as_posix()
                        zf.write(file_path, arcname)
            # Include text pipeline config for NLP models
            pipeline_config_path = MODELS_DIR / model_id / "text_pipeline.json"
            if pipeline_config_path.exists():
                zf.write(pipeline_config_path, "text_pipeline.json")

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

        add_activity("exported_airflow", model_id, meta["name"],
                    f"{submodel_name} için Airflow paketi dışa aktarıldı", username=user["username"])

        filename = self._safe_filename(f"{safe_submodel}_airflow_paketi.zip")
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(zip_bytes)))
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            if self._cors_allowed and self._cors_allowed != "*":
                self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
        self.end_headers()
        self.wfile.write(zip_bytes)

    def handle_export_mssql(self, model_id, submodel_name):
        """Generate and download MSSQL export — Easy SQL only (no ONNX)."""
        user = self._require_auth()
        if not user:
            return

        if not AUTOGLUON_AVAILABLE:
            return self.send_json({"error": "AutoGluon yüklü değil"}, 500)

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        # Validate submodel exists
        if not any(sm["name"] == submodel_name for sm in meta.get("submodels", [])):
            return self.send_json({"error": f"Alt model bulunamadı: {submodel_name}"}, 404)

        # Time series models don't support SQL export
        if meta.get("task_type") == "timeseries":
            return self.send_json({
                "error": "Zaman serisi modelleri SQL olarak dışa aktarılamaz.",
                "detail": "ETS, Theta, DeepAR, TFT gibi zaman serisi modelleri dahili durum (state) "
                          "bilgisi taşır ve satır bazlı CASE WHEN mantığıyla ifade edilemez.",
                "suggestion": "Bunun yerine Airflow DAG dışa aktarımını kullanın."
            }, 400)

        # Models with text columns use TabularPredictor with embeddings — SQL export works
        # but operates on pre-computed embedding columns. Add a note to the SQL output.
        _text_sql_note = ""
        _model_text_columns = meta.get("text_columns", [])
        if _model_text_columns:
            _text_sql_note = (
                "-- NOT: Bu SQL, metin sütunlarının önceden embedding'e dönüştürülmüş halini bekler.\n"
                "-- Metin sütunları sentence-transformers ile embedding'e çevrilmeli ve\n"
                "-- embedding sütunları (örn: review_emb_0, review_emb_1, ...) tabloda bulunmalıdır.\n"
                "-- Embedding modeli: " + meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL) + "\n\n"
            )

        target_col = meta["target_column"]
        # For models with text columns, use the embedded feature columns for SQL generation
        if _model_text_columns:
            feature_cols = meta.get("feature_columns_embedded", meta["feature_columns"])
        else:
            feature_cols = meta["feature_columns"]

        if not model_ref_counter.acquire(model_id):
            return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)
        try:
            predictor = model_cache.load_model(model_id, meta)
            if predictor is None:
                return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)

            sql = generate_tree_sql(predictor, submodel_name, feature_cols, target_col)
            if sql is None:
                sql = generate_linear_sql(predictor, submodel_name, feature_cols, target_col)

            if sql is not None:
                # Prepend embedding note for text/NLP models
                if _text_sql_note:
                    sql = _text_sql_note + sql
                add_activity("exported_mssql", model_id, meta["name"],
                            f"{submodel_name} için SQL dışa aktarıldı", username=user["username"])
                safe_sub = re.sub(r'[^a-zA-Z0-9._-]', '_', submodel_name).lower()
                filename = f"tahmin_{model_id[:8]}_{safe_sub}.sql"
                return self.send_file_download(sql, filename, "application/sql")

            return self.send_json({
                "error": f"{submodel_name} için SQL dışa aktarma yöntemi bulunamadı.",
                "suggestion": "Bunun yerine Günlük Çalışabilir (Airflow) dışa aktarımını kullanın."
            }, 400)

        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": self._safe_error_message(e)}, 500)
        finally:
            model_ref_counter.release(model_id)

    def handle_set_visibility(self, model_id, body):
        user = self._require_auth()
        if not user:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)

        # Only owner or admin can change visibility
        if meta.get("owner") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Yetkiniz yok"}, 403)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        visibility = data.get("visibility", "public")
        if visibility not in ("public", "private"):
            return self.send_json({"error": "Geçersiz görünürlük"}, 400)

        old_visibility = meta.get("visibility", "private")
        was_ever_public = meta.get("_was_public", False)
        label = "Herkese Açık" if visibility == "public" else "Özel"
        meta["visibility"] = visibility
        update_fields = {"visibility": visibility}

        # Log all visibility changes for admin audit trail
        if visibility != old_visibility:
            add_activity("visibility_changed", model_id, meta.get("name", ""),
                         f"Görünürlük değiştirildi: {label}", username=user["username"])

        # Public announcement only on first-ever public share (prevents spam)
        if visibility == "public" and not was_ever_public:
            update_fields["_was_public"] = True
            user_obj = find_user(user["username"])
            display_name = user_obj.get("display_name", user["username"]) if user_obj else user["username"]
            add_activity("model_shared", model_id, meta.get("name", ""),
                         f"{display_name} bir model paylaştı: {meta.get('name', '')}",
                         username=user["username"], visibility="public")

        update_model_meta_fields(model_id, **update_fields)
        self.send_json({"success": True, "message": f"Görünürlük güncellendi: {label}"})

    def handle_endorse(self, model_id, body):
        admin = self._require_admin()
        if not admin:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz istek"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)

        endorse = data.get("endorsed", False)
        endorsed_by = admin["username"] if endorse else None
        update_model_meta_fields(model_id, endorsed=bool(endorse), endorsed_by=endorsed_by)

        if endorse:
            add_activity("endorsed", model_id, meta["name"],
                        f"'{meta['name']}' yönetici tarafından onaylandı",
                        username=admin["username"], visibility="public")
        else:
            add_activity("unendorsed", model_id, meta["name"],
                        f"Model onayı kaldırıldı", username=admin["username"])

        label = "Onaylandı" if endorse else "Onay kaldırıldı"
        self.send_json({"success": True, "message": label})

    def handle_cost_estimate(self, body):
        user = self._require_auth()
        if not user:
            return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self.send_json({"error": "Geçersiz JSON verisi"}, 400)
        if not isinstance(data, dict):
            return self.send_json({"error": "JSON nesnesi bekleniyor"}, 400)
        try:
            inference_time = float(data.get("inference_time_sec", 0.1))
            num_rows = int(data.get("num_rows", 1000))
            num_columns = int(data.get("num_columns", 10))
            frequency = data.get("frequency", "daily")
            avg_row_bytes = int(data.get("avg_row_bytes", 200))
        except (ValueError, TypeError):
            return self.send_json({"error": "Geçersiz parametre değeri"}, 400)
        result = estimate_cost(inference_time, num_rows, num_columns, frequency, avg_row_bytes)
        self.send_json(result)

    def handle_delete_model(self, model_id):
        user = self._require_auth()
        if not user:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)

        # Only owner or admin can delete
        if meta.get("owner") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu modeli silme yetkiniz yok"}, 403)

        if not model_ref_counter.try_mark_for_deletion(model_id):
            return self.send_json({"error": "Bu analiz şu anda kullanılıyor. Aktif işlerin bitmesini bekleyin."}, 409)

        model_name = meta.get("name", "Bilinmeyen")
        model_dir = MODELS_DIR / model_id

        try:
            # Remove from memory cache before deleting from disk
            model_cache.evict(model_id)
            shutil.rmtree(model_dir, ignore_errors=True)
            add_activity("deleted", model_id, model_name,
                        f"'{model_name}' silindi", username=user["username"])
            self.send_json({"success": True, "message": f"'{model_name}' silindi."})
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"Silme hatası: {self._safe_error_message(e)}"}, 500)
        finally:
            model_ref_counter.unmark_deletion(model_id)
            # Clean up per-model lock to prevent unbounded dict growth
            with _per_model_locks_guard:
                _per_model_locks.pop(model_id, None)
            invalidate_models_cache()

    def handle_explain(self, model_id, submodel_name):
        user = self._require_auth()
        if not user:
            return

        if not AUTOGLUON_AVAILABLE:
            return self.send_json({"error": "AutoGluon yüklü değil"}, 500)

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return

        if meta.get("owner") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu analiz yalnızca sahibi veya yöneticiler tarafından incelenebilir"}, 403)

        sm = None
        for s in meta.get("submodels", []):
            if s["name"] == submodel_name:
                sm = s
                break
        if not sm:
            return self.send_json({"error": f"'{submodel_name}' alt modeli bulunamadı"}, 404)

        # Auto-detect text/NLP models: apply embeddings to training data before explainability
        _text_columns_for_explainability = meta.get("text_columns", [])
        if not _text_columns_for_explainability:
            pipeline_config = _load_text_pipeline_config(model_id)
            if pipeline_config:
                _text_columns_for_explainability = pipeline_config.get("text_columns", [])
        _is_text_model = bool(_text_columns_for_explainability)
        if _is_text_model and not SENTENCE_TRANSFORMERS_AVAILABLE:
            return self.send_json({
                "error": "Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin.",
            }, 500)

        # Time series explainability analysis
        if not model_ref_counter.acquire(model_id):
            return self.send_json({"error": "Model siliniyor, lütfen bekleyin"}, 409)
        if meta.get("task_type") == "timeseries":
            try:
                csv_path = MODELS_DIR / model_id / "training_data.csv"
                if not csv_path.exists():
                    return self.send_json({"error": "Eğitim verisi bulunamadı"}, 404)

                df = _read_csv_with_fallback(csv_path)
                target_col = meta["target_column"]
                ts_col = meta.get("timestamp_column")
                id_col = meta.get("item_id_column", "__item_id")

                # Parse and sort
                if not ts_col or ts_col not in df.columns:
                    return self.send_json({"error": f"Zaman damgası sütunu '{ts_col}' bulunamadı"}, 400)
                df[ts_col] = pd.to_datetime(df[ts_col])
                if id_col not in df.columns:
                    df[id_col] = "series_0"
                if len(df) == 0:
                    return self.send_json({"error": "Eğitim verisi boş"}, 400)
                # Use first series for analysis
                first_id = df[id_col].iloc[0]
                ts = df[df[id_col] == first_id].sort_values(ts_col).copy()
                ts = ts.set_index(ts_col)
                y = ts[target_col].dropna().astype(float)

                if len(y) < 10:
                    return self.send_json({"error": "Analiz için en az 10 veri noktası gerekli"}, 400)

                result = {
                    "model_id": model_id,
                    "submodel_name": submodel_name,
                    "target_column": target_col,
                    "task_type": "timeseries",
                    "series_id": str(first_id),
                    "num_series": int(df[id_col].nunique()),
                }

                # ── 1. Basic Statistics ──
                basic_stats = {
                    "count": int(len(y)),
                    "mean": round(float(y.mean()), 4),
                    "std": round(float(y.std()), 4),
                    "min": round(float(y.min()), 4),
                    "max": round(float(y.max()), 4),
                    "cv": round(float(y.std() / y.mean()), 4) if abs(float(y.mean())) > 1e-10 else None,
                    "start": y.index.min().isoformat() if pd.notna(y.index.min()) else "N/A",
                    "end": y.index.max().isoformat() if pd.notna(y.index.max()) else "N/A",
                }
                result["basic_stats"] = basic_stats

                # ── 2. Trend Analysis (linear regression) ──
                x_vals = np.arange(len(y), dtype=float)
                y_vals = y.values.astype(float)
                slope = 0.0
                if len(x_vals) > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    trend_line = (slope * x_vals + intercept).tolist()
                    # Trend strength: R² of linear fit
                    ss_res = np.sum((y_vals - (slope * x_vals + intercept)) ** 2)
                    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    trend_direction = "yükseliş" if slope > 0 else "düşüş"
                    trend_pct_per_step = round(float(slope / y.mean() * 100), 3) if abs(float(y.mean())) > 1e-10 else 0
                else:
                    trend_line = y_vals.tolist()
                    r_squared = 0.0
                    trend_direction = "sabit"
                    trend_pct_per_step = 0

                result["trend"] = {
                    "direction": trend_direction,
                    "slope": round(float(slope), 6) if len(x_vals) > 1 else 0,
                    "r_squared": round(float(r_squared), 4),
                    "pct_change_per_step": trend_pct_per_step,
                    "trend_line": [round(v, 2) for v in trend_line],
                }

                # ── 3. Autocorrelation (ACF) ──
                max_lag = min(40, len(y) // 2 - 1)
                acf_values = []
                y_centered = y_vals - y_vals.mean()
                var = np.sum(y_centered ** 2)
                for lag in range(0, max_lag + 1):
                    if var > 0 and lag < len(y_centered):
                        acf_val = np.sum(y_centered[:len(y_centered) - lag] * y_centered[lag:]) / var
                        acf_values.append(round(float(acf_val), 4))
                    else:
                        acf_values.append(0.0)

                # Confidence band (approximate 95%)
                acf_conf = round(1.96 / np.sqrt(len(y)), 4) if len(y) > 0 else 0.1

                # Detect dominant seasonal period from ACF peaks
                seasonal_period = None
                if len(acf_values) > 4:
                    # Skip lag 0, find first significant peak
                    for lag in range(2, len(acf_values)):
                        if acf_values[lag] > acf_conf:
                            # Check if it's a local peak
                            prev_ok = lag == 2 or acf_values[lag] > acf_values[lag - 1]
                            next_ok = lag == len(acf_values) - 1 or acf_values[lag] >= acf_values[lag + 1]
                            if prev_ok and next_ok and acf_values[lag] > 0.2:
                                seasonal_period = lag
                                break

                result["autocorrelation"] = {
                    "acf_values": acf_values,
                    "confidence_band": acf_conf,
                    "max_lag": max_lag,
                    "detected_seasonal_period": seasonal_period,
                }

                # ── 4. Seasonal Decomposition ──
                decomposition = None
                seasonal_strength = 0.0
                trend_strength = 0.0
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    period = seasonal_period if seasonal_period and seasonal_period >= 2 else (
                        12 if len(y) >= 24 else (7 if len(y) >= 14 else None))
                    if period and len(y) >= period * 2:
                        decomp = seasonal_decompose(y, model='additive', period=period)
                        trend_comp = decomp.trend.dropna()
                        seasonal_comp = decomp.seasonal.dropna()
                        resid_comp = decomp.resid.dropna()

                        # Strengths (STL-style): F_t = 1 - Var(R) / Var(T+R), F_s = 1 - Var(R) / Var(S+R)
                        var_resid = resid_comp.var()
                        var_trend_resid = (trend_comp.values + resid_comp.values[:len(trend_comp)]).var() if len(trend_comp) > 0 else 1.0
                        # Align seasonal with resid by index (seasonal has full range, resid has edges trimmed)
                        common_idx = seasonal_comp.index.intersection(resid_comp.index)
                        var_season_resid = (seasonal_comp.loc[common_idx].values + resid_comp.loc[common_idx].values).var() if len(common_idx) > 0 else 1.0

                        trend_strength = max(0.0, float(1.0 - var_resid / var_trend_resid)) if var_trend_resid > 0 else 0.0
                        seasonal_strength = max(0.0, float(1.0 - var_resid / var_season_resid)) if var_season_resid > 0 else 0.0

                        # Prepare decomposition data for charting
                        _safe_round = lambda v: round(float(v), 2) if not (np.isnan(v) or np.isinf(v)) else None
                        decomposition = {
                            "period": period,
                            "trend": [_safe_round(v) for v in decomp.trend.values],
                            "seasonal": [_safe_round(v) for v in decomp.seasonal.values],
                            "residual": [_safe_round(v) for v in decomp.resid.values],
                        }
                except ImportError:
                    result["decomposition_note"] = "statsmodels yüklü değil — mevsimsel ayrıştırma atlandı"
                except Exception as e:
                    log.warning(f"[Explain] Decomposition failed: {e}")
                    result["decomposition_note"] = "Mevsimsel ayrıştırma hesaplanamadı"

                result["decomposition"] = decomposition
                result["seasonal_strength"] = round(seasonal_strength, 4)
                result["trend_strength"] = round(trend_strength, 4)

                # ── 5. Stationarity (ADF test) ──
                stationarity = {"stationary": None, "adf_stat": None, "p_value": None}
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(y_vals, maxlag=min(20, len(y) // 4))
                    stationarity = {
                        "stationary": bool(adf_result[1] < 0.05),
                        "adf_stat": round(float(adf_result[0]), 4),
                        "p_value": round(float(adf_result[1]), 6),
                        "critical_1pct": round(float(adf_result[4].get('1%', 0)), 4),
                        "critical_5pct": round(float(adf_result[4].get('5%', 0)), 4),
                    }
                except ImportError:
                    # Fallback: rolling mean stability check
                    half = len(y_vals) // 2
                    if half > 1:
                        mean_1 = y_vals[:half].mean()
                        mean_2 = y_vals[half:].mean()
                        std_1 = y_vals[:half].std()
                        std_2 = y_vals[half:].std()
                        mean_drift = abs(mean_2 - mean_1) / (y_vals.std() + 1e-10)
                        stationarity = {
                            "stationary": bool(mean_drift < 0.5),
                            "mean_drift_ratio": round(float(mean_drift), 4),
                            "note": "statsmodels yüklü değil — yaklaşık analiz"
                        }
                except Exception:
                    pass
                result["stationarity"] = stationarity

                # ── 6. Rolling Statistics for visualization ──
                window = max(3, len(y) // 10)
                rolling_mean = y.rolling(window=window, center=True).mean()
                rolling_std = y.rolling(window=window, center=True).std()
                result["rolling_stats"] = {
                    "window": window,
                    "rolling_mean": [round(float(v), 2) if not np.isnan(v) else None for v in rolling_mean.values],
                    "rolling_std": [round(float(v), 2) if not np.isnan(v) else None for v in rolling_std.values],
                }

                # ── 7. Original series data (for charts) ──
                result["series_data"] = {
                    "timestamps": [t.isoformat() for t in y.index],
                    "values": [round(float(v), 4) for v in y_vals],
                }

                # ── 8. Model Leaderboard Insight ──
                lb_models = meta.get("submodels", [])
                model_insights = []
                for m in lb_models[:5]:
                    model_insights.append({
                        "name": m["name"],
                        "mase": m["score"],
                        "type": m.get("model_type", "unknown"),
                    })
                result["model_insights"] = model_insights

                # ── 9. Text Summary (business-friendly language) ──
                # Trend description in plain language
                if trend_direction == "yükseliş":
                    trend_plain = "artış eğiliminde (zaman içinde yükseliyor)"
                elif trend_direction == "düşüş":
                    trend_plain = "azalış eğiliminde (zaman içinde düşüyor)"
                else:
                    trend_plain = "genel olarak sabit (belirgin bir yön yok)"

                trend_strength_label = (
                    "çok güçlü" if r_squared > 0.8 else
                    "güçlü" if r_squared > 0.6 else
                    "orta düzeyde" if r_squared > 0.3 else
                    "zayıf"
                )

                summary_lines = [
                    f"## Verilerinizin Özeti",
                    f"**{len(y)}** veri noktası analiz edildi "
                    f"(**{basic_stats['start'][:10]}** — **{basic_stats['end'][:10]}** arası).",
                    f"",
                    f"### Genel Yön (Trend)",
                    f"Verileriniz **{trend_plain}**. "
                    f"Bu eğilim **{trend_strength_label}** ({r_squared:.0%} oranında açıklayıcı). "
                    + (f"Her zaman adımında ortalama **%{abs(trend_pct_per_step):.2f}** "
                       f"{'artış' if trend_pct_per_step > 0 else 'azalış'} var."
                       if abs(trend_pct_per_step) > 0.001 else
                       "Adım başı değişim ihmal edilebilir düzeyde."),
                ]

                if seasonal_period:
                    season_strength_label = (
                        "çok belirgin" if seasonal_strength > 0.7 else
                        "belirgin" if seasonal_strength > 0.4 else
                        "hafif"
                    )
                    summary_lines.append(
                        f"\n### Tekrar Eden Kalıplar (Mevsimsellik)\n"
                        f"Verilerinizde **her {seasonal_period} adımda bir** tekrar eden bir kalıp tespit edildi. "
                        f"Bu kalıp **{season_strength_label}** ({seasonal_strength:.0%} güçte). "
                        f"Yani geçmişteki döngüler gelecekte de benzer şekilde tekrarlanabilir."
                    )
                else:
                    summary_lines.append(
                        "\n### Tekrar Eden Kalıplar (Mevsimsellik)\n"
                        "Belirgin bir tekrar eden kalıp (mevsimsellik) tespit edilemedi. "
                        "Bu, verilerinizin düzensiz hareket ettiği veya döngü boyutunun veri aralığından büyük olduğu anlamına gelebilir."
                    )

                if trend_strength > 0:
                    summary_lines.append(
                        f"Trend gücü: **{trend_strength:.0%}** — "
                        + ("uzun vadeli yön çok baskın" if trend_strength > 0.7 else
                           "uzun vadeli yön belirgin" if trend_strength > 0.4 else
                           "uzun vadeli yön zayıf, kısa vadeli dalgalanmalar daha baskın")
                    )

                stat_label = stationarity.get("stationary")
                if stat_label is not None:
                    if stat_label:
                        summary_lines.append(
                            f"\n### Tahmin Edilebilirlik\n"
                            f"Verileriniz **kararlı bir yapıda** — ortalaması ve dalgalanması zaman içinde fazla değişmiyor. "
                            f"Bu, modelin daha güvenilir tahminler üretebileceği anlamına gelir."
                        )
                    else:
                        summary_lines.append(
                            f"\n### Tahmin Edilebilirlik\n"
                            f"Verilerinizin ortalaması zaman içinde **değişiyor** (durağan değil). "
                            f"Bu, tahmin yapmayı biraz zorlaştırır ancak model bunu otomatik olarak hesaba katmaya çalışır."
                        )

                best_model = model_insights[0] if model_insights else None
                if best_model:
                    if best_model['mase'] < 0.5:
                        perf_label = "**Çok başarılı** — basit tekrarlama yönteminden kat kat daha iyi tahmin yapıyor."
                    elif best_model['mase'] < 1.0:
                        perf_label = "**Başarılı** — verilerinizdeki kalıpları yakalayarak basit yöntemlerden daha iyi tahmin yapıyor."
                    elif best_model['mase'] < 1.5:
                        perf_label = "**Kabul edilebilir** — basit yöntemlerle benzer düzeyde. Daha fazla veri veya farklı ayarlar ile iyileştirilebilir."
                    else:
                        perf_label = "**İyileştirme gerekli** — daha fazla veri toplanması veya model ayarlarının değiştirilmesi önerilir."
                    summary_lines.append(
                        f"\n### En İyi Model\n"
                        f"**{best_model['name']}** (skor: {best_model['mase']:.2f}) — {perf_label}"
                    )

                result["method"] = "Zaman Serisi Analizi"
                result["summary"] = "\n".join(summary_lines)

                add_activity("explained", model_id, meta["name"],
                            f"{submodel_name} için zaman serisi açıklanabilirlik analizi",
                            username=user["username"])
                return self.send_json(result)

            except Exception as e:
                traceback.print_exc()
                return self.send_json({"error": f"Zaman serisi analiz hatası: {self._safe_error_message(e)}"}, 500)
            finally:
                model_ref_counter.release(model_id)

        # ── Call Analysis (LLM) explainability ──
        if meta.get("task_type") == "call_analysis":
            try:
                ca = meta.get("call_analysis", {})
                schema = ca.get("schema", [])
                row_results = ca.get("row_results", [])

                if not row_results:
                    return self.send_json({"error": "Değerlendirme sonuçları bulunamadı"}, 404)
                # Validate schema structure
                schema = [v for v in schema if isinstance(v, dict) and "name" in v]

                result = {
                    "model_id": model_id,
                    "submodel_name": submodel_name,
                    "target_column": meta.get("target_column", ""),
                    "task_type": "call_analysis",
                    "num_files": len(row_results),
                    "method": "LLM Çağrı Analizi Özeti",
                }

                from collections import Counter

                var_analysis = {}
                for var in schema:
                    vn = var["name"]
                    vtype = var.get("type", "classification")
                    actuals = []
                    predictions = []
                    all_predictions = []
                    for r in row_results:
                        if r.get("error"):
                            continue
                        pred = (r.get("predicted") or {}).get(vn)
                        if pred is not None:
                            all_predictions.append(pred)
                        actual = (r.get("actuals") or {}).get(vn)
                        if actual is not None and pred is not None:
                            actuals.append(actual)
                            predictions.append(pred)

                    va = {"name": vn, "type": vtype, "n": len(actuals)}

                    if vtype == "classification":
                        pred_dist = dict(Counter(str(p).strip() for p in all_predictions))
                        va["predicted_distribution"] = pred_dist
                        if actuals:
                            correct = sum(1 for a, p in zip(actuals, predictions)
                                          if a is not None and p is not None
                                          and str(a).lower().strip() == str(p).lower().strip())
                            va["accuracy"] = round(correct / len(actuals) * 100, 1) if actuals else 0
                            va["correct"] = correct
                            actual_dist = dict(Counter(str(a).strip() for a in actuals))
                            va["actual_distribution"] = actual_dist
                            class_acc = {}
                            for cls in actual_dist:
                                cls_total = sum(1 for a in actuals if str(a).strip() == cls)
                                cls_correct = sum(1 for a, p in zip(actuals, predictions)
                                                  if str(a).strip() == cls and str(p).lower().strip() == cls.lower())
                                if cls_total > 0:
                                    class_acc[cls] = round(cls_correct / cls_total * 100, 1)
                            va["per_class_accuracy"] = class_acc

                    elif vtype == "regression":
                        if actuals:
                            try:
                                actual_floats = [float(a) for a in actuals]
                                pred_floats = [float(p) for p in predictions]
                                errors = [abs(a - p) for a, p in zip(actual_floats, pred_floats)]
                                va["mae"] = round(sum(errors) / len(errors), 4)
                                va["rmse"] = round((sum(e ** 2 for e in errors) / len(errors)) ** 0.5, 4)
                            except (ValueError, TypeError):
                                pass

                    var_analysis[vn] = va

                result["variable_analysis"] = var_analysis

                # Error rate
                error_count = sum(1 for r in row_results if r.get("error"))
                result["error_rate"] = round(error_count / len(row_results) * 100, 1) if row_results else 0
                result["error_count"] = error_count

                # Reasoning patterns — extract common themes from LLM explanations
                reasoning_texts = [r.get("summary_reasoning", "") for r in row_results
                                   if r.get("summary_reasoning") and not r.get("error")]
                result["reasoning_sample_count"] = len(reasoning_texts)
                # Provide first 5 reasoning samples (truncated) for display
                result["reasoning_samples"] = [
                    {"filename": r.get("filename", ""), "reasoning": (r.get("summary_reasoning") or "")[:300]}
                    for r in row_results[:5]
                    if r.get("summary_reasoning") and not r.get("error")
                ]

                has_any_actuals = any(va["n"] > 0 for va in var_analysis.values())

                summary_lines = [
                    f"## \u00c7a\u011fr\u0131 Analizi A\u00e7\u0131klanabilirlik Raporu",
                    f"**{len(row_results)}** ses dosyas\u0131 analiz edildi, **{error_count}** hata olu\u015ftu.",
                    "",
                ]

                if has_any_actuals:
                    for vn, va in var_analysis.items():
                        if va["type"] == "classification" and va["n"] > 0:
                            acc = va.get("accuracy", 0)
                            acc_label = ("\u00c7ok iyi" if acc >= 80 else "\u0130yi" if acc >= 60
                                         else "Orta" if acc >= 40 else "D\u00fc\u015f\u00fck")
                            summary_lines.append(
                                f"### {vn} (S\u0131n\u0131fland\u0131rma)\n"
                                f"Do\u011fruluk: **%{acc}** ({va.get('correct', 0)}/{va['n']}) \u2014 {acc_label}"
                            )
                            if va.get("per_class_accuracy"):
                                for cls, cls_acc in va["per_class_accuracy"].items():
                                    summary_lines.append(f"  - {cls}: %{cls_acc}")
                            agree_count = va.get("correct", 0)
                            disagree_count = va["n"] - agree_count
                            if disagree_count > 0:
                                summary_lines.append(
                                    f"LLM **{agree_count}** dosyada ger\u00e7ek de\u011ferle ayn\u0131 fikirde, **{disagree_count}** dosyada farkl\u0131 de\u011ferlendirme yapt\u0131."
                                )
                        elif va["type"] == "regression" and va["n"] > 0:
                            mae = va.get("mae")
                            if mae is not None:
                                summary_lines.append(
                                    f"### {vn} (Say\u0131sal)\n"
                                    f"Ortalama Hata (MAE): **{mae}** \u00b7 RMSE: {va.get('rmse', '?')}"
                                )
                    clf_vars = [va for va in var_analysis.values() if va["type"] == "classification" and va["n"] > 0]
                    if clf_vars:
                        avg_acc = sum(va.get("accuracy", 0) for va in clf_vars) / len(clf_vars)
                        if avg_acc >= 80:
                            summary_lines.append(f"\n### Talimat Etkinli\u011fi\nOrtalama do\u011fruluk **%{round(avg_acc, 1)}** \u2014 talimat LLM\u2019i ba\u015far\u0131l\u0131 y\u00f6nlendiriyor.")
                        elif avg_acc >= 60:
                            summary_lines.append(f"\n### Talimat Etkinli\u011fi\nOrtalama do\u011fruluk **%{round(avg_acc, 1)}** \u2014 talimat k\u0131smen etkili, iyile\u015ftirme yap\u0131labilir.")
                        else:
                            summary_lines.append(f"\n### Talimat Etkinli\u011fi\nOrtalama do\u011fruluk **%{round(avg_acc, 1)}** \u2014 talimat\u0131n g\u00f6zden ge\u00e7irilmesi \u00f6nerilir.")
                else:
                    summary_lines.append("### LLM Tahmin Da\u011f\u0131l\u0131m\u0131")
                    for vn, va in var_analysis.items():
                        pred_dist = va.get("predicted_distribution", {})
                        if va["type"] == "classification" and pred_dist:
                            dist_parts = ", ".join(
                                f"**{cnt}** dosyada '**{cls}**'" for cls, cnt in pred_dist.items()
                            )
                            summary_lines.append(f"  - **{vn}**: {dist_parts}")
                        elif va["type"] == "regression":
                            summary_lines.append(f"  - **{vn}**: say\u0131sal tahminler \u00fcretildi")
                    summary_lines.append("")
                    if reasoning_texts:
                        common_words = Counter()
                        for rt in reasoning_texts:
                            for word in rt.lower().split():
                                if len(word) > 4:
                                    common_words[word] += 1
                        top_themes = [w for w, _ in common_words.most_common(5)]
                        if top_themes:
                            summary_lines.append(
                                f"### LLM Odak Noktalar\u0131\n"
                                f"LLM gerek\u00e7elerinde s\u0131k ge\u00e7en temalar: {', '.join(top_themes)}"
                            )
                    summary_lines.append(
                        f"\n### \u00d6neri\n"
                        f"Bu analizde ger\u00e7ek de\u011ferler girilmedi\u011fi i\u00e7in do\u011fruluk \u00f6l\u00e7\u00fcm\u00fc yap\u0131lamad\u0131. "
                        f"Ger\u00e7ek de\u011ferleri girerek LLM\u2019in do\u011frulu\u011funu \u00f6l\u00e7ebilirsiniz."
                    )

                if has_any_actuals and reasoning_texts:
                    summary_lines.append(
                        f"\n### LLM Gerek\u00e7eleri\n"
                        f"LLM, {len(reasoning_texts)} dosya i\u00e7in a\u00e7\u0131klama \u00fcretti. "
                        f"A\u015fa\u011f\u0131da \u00f6rnek gerek\u00e7eler g\u00f6r\u00fcnt\u00fclenebilir."
                    )

                result["summary"] = "\n".join(summary_lines)

                add_activity("explained", model_id, meta["name"],
                            f"{submodel_name} için çağrı analizi açıklanabilirlik",
                            username=user["username"])
                return self.send_json(result)

            except Exception as e:
                traceback.print_exc()
                return self.send_json({"error": f"Çağrı analizi açıklanabilirlik hatası: {self._safe_error_message(e)}"}, 500)
            finally:
                model_ref_counter.release(model_id)

        if not _prediction_semaphore.acquire(timeout=0):
            model_ref_counter.release(model_id)
            return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyin."}, 503)

        _explain_resource_id = f"explain_{uuid.uuid4().hex[:8]}"
        _explain_resource_acquired = False
        try:
            _explain_ram = _estimate_model_ram_mb(model_id, meta)
            if not resource_manager.try_acquire(_explain_resource_id, "prediction_model_load",
                                                vram_mb=0, ram_mb=_explain_ram):
                return self.send_json({"error": "Sunucu meşgul. Lütfen birkaç saniye bekleyip tekrar deneyin."}, 503)
            _explain_resource_acquired = True

            predictor = model_cache.load_model(model_id, meta)
            if predictor is None:
                return self.send_json({"error": "Model yüklenemedi — bellek yetersiz olabilir"}, 503)

            csv_path = MODELS_DIR / model_id / "training_data.csv"
            if not csv_path.exists():
                return self.send_json({"error": "Eğitim verisi bulunamadı"}, 404)

            df = _read_csv_with_fallback(csv_path)
            target_col = meta["target_column"]
            feature_cols = meta["feature_columns"]

            df, _ = clean_dataframe(df, context="prediction")
            df = df.dropna(subset=[target_col])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

            # For text models: embed text columns, use embedded feature columns
            _emb_col_map = {}  # maps embedded col name → original text column name
            if _is_text_model and _text_columns_for_explainability:
                embedding_model_name = meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                text_cols_present = [c for c in _text_columns_for_explainability if c in df.columns]
                for col in text_cols_present:
                    df[col] = df[col].fillna("")
                if text_cols_present:
                    st_model = _get_sentence_model(embedding_model_name)
                    embed_dim = st_model.get_sentence_embedding_dimension()
                    for col in text_cols_present:
                        for i in range(embed_dim):
                            _emb_col_map[f"{col}_emb_{i}"] = col
                    df = _embed_text_columns(df, text_cols_present, embedding_model_name)
                # Use embedded feature columns for the predictor
                feature_cols = meta.get("feature_columns_embedded", feature_cols)

            sample_size = min(200, len(df))
            if sample_size == 0:
                return self.send_json({"error": "Temizlik sonrası analiz için yeterli veri kalmadı"}, 400)
            feature_cols = [c for c in feature_cols if c in df.columns]
            if not feature_cols:
                return self.send_json({"error": "Analiz için kullanılabilir özellik sütunu bulunamadı"}, 400)
            sample_df = df.sample(n=sample_size, random_state=42)
            sample_features = sample_df[feature_cols]

            result = {
                "model_id": model_id,
                "submodel_name": submodel_name,
                "target_column": target_col,
                "task_type": meta.get("task_type"),
                "num_features": len(feature_cols),
                "sample_size": sample_size,
            }

            # Try permutation importance
            importance = {}
            method_label = "Permütasyon Önemi"
            try:
                from sklearn.inspection import permutation_importance as sklearn_pi
                from sklearn.base import BaseEstimator

                class AGWrapper(BaseEstimator):
                    def __init__(self, pred, mn):
                        self.pred = pred
                        self.mn = mn
                    def fit(self, X, y=None):
                        return self
                    def predict(self, X):
                        return self.pred.predict(X, model=self.mn).values
                    def score(self, X, y):
                        preds = self.predict(X)
                        if meta.get("task_type") == "classification":
                            return float((preds == y).mean())
                        else:
                            return -float(np.sqrt(((preds - y) ** 2).mean()))

                wrapper = AGWrapper(predictor, submodel_name)
                perm_result = sklearn_pi(
                    wrapper, sample_features, sample_df[target_col],
                    n_repeats=5, random_state=42, n_jobs=1
                )
                for i, col in enumerate(feature_cols):
                    importance[col] = max(0.0, float(perm_result.importances_mean[i]))
                method_label = "Permütasyon Önemi (sklearn)"
            except Exception:
                for col in feature_cols:
                    importance[col] = 1.0 / max(1, len(feature_cols))
                method_label = "Eşit Dağılım (Varsayılan)"

            # For text models: aggregate embedding column importances → original text column names
            if _is_text_model and _emb_col_map:
                aggregated = {}
                for col, imp in importance.items():
                    original_col = _emb_col_map.get(col, col)
                    aggregated[original_col] = aggregated.get(original_col, 0.0) + imp
                importance = aggregated
                # Update feature_cols to original names for the rest of the analysis
                feature_cols = meta["feature_columns"]

            total = sum(importance.values()) or 1
            normalized = {k: v / total for k, v in importance.items()}
            sorted_features = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

            result["method"] = method_label
            result["feature_importance"] = [
                {"feature": f, "importance": imp, "raw_value": round(importance.get(f, 0), 6)}
                for f, imp in sorted_features
            ]

            # ── SHAP analysis for tree-based models ──
            # Provides model-intrinsic feature importance (more accurate than permutation)
            shap_data = None
            model_type_str = (sm.get("model_type") or sm.get("name") or "").lower()
            _tree_keywords = ["xgboost", "lightgbm", "randomforest", "extratrees",
                              "catboost", "gbm", "rf", "xt", "xgb", "lgb", "cat"]
            is_tree_model = any(kw in model_type_str for kw in _tree_keywords)

            _has_categorical = False
            if is_tree_model and not _is_text_model:
                try:
                    import shap

                    # Extract the inner sklearn/xgb/lgb model from AutoGluon wrapper
                    inner_model = None
                    try:
                        ag_model = predictor._trainer.load_model(submodel_name)
                        for attr in ['model', '_model', 'model_', '_learner', 'estimator']:
                            candidate = getattr(ag_model, attr, None)
                            if candidate is not None:
                                # Check nested (e.g. ag_model.model.model for XGBoost)
                                for attr2 in ['model', '_model', 'booster_', '_Booster']:
                                    nested = getattr(candidate, attr2, None)
                                    if nested is not None:
                                        inner_model = candidate
                                        break
                                if inner_model is None:
                                    inner_model = candidate
                                break
                    except Exception:
                        pass

                    if inner_model is not None:
                        # Use a smaller subsample for SHAP to keep it fast
                        shap_sample_size = min(100, len(sample_features))
                        shap_sample = sample_features.head(shap_sample_size).copy()

                        # Skip SHAP if categorical columns present — pd.factorize encoding
                        # doesn't match AutoGluon's internal encoding, producing wrong values
                        _has_categorical = any(
                            shap_sample[c].dtype in ['object', 'category']
                            for c in shap_sample.columns
                        )
                        if _has_categorical:
                            print(f"  [Explain] SHAP skipped: categorical features present (encoding mismatch)")
                            shap_sample = None

                    if inner_model is not None and shap_sample is not None:
                        shap_sample = shap_sample.fillna(0)

                        explainer = shap.TreeExplainer(inner_model)
                        shap_values_raw = explainer.shap_values(shap_sample)

                        # Handle different SHAP output shapes:
                        # - Binary/regression: 2D array (samples, features)
                        # - Multiclass old API: list of 2D arrays
                        # - Multiclass new API: 3D array (samples, features, classes)
                        if isinstance(shap_values_raw, list):
                            shap_abs = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
                        else:
                            shap_abs = np.abs(shap_values_raw)
                            # New SHAP API returns 3D (samples, features, classes) for multiclass
                            if shap_abs.ndim == 3:
                                shap_abs = shap_abs.mean(axis=2)  # average across classes first

                        # Mean |SHAP| per feature → 1D array
                        shap_means = shap_abs.mean(axis=0)

                        # Build SHAP importance dict
                        shap_feature_cols = list(shap_sample.columns)
                        shap_importance = {}
                        for i, col in enumerate(shap_feature_cols):
                            if i < len(shap_means):
                                shap_importance[col] = round(float(shap_means[i]), 6)

                        # Normalize
                        shap_total = sum(shap_importance.values()) or 1
                        shap_normalized = {k: round(v / shap_total, 6) for k, v in shap_importance.items()}
                        shap_sorted = sorted(shap_normalized.items(), key=lambda x: x[1], reverse=True)

                        shap_data = {
                            "shap_importance": [
                                {"feature": f, "importance": imp, "raw_mean_abs_shap": shap_importance.get(f, 0)}
                                for f, imp in shap_sorted[:20]  # top 20
                            ],
                            "method": "SHAP TreeExplainer",
                        }
                        print(f"  [Explain] SHAP computed for {submodel_name} ({len(shap_feature_cols)} features)")
                except ImportError:
                    print("  [Explain] SHAP not installed — skipping (pip install shap)")
                except (MemoryError, RuntimeError) as e:
                    err_str = str(e).lower()
                    if isinstance(e, MemoryError) or "out of memory" in err_str or "cuda" in err_str:
                        raise  # Propagate OOM to outer handler for proper 503 response
                    log.warning(f"[Explain] SHAP computation failed (non-fatal): {e}")
                except Exception as e:
                    log.warning(f"[Explain] SHAP computation failed (non-fatal): {e}")

            if shap_data:
                result["shap"] = shap_data
            elif is_tree_model and not _is_text_model:
                if _has_categorical:
                    result["shap_note"] = "SHAP analizi kategorik özellik içeren modellerde desteklenmiyor — korelasyon analizini kullanın"
                else:
                    result["shap_note"] = "SHAP analizi bu model için hesaplanamadı (kütüphane eksik veya model tipi desteklenmiyor)"

            # Correlation analysis
            # For text models, reload original df for correlation since embedding replaced text cols
            if _is_text_model:
                corr_df = _read_csv_with_fallback(csv_path)
                corr_df, _ = clean_dataframe(corr_df, context="prediction")
                corr_df = corr_df.dropna(subset=[target_col])
            else:
                corr_df = df
            correlations = {}
            for col in feature_cols:
                if col not in corr_df.columns:
                    continue
                try:
                    if corr_df[col].dtype == 'bool':
                        encoded = corr_df[col].astype(float)
                        if corr_df[target_col].dtype in ['object', 'category']:
                            target_numeric = pd.factorize(corr_df[target_col])[0].astype(float)
                            target_numeric[target_numeric == -1] = np.nan
                        else:
                            target_numeric = corr_df[target_col].astype(float)
                        mask = ~(np.isnan(encoded) | np.isnan(target_numeric))
                        if mask.sum() > 3:
                            corr = np.corrcoef(encoded[mask], target_numeric[mask])[0, 1]
                            if not np.isnan(corr):
                                correlations[col] = round(float(corr), 4)
                    elif corr_df[col].dtype in ['object', 'category']:
                        encoded = pd.factorize(corr_df[col])[0].astype(float)
                        encoded[encoded == -1] = np.nan
                        if corr_df[target_col].dtype in ['object', 'category']:
                            target_numeric = pd.factorize(corr_df[target_col])[0].astype(float)
                            target_numeric[target_numeric == -1] = np.nan
                        else:
                            target_numeric = corr_df[target_col].astype(float)
                        mask = ~(np.isnan(encoded) | np.isnan(target_numeric))
                        if mask.sum() > 3:
                            corr = np.corrcoef(encoded[mask], target_numeric[mask])[0, 1]
                            if not np.isnan(corr):
                                correlations[col] = round(float(corr), 4)
                    else:
                        if corr_df[target_col].dtype in ['object', 'category']:
                            target_numeric = pd.factorize(corr_df[target_col])[0].astype(float)
                            target_numeric[target_numeric == -1] = np.nan
                        else:
                            target_numeric = corr_df[target_col].astype(float)
                        mask = ~(corr_df[col].isna() | np.isnan(target_numeric))
                        if mask.sum() > 3:
                            corr = np.corrcoef(corr_df[col][mask].astype(float), target_numeric[mask])[0, 1]
                            if not np.isnan(corr):
                                correlations[col] = round(float(corr), 4)
                except Exception:
                    pass
            result["correlations"] = correlations

            # Data profile
            data_profile = []
            for col in feature_cols:
                source_df = corr_df if col in corr_df.columns else df
                if col not in source_df.columns:
                    continue
                prof = {"feature": col, "dtype": str(source_df[col].dtype)}
                null_pct = round(source_df[col].isna().mean() * 100, 1)
                prof["null_pct"] = null_pct
                prof["unique_count"] = int(source_df[col].nunique())
                if source_df[col].dtype in ['object', 'category']:
                    prof["type"] = "kategorik"
                    top_vals = source_df[col].value_counts().head(5).to_dict()
                    prof["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
                else:
                    prof["type"] = "sayısal"
                    prof["mean"] = round(float(source_df[col].mean()), 4) if not source_df[col].isna().all() else None
                    prof["std"] = round(float(source_df[col].std()), 4) if not source_df[col].isna().all() else None
                    prof["min"] = round(float(source_df[col].min()), 4) if not source_df[col].isna().all() else None
                    prof["max"] = round(float(source_df[col].max()), 4) if not source_df[col].isna().all() else None
                data_profile.append(prof)
            result["data_profile"] = data_profile

            # Text summary
            summary_lines = []
            model_type_label = sm.get("model_type", "bilinmeyen")
            holdout_score = sm.get("score", 0)
            summary_lines.append(
                f"## Açıklanabilirlik Raporu: {submodel_name}\n"
                f"**Model tipi:** {model_type_label.title()} | "
                f"**Test skoru:** %{holdout_score*100:.1f} | "
                f"**Özellik sayısı:** {len(feature_cols)} | "
                f"**Analiz yöntemi:** {method_label}"
            )

            top_n = min(5, len(sorted_features))
            top_features = sorted_features[:top_n]
            if top_features:
                summary_lines.append(f"\n### En Etkili {top_n} Özellik")
                for rank, (feat, imp) in enumerate(top_features, 1):
                    corr_val = correlations.get(feat)
                    corr_str = ""
                    if corr_val is not None:
                        direction = "pozitif" if corr_val > 0 else "negatif"
                        corr_str = f" (hedefle korelasyon: {corr_val:+.3f}, {direction} ilişki)"
                    prof = next((p for p in data_profile if p["feature"] == feat), {})
                    type_str = prof.get("type", "bilinmeyen")
                    null_str = f", %{prof.get('null_pct', 0)} eksik" if prof.get('null_pct', 0) > 0 else ""
                    summary_lines.append(
                        f"  {rank}. **{feat}** — önem: %{imp:.1%}, "
                        f"{type_str}{null_str}{corr_str}"
                    )

            low_features = [f for f, imp in sorted_features if imp < 0.05]
            if low_features:
                summary_lines.append(
                    f"\n### Düşük Etkili Özellikler ({len(low_features)})\n"
                    f"Bu özellikler tahminlere çok az katkı sağlıyor: {', '.join(low_features[:10])}"
                )

            result["summary"] = "\n".join(summary_lines)

            add_activity("explained", model_id, meta["name"],
                        f"{submodel_name} için açıklanabilirlik analizi", username=user["username"])
            self.send_json(result)

        except (MemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if isinstance(e, MemoryError) or "out of memory" in err_str or "cuda" in err_str:
                model_cache.evict(model_id)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.send_json({"error": "Bellek yetersiz. Lütfen tekrar deneyin."}, 503)
            else:
                traceback.print_exc()
                self.send_json({"error": self._safe_error_message(e)}, 500)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": self._safe_error_message(e)}, 500)
        finally:
            if _explain_resource_acquired:
                resource_manager.release(_explain_resource_id)
            _prediction_semaphore.release()
            model_ref_counter.release(model_id)

    # ── Audio Evaluation Handlers ─────────────────────────────────

    def handle_audio_evaluate(self, body):
        """Handle the audio evaluation multipart request."""
        user = self._require_auth()
        if not user:
            return

        if not _api_rate_limiter.check_heavy(user["username"]):
            return self.send_json({"error": "Saatlik işlem limitinize ulaştınız. Lütfen daha sonra tekrar deneyin."}, 429)

        if not FASTER_WHISPER_AVAILABLE:
            return self.send_json({"error": "Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin."}, 500)
        if not REQUESTS_AVAILABLE:
            return self.send_json({"error": "Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin."}, 500)

        # Per-user model quota check (atomic to prevent TOCTOU bypass)
        if not check_and_reserve_model_quota(user["username"]):
            return self.send_json({"error": f"Analiz limitinize ulaştınız (maksimum {MAX_MODELS_PER_USER}). Eski analizleri silerek yer açabilirsiniz."}, 429)

        # ── Per-user concurrency check ──
        _username = user["username"]
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            release_model_quota(_username)
            return self.send_json({"error": "multipart/form-data bekleniyor"}, 400)

        boundary = self._extract_boundary(content_type)
        if not boundary:
            release_model_quota(_username)
            return self.send_json({"error": "Geçersiz Content-Type başlığı"}, 400)

        parts = self._parse_multipart_multi(body, boundary)

        # Extract text fields
        model_name = parts.get("model_name", {}).get("value", f"Ses Analizi {datetime.now().strftime('%Y%m%d_%H%M')}")
        if len(model_name) > 200:
            release_model_quota(_username)
            return self.send_json({"error": "Analiz adı en fazla 200 karakter olabilir"}, 400)
        schema_json = parts.get("schema", {}).get("value", "[]")
        prompt = parts.get("prompt", {}).get("value", "")
        language = parts.get("language", {}).get("value", "turkish")
        actuals_json = parts.get("actuals", {}).get("value", "{}")
        visibility = parts.get("visibility", {}).get("value", "private")

        try:
            schema = json.loads(schema_json)
            actuals_map = json.loads(actuals_json)
        except json.JSONDecodeError as e:
            release_model_quota(_username)
            return self.send_json({"error": f"JSON ayrıştırma hatası: {self._safe_error_message(e)}"}, 400)

        if not schema:
            release_model_quota(_username)
            return self.send_json({"error": "En az bir değişken tanımlanmalı"}, 400)
        if not prompt.strip():
            release_model_quota(_username)
            return self.send_json({"error": "Değerlendirme prompt'u gerekli"}, 400)

        # Collect audio files
        audio_files_parts = parts.get("_files_audio_files", [])
        if not audio_files_parts:
            release_model_quota(_username)
            return self.send_json({"error": "En az bir ses dosyası yüklenmeli"}, 400)
        if len(audio_files_parts) > 50:
            release_model_quota(_username)
            return self.send_json({"error": f"Tek seferde en fazla 50 ses dosyası yüklenebilir ({len(audio_files_parts)} gönderildi)"}, 413)

        # Validate individual audio file sizes
        for af in audio_files_parts:
            if len(af.get("data", b"")) > MAX_AUDIO_FILE_SIZE_BYTES:
                release_model_quota(_username)
                return self.send_json({
                    "error": f"Ses dosyası '{af.get('filename', '?')}' çok büyük. "
                             f"Maksimum: {MAX_AUDIO_FILE_SIZE_MB}MB"
                }, 413)

        # Save audio files to temp directory
        model_id = str(uuid.uuid4())
        temp_dir = DATA_DIR / "temp" / f"audio_{model_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for _afi, af in enumerate(audio_files_parts):
            # Sanitize filename to prevent path traversal
            fname = re.sub(r'[^\w\-. ]', '_', af["filename"]).strip('. ')
            if not fname:
                fname = f"audio_{_afi}.wav"
            # Prepend index to prevent collisions when different names sanitize identically
            fname_base, fname_ext = os.path.splitext(fname)
            fname = f"{_afi}_{fname_base}{fname_ext}"
            fpath = temp_dir / fname
            with open(fpath, "wb") as f:
                f.write(af["data"])
            audio_files.append({"path": str(fpath), "filename": af["filename"]})

        # Atomic concurrency check + register (prevents TOCTOU race)
        job_id = str(uuid.uuid4())
        allowed, reason = user_action_tracker.try_register(user["username"], "audio_eval", job_id)
        if not allowed:
            shutil.rmtree(temp_dir, ignore_errors=True)
            release_model_quota(user["username"])
            return self.send_json({"error": reason}, 429)

        audio_eval_jobs[job_id] = {
            "status": "queued",
            "model_id": model_id,
            "total": len(audio_files),
            "processed": 0,
            "_username": user["username"],
            "_action_type": "audio_eval",
            "_resource_task_id": f"audio_eval_{job_id}",
            "_model_id": model_id,
        }

        try:
            _audio_eval_queue.submit(
                user["username"], job_id,
                audio_evaluate_pipeline,
                job_id, model_id, model_name,
                audio_files, schema, prompt, language,
                actuals_map, user["username"], visibility,
            )
        except RuntimeError:
            user_action_tracker.unregister(user["username"], "audio_eval")
            release_model_quota(user["username"])
            shutil.rmtree(temp_dir, ignore_errors=True)
            return self.send_json({"error": "Sunucu kapanıyor. Lütfen tekrar deneyin."}, 503)

        add_activity("started_audio_eval", model_id, model_name,
                     f"{len(audio_files)} ses dosyası ile değerlendirme başlatıldı",
                     username=user["username"])

        return self.send_json({"job_id": job_id, "model_id": model_id, "status": "queued"})

    def handle_audio_eval_status(self, job_id):
        """Return the status of an audio evaluation job."""
        user = self._require_auth()
        if not user:
            return

        job = audio_eval_jobs.get(job_id)
        if not job:
            return self.send_json({"error": "İş bulunamadı"}, 404)
        if job.get("_username") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu işi görüntüleme yetkiniz yok"}, 403)

        response = {
            "status": job["status"],
            "model_id": job.get("model_id"),
            "total": job.get("total", 0),
            "processed": job.get("processed", 0),
        }
        if job["status"] == "queued":
            response["position"] = _audio_eval_queue.get_position(job_id)
        if job["status"] == "done":
            meta = load_model_meta(job["model_id"])
            if meta:
                response["model"] = meta
        if job["status"] == "error":
            response["error"] = job.get("error", "Bilinmeyen hata")

        self.send_json(response)

    def handle_active_audio_eval(self):
        """Check if current user has an active audio evaluation job."""
        user = self._require_auth()
        if not user:
            return
        active = user_action_tracker.get_active(user["username"])
        audio_job_id = active.get("audio_eval")
        if audio_job_id:
            job = audio_eval_jobs.get(audio_job_id)
            if job and job.get("status") not in ("done", "error"):
                return self.send_json({
                    "active": True,
                    "job_id": audio_job_id,
                    "status": job.get("status"),
                    "model_id": job.get("model_id"),
                    "total": job.get("total", 0),
                    "processed": job.get("processed", 0),
                })
        self.send_json({"active": False})

    # ── Audio Predict Handlers (run saved call_analysis models) ───

    def handle_audio_predict(self, model_id, body):
        """Accept audio files and run them through a saved call_analysis model."""
        user = self._require_auth()
        if not user:
            return

        if not _api_rate_limiter.check_heavy(user["username"]):
            return self.send_json({"error": "Saatlik işlem limitinize ulaştınız. Lütfen daha sonra tekrar deneyin."}, 429)

        if not FASTER_WHISPER_AVAILABLE:
            return self.send_json({"error": "Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin."}, 500)
        if not REQUESTS_AVAILABLE:
            return self.send_json({"error": "Gerekli bileşen yüklü değil. Lütfen sistem yöneticisine bildirin."}, 500)

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        if meta.get("task_type") != "call_analysis":
            return self.send_json({"error": "Bu model bir Çağrı Analizi modeli değil"}, 400)

        ca = meta.get("call_analysis", {})
        schema = ca.get("schema", [])
        prompt = ca.get("prompt", "")
        language = ca.get("language", "turkish")

        if not schema or not prompt:
            return self.send_json({"error": "Model şeması veya prompt eksik"}, 400)

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return self.send_json({"error": "multipart/form-data bekleniyor"}, 400)

        boundary = self._extract_boundary(content_type)
        if not boundary:
            return self.send_json({"error": "Geçersiz Content-Type başlığı"}, 400)

        parts = self._parse_multipart_multi(body, boundary)

        audio_files_parts = parts.get("_files_audio_files", [])
        if not audio_files_parts:
            return self.send_json({"error": "En az bir ses dosyası yüklenmeli"}, 400)
        if len(audio_files_parts) > 50:
            return self.send_json({"error": f"Tek seferde en fazla 50 ses dosyası yüklenebilir ({len(audio_files_parts)} gönderildi)"}, 413)

        # Validate individual audio file sizes
        for af in audio_files_parts:
            if len(af.get("data", b"")) > MAX_AUDIO_FILE_SIZE_BYTES:
                return self.send_json({
                    "error": f"Ses dosyası '{af.get('filename', '?')}' çok büyük. "
                             f"Maksimum: {MAX_AUDIO_FILE_SIZE_MB}MB"
                }, 413)

        # Save audio files to temp directory
        temp_id = str(uuid.uuid4())
        temp_dir = DATA_DIR / "temp" / f"audio_predict_{temp_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for _afi, af in enumerate(audio_files_parts):
            fname = re.sub(r'[^\w\-. ]', '_', af["filename"]).strip('. ')
            if not fname:
                fname = f"audio_{_afi}.wav"
            # Prepend index to prevent collisions when different names sanitize identically
            fname_base, fname_ext = os.path.splitext(fname)
            fname = f"{_afi}_{fname_base}{fname_ext}"
            fpath = temp_dir / fname
            with open(fpath, "wb") as f:
                f.write(af["data"])
            audio_files.append({"path": str(fpath), "filename": af["filename"]})

        # Atomic concurrency check + register
        job_id = str(uuid.uuid4())
        allowed, reason = user_action_tracker.try_register(user["username"], "audio_predict", job_id)
        if not allowed:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return self.send_json({"error": reason}, 429)

        audio_predict_jobs[job_id] = {
            "status": "queued",
            "model_id": model_id,
            "total": len(audio_files),
            "processed": 0,
            "_username": user["username"],
            "_action_type": "audio_predict",
            "_resource_task_id": f"audio_predict_{job_id}",
            "_model_id": model_id,
        }

        try:
            _audio_eval_queue.submit(
                user["username"], job_id,
                audio_predict_pipeline,
                job_id, model_id,
                audio_files, schema, prompt, language,
                user["username"],
            )
        except RuntimeError:
            user_action_tracker.unregister(user["username"], "audio_predict")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return self.send_json({"error": "Sunucu kapanıyor. Lütfen tekrar deneyin."}, 503)

        add_activity("started_audio_predict", model_id, meta["name"],
                     f"{len(audio_files)} ses dosyası ile tahmin başlatıldı",
                     username=user["username"])

        return self.send_json({"job_id": job_id, "model_id": model_id, "status": "queued"})

    def handle_audio_predict_status(self, job_id):
        """Return the status of an audio prediction job."""
        user = self._require_auth()
        if not user:
            return

        job = audio_predict_jobs.get(job_id)
        if not job:
            return self.send_json({"error": "İş bulunamadı"}, 404)
        if job.get("_username") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu işi görüntüleme yetkiniz yok"}, 403)

        response = {
            "status": job["status"],
            "model_id": job.get("model_id"),
            "total": job.get("total", 0),
            "processed": job.get("processed", 0),
        }
        if job["status"] == "queued":
            response["position"] = _audio_eval_queue.get_position(job_id)
        if job["status"] == "done":
            response["row_results"] = job.get("row_results", [])
        if job["status"] == "error":
            response["error"] = job.get("error", "Bilinmeyen hata")

        self.send_json(response)

    def handle_audio_predict_download(self, job_id):
        """Download the CSV results of a completed audio prediction job."""
        user = self._require_auth()
        if not user:
            return

        job = audio_predict_jobs.get(job_id)
        if not job:
            return self.send_json({"error": "İş bulunamadı"}, 404)
        if job.get("_username") != user["username"] and user["role"] not in ("admin", "master_admin"):
            return self.send_json({"error": "Bu işi indirme yetkiniz yok"}, 403)
        if job["status"] != "done":
            return self.send_json({"error": "İş henüz tamamlanmadı"}, 400)

        csv_content = job.get("csv_content", "")
        model_id = job.get("model_id", "unknown")
        filename = f"tahminler_ses_{model_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return self.send_file_download(csv_content, filename, "text/csv")

    def handle_call_analysis_download(self, model_id):
        """Download the evaluation results of a call_analysis model as CSV."""
        user = self._require_auth()
        if not user:
            return

        meta = load_model_meta(model_id)
        if not meta:
            return self.send_json({"error": "Model bulunamadı"}, 404)
        if not self._check_model_access(meta, user):
            return
        if meta.get("task_type") != "call_analysis":
            return self.send_json({"error": "Bu model bir Çağrı Analizi modeli değil"}, 400)

        ca = meta.get("call_analysis", {})
        schema = ca.get("schema", [])
        row_results = ca.get("row_results", [])

        if not row_results:
            return self.send_json({"error": "Sonuç verisi bulunamadı"}, 404)

        # Build CSV
        csv_buffer = io.StringIO()
        var_names = [v["name"] for v in schema]
        # Columns: filename, then for each variable: actual + predicted, then LLM explanation
        fieldnames = ["filename"]
        for vn in var_names:
            fieldnames.append(f"{vn}_gercek")
            fieldnames.append(f"{vn}_tahmin")
        fieldnames.append("llm_aciklama")

        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for r in row_results:
            csv_row = {"filename": r.get("filename", "")}
            for vn in var_names:
                csv_row[f"{vn}_gercek"] = (r.get("actuals") or {}).get(vn, "")
                csv_row[f"{vn}_tahmin"] = (r.get("predicted") or {}).get(vn, "")
            csv_row["llm_aciklama"] = r.get("summary_reasoning") or r.get("error") or ""
            writer.writerow(csv_row)

        model_name_safe = re.sub(r'[^\w\-]', '_', meta.get("name", "model"))
        filename = f"sonuclar_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return self.send_file_download(csv_buffer.getvalue(), filename, "text/csv")

    # ── Multipart Parser ───────────────────────────────────────

    def _extract_boundary(self, content_type: str) -> str:
        """Extract multipart boundary from Content-Type, handling quotes and extra params."""
        try:
            raw = content_type.split("boundary=")[1].strip()
            # Remove anything after semicolon (extra params)
            if ";" in raw:
                raw = raw.split(";")[0].strip()
            # Remove surrounding quotes
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            # Validate boundary length and format (RFC 2046: max 70 chars)
            if not raw or len(raw) > 200:
                return ""
            return raw
        except (IndexError, AttributeError):
            return ""

    def _parse_multipart(self, body: bytes, boundary: str) -> dict:
        parts = {}
        boundary_bytes = boundary.encode()
        # RFC 2046: boundaries are preceded by CRLF (except at start of body)
        delimiter = b"\r\n--" + boundary_bytes
        # Handle first boundary (may not have leading CRLF)
        first_boundary = b"--" + boundary_bytes
        start_idx = body.find(first_boundary)
        if start_idx == -1:
            return parts
        body_after_first = body[start_idx + len(first_boundary):]
        sections = body_after_first.split(delimiter)

        for idx, section in enumerate(sections):
            if b"Content-Disposition" not in section:
                continue
            # Strip leading CRLF from section
            if section.startswith(b"\r\n"):
                section = section[2:]
            header_end = section.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            header_part = section[:header_end].decode("utf-8", errors="replace")
            content = section[header_end + 4:]
            # Remove trailing CRLF (part of the delimiter, not the content)
            if content.endswith(b"\r\n"):
                content = content[:-2]
            # Remove closing boundary marker "--" only from the LAST section
            if idx == len(sections) - 1:
                if content.endswith(b"--\r\n"):
                    content = content[:-4]
                elif content.endswith(b"--"):
                    content = content[:-2]

            name_match = re.search(r'name="([^"]+)"', header_part)
            filename_match = re.search(r'filename="([^"]+)"', header_part)

            if name_match:
                field_name = name_match.group(1)
                if filename_match:
                    parts[field_name] = {"data": content, "filename": filename_match.group(1)}
                else:
                    parts[field_name] = {"data": content, "value": content.decode("utf-8", errors="replace").strip()}

        return parts

    def _parse_multipart_multi(self, body: bytes, boundary: str) -> dict:
        """Parse multipart form data supporting multiple files with the same field name.
        File fields are collected into lists under the key '_files_{fieldname}'.
        """
        parts = {}
        boundary_bytes = boundary.encode()
        sections = body.split(b"--" + boundary_bytes)

        for idx, section in enumerate(sections):
            if b"Content-Disposition" not in section:
                continue
            header_end = section.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            header_part = section[:header_end].decode("utf-8", errors="replace")
            content = section[header_end + 4:]
            # Strip trailing CRLF (delimiter artifact, not content)
            if content.endswith(b"\r\n"):
                content = content[:-2]
            # Only strip closing boundary marker "--" from the LAST section
            if idx == len(sections) - 1:
                if content.endswith(b"--"):
                    content = content[:-2]
                if content.endswith(b"\r\n"):
                    content = content[:-2]

            name_match = re.search(r'name="([^"]+)"', header_part)
            filename_match = re.search(r'filename="([^"]*)"', header_part)

            if name_match:
                field_name = name_match.group(1)
                if filename_match and filename_match.group(1):
                    # File field — collect into a list
                    list_key = f"_files_{field_name}"
                    if list_key not in parts:
                        parts[list_key] = []
                    parts[list_key].append({
                        "data": content,
                        "filename": filename_match.group(1),
                    })
                else:
                    # Text field
                    parts[field_name] = {
                        "data": content,
                        "value": content.decode("utf-8", errors="replace").strip(),
                    }

        return parts


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    # ── Initialize master admin (moved from module-level to avoid side effects on import) ──
    init_master_admin()

    # ── Startup cleanup ──
    _startup_cleanup()

    # ── Bundled LLM server ──
    _start_bundled_llama()

    # ── Pre-download ML models (background thread) ──
    _warmup_models()

    # ── Resource summary ──
    res = resource_manager.get_status()
    vram_info = (f"{res['vram_total_mb']}MB total, {res['vram_safe_mb']}MB budget"
                 if res['vram_total_mb'] > 0
                 else "Tracking disabled (torch can't see GPU)")

    # Detect if ctranslate2 can see a GPU even when torch can't
    _ct2_gpu = False
    try:
        import ctranslate2
        _ct2_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass

    _whisper_device_label = "GPU (ct2)" if _ct2_gpu else "CPU"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Tahmin Platformu — Production                       ║
║           AutoGluon Destekli                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Sunucu:    http://localhost:{PORT} (threaded)                  ║
║  AutoGluon: {'Mevcut ✓' if AUTOGLUON_AVAILABLE else 'YÜKLÜ DEĞİL ✗'}                                  ║
║  TimeSeries:{'Mevcut ✓' if AUTOGLUON_TS_AVAILABLE else 'YÜKLÜ DEĞİL ✗'}                                  ║
║  NLP/Embed: {'Mevcut ✓' if SENTENCE_TRANSFORMERS_AVAILABLE else 'YÜKLÜ DEĞİL ✗'}                                  ║
║  GPU/CUDA:  {'Mevcut ✓' if CUDA_AVAILABLE else 'Yok (CPU modu) ✗'}                                  ║
║  GPU VRAM:  {vram_info:<47}║
║  RAM:       {res['ram_total_mb']}MB total, {res['ram_safe_mb']}MB budget{' ' * max(0, 30 - len(str(res['ram_total_mb'])))}║
║  CPUs:      {res['cpu_count']:<47}║
║  Whisper:   {'Mevcut ✓ (' + _whisper_device_label + ')' if FASTER_WHISPER_AVAILABLE else 'YÜKLÜ DEĞİL ✗'}                            ║
║  LLM API:   {LLAMA_CPP_URL:<47}║
║  Veri:      {str(DATA_DIR):<47}║
╠══════════════════════════════════════════════════════════════╣
║  Concurrency: 1 training + 1 audio pipeline + N predictions  ║
║  LLM Retry:   3 attempts with exponential backoff             ║
║  File I/O:    Atomic writes with per-file locks               ║
║  Max Upload:  {MAX_UPLOAD_SIZE_MB}MB CSV / {MAX_AUDIO_FILE_SIZE_MB}MB audio                            ║
║  Batch Limit: {MAX_BATCH_ROWS:,} rows / TS horizon: {MAX_PREDICTION_LENGTH} steps         ║
║  Login:       5 attempts / 5 min rate limit                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Create threaded HTTP server with connection limit ──
    class BoundedThreadingHTTPServer(http.server.ThreadingHTTPServer):
        """ThreadingHTTPServer with a bounded thread pool to prevent thread exhaustion DoS."""
        _MAX_THREADS = 100

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._thread_semaphore = threading.BoundedSemaphore(self._MAX_THREADS)

        def process_request(self, request, client_address):
            if self._thread_semaphore.acquire(blocking=False):
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self._thread_semaphore.release()
                    raise
            else:
                # Too many connections — send 503 then close
                try:
                    request.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\n"
                        b"Content-Type: text/plain\r\n"
                        b"Content-Length: 29\r\n"
                        b"Connection: close\r\n"
                        b"\r\n"
                        b"Server busy, try again later."
                    )
                except Exception:
                    pass
                finally:
                    try:
                        request.close()
                    except Exception:
                        pass

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self._thread_semaphore.release()

    server = BoundedThreadingHTTPServer((HOST, PORT), PredictionAPIHandler)
    server.daemon_threads = True  # Threads die with the server

    # ── Graceful shutdown handler ──
    _shutdown_event = threading.Event()
    _SHUTDOWN_TIMEOUT = 8  # seconds — must finish before container runtime sends SIGKILL (typically 10s)

    def _graceful_shutdown(signum, frame):
        if _shutdown_event.is_set():
            # Second signal — force exit
            log.warning("Second signal received — forcing exit.")
            os._exit(1)

        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        log.info(f"Received {sig_name} — initiating graceful shutdown...")
        _shutdown_event.set()

        # Shutdown server from a separate thread (server.shutdown blocks until serve_forever returns)
        def _do_shutdown():
            log.info("Stopping HTTP server...")
            server.shutdown()
        shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True)
        shutdown_thread.start()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    # ── atexit fallback ──
    def _atexit_cleanup():
        if not _shutdown_event.is_set():
            log.info("atexit cleanup: releasing resources...")
        # Always clean up pools and whisper, whether shutdown was graceful or not
        log.info(f"Shutting down thread pools (timeout={_SHUTDOWN_TIMEOUT}s)...")
        _training_pool.shutdown(wait=False)
        _audio_eval_pool.shutdown(wait=False)
        _unload_whisper_model()
        _stop_bundled_llama()
        log.info("Cleanup complete.")

    atexit.register(_atexit_cleanup)

    # ── Run server ──
    log.info(f"Server starting on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass  # Signal handler already invoked
    log.info("Server stopped. Running atexit cleanup...")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()