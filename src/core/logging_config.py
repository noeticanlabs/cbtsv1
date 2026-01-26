# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]

import logging
import json
import time
import threading
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": threading.current_thread().name,
            "thread_id": threading.get_ident(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def setup_logging(level=logging.INFO, log_file=None):
    """Setup structured JSON logging for GR solver components."""
    logger = logging.getLogger('gr_solver')
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = JSONFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Child loggers for components
    components = ['orchestrator', 'stepper', 'constraints', 'solver']
    for comp in components:
        comp_logger = logging.getLogger(f'gr_solver.{comp}')
        comp_logger.setLevel(level)
        comp_logger.propagate = True  # Propagate to parent

    return logger

# Utility function for timing
class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def elapsed_ms(self):
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000

# Utility function to compute array statistics
def array_stats(arr, name=""):
    """Compute min, max, mean for array."""
    if arr is None or arr.size == 0:
        return {"name": name, "min": None, "max": None, "mean": None, "shape": None}

    flat_arr = arr.flatten()
    return {
        "name": name,
        "min": float(np.min(flat_arr)) if flat_arr.size > 0 else None,
        "max": float(np.max(flat_arr)) if flat_arr.size > 0 else None,
        "mean": float(np.mean(flat_arr)) if flat_arr.size > 0 else None,
        "shape": list(arr.shape) if hasattr(arr, 'shape') else None
    }

import numpy as np  # Import here to avoid circular imports

class ReceiptLevels:
    """Configuration for receipt emission levels."""
    def __init__(self):
        self.K = 100  # macro every K steps
        self.enable_M_step = True
        self.enable_M_solve = True
        self.enable_macro = True