from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "search_cache"

# Models
LOCAL_MODEL = "Qwen/Qwen3-0.6B"
TRAINING_MODEL = "Qwen/Qwen3-7B"

# Search
DDG_RATE_LIMIT = 1.0  # requests per second
SERPER_RATE_LIMIT = 5.0
DEFAULT_SEARCH_MAX_RESULTS = 5
DEFAULT_FETCH_MAX_CHARS = 3000
