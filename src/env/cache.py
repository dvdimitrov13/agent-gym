import hashlib
import json
import threading
from pathlib import Path


class SearchCache:
    def __init__(self, cache_dir: Path | None = None):
        self._memory: dict[str, str] = {}
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts: str) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, *parts: str) -> str | None:
        key = self._key(*parts)
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        # Try disk
        if self._cache_dir:
            path = self._cache_dir / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text())
                with self._lock:
                    self._memory[key] = data["value"]
                return data["value"]
        return None

    def set(self, *parts: str, value: str):
        key = self._key(*parts)
        with self._lock:
            self._memory[key] = value
        if self._cache_dir:
            path = self._cache_dir / f"{key}.json"
            path.write_text(json.dumps({"parts": list(parts), "value": value}))
