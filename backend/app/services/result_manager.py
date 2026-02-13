import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from config import get_config

logger = logging.getLogger(__name__)


class ResultManager:
    """File-based storage for analysis results."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            # Use config to get the correct absolute path (e.g. /app/results in Docker)
            config = get_config()
            storage_path = config.RESULTS_FOLDER

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.storage_path / 'index.json'
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {'results': [], 'count': 0}

    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def save_result(self, result_id: str, result: Dict[str, Any]) -> bool:
        try:
            result_file = self.storage_path / f"{result_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            self.index['results'].insert(0, {
                'id': result_id,
                'type': result.get('type', 'unknown'),
                'verdict': result.get('verdict', 'unknown'),
                'confidence': result.get('confidence', 0),
                'model': result.get('model', 'unknown'),
                'timestamp': result.get('timestamp', datetime.now().isoformat())
            })
            self.index['count'] = len(self.index['results'])
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        result_file = self.storage_path / f"{result_id}.json"
        if not result_file.exists():
            return None
        with open(result_file, 'r') as f:
            return json.load(f)

    def get_all_results(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        self._load_index()
        return self.index['results'][offset:offset + limit]

    def get_total_count(self) -> int:
        return self.index['count']

    def delete_result(self, result_id: str) -> bool:
        try:
            result_file = self.storage_path / f"{result_id}.json"
            if result_file.exists():
                os.remove(result_file)

            self.index['results'] = [r for r in self.index['results'] if r['id'] != result_id]
            self.index['count'] = len(self.index['results'])
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"Error deleting result: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            for f in self.storage_path.glob("*.json"):
                if f.name != 'index.json':
                    os.remove(f)
            self.index = {'results': [], 'count': 0}
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"Error clearing results: {e}")
            return False
