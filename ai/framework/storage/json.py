import json
from typing import Any, Dict

from framework.contracts import StorageAdapter


class JSONStorage(StorageAdapter):

    def save(self, path: str, state_dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


json_storage = JSONStorage()
