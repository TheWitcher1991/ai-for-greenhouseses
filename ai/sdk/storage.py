import json
from typing import Any, Dict

from sdk.contracts import StorageAdapter, StorageType


class Storage(StorageAdapter):
    def __init__(self, storage_type: StorageType = StorageType.json):
        self.storage_type = storage_type

    def save(self, path: str, state_dict: Dict) -> None:
        if self.storage_type == StorageType.json:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> Any:
        if self.storage_type == StorageType.json:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)


json_storage = Storage(StorageType.json)
