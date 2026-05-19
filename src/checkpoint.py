"""Model checkpoint save/load utilities."""
import os, json
from typing import Any, Dict

def save_checkpoint(state_dict: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.json', 'w') as f:
        json.dump({k: v for k, v in state_dict.items() if isinstance(v, (int, float, str, bool))}, f)

def load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path + '.json', 'r') as f:
        return json.load(f)
