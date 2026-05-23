import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_manifests(outputs_root: Path) -> Iterable[Path]:
    runs_dir = outputs_root / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/manifest.json"))


def load_run_manifests(outputs_root: Path) -> List[Dict[str, Any]]:
    manifests: List[Dict[str, Any]] = []
    for manifest_path in iter_run_manifests(outputs_root):
        try:
            manifests.append(read_json(manifest_path))
        except Exception:
            continue
    return manifests
