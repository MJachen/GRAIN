"""Verify that commands run under the bound legacy attention interpreter."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.data.paths import RepositoryPathPolicy
from grain.config import load_config


def collect() -> dict[str, object]:
    modules: dict[str, object] = {}
    loaded: dict[str, object] = {}
    for import_name, display_name in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit_learn"),
        ("scipy", "scipy"),
    ):
        try:
            module = importlib.import_module(import_name)
            loaded[import_name] = module
            modules[display_name] = {
                "importable": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as error:
            modules[display_name] = {
                "importable": False,
                "error": f"{type(error).__name__}: {error}",
            }

    report: dict[str, object] = {
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "environment_prefix": Path(sys.prefix).resolve().as_posix(),
        "python_version": platform.python_version(),
        "packages": modules,
    }
    torch = loaded.get("torch")
    if torch is not None:
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        report["gpu_count"] = int(torch.cuda.device_count())
        report["gpus"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default="configs/environment.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    expected = json.loads(
        (REPOSITORY_ROOT / args.expected).read_text(encoding="utf-8")
    )
    report = collect()
    report["environment_name"] = expected["environment_name"]
    report["paper_environment"] = expected["paper_environment"]

    expected_executable = Path(expected["python_executable"]).resolve()
    if Path(sys.executable).resolve() != expected_executable:
        raise SystemExit(
            f"Wrong interpreter: {sys.executable}; expected {expected_executable}"
        )
    if not report["packages"]["torch"]["importable"]:
        raise SystemExit("PyTorch import failed in the bound attention environment")

    if args.output:
        base_config = load_config(REPOSITORY_ROOT / "configs" / "base.json")
        policy = RepositoryPathPolicy.create(
            REPOSITORY_ROOT, base_config["data"]["source_root"]
        )
        output = policy.resolve_write(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
