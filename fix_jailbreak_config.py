
import json
from pathlib import Path

cache = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--zhx123--ftrobertallm"
    / "snapshots"
)

configs = list(cache.glob("*/config.json"))

if not configs:
    print("Model config.json not found in Hugging Face cache.")
else:
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))

        data["id2label"] = {
            str(key): str(value)
            for key, value in data["id2label"].items()
        }

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

        print("Fixed:", path)
