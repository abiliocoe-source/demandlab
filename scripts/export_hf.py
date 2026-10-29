"""Copy model card, models and metrics into hf_repo/. Does NOT upload anything."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "hf_repo"
shutil.copy(root / "MODEL_CARD.md", out / "README.md")
for sub in ("models", "reports"):
    shutil.copytree(root / sub, out / sub, dirs_exist_ok=True)
print("hf_repo ready (local only). To publish later: huggingface-cli upload <user>/<repo> hf_repo")
