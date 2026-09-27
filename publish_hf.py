"""Publish the frozen dataset and the trained adapters to Hugging Face.

Run after copying artifacts into release/gemma4-calculus-sft/:
  data/                     frozen splits + manifest
  hf-dataset-card.md
  hf-model-card.md
  hf-model/                 selected adapter at root, plus
                            learning-curve-adapters/ and results/
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
DATASET_REPO = "ankit-pn/gemma4-calculus-differentiation"
MODEL_REPO = "ankit-pn/gemma4-e2b-calculus-lora"


def token():
    env = Path.home() / "Projects/.env"
    for line in env.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("HF_TOKEN not found in ~/Projects/.env")


def main():
    api = HfApi(token=token())
    user = api.whoami()["name"]
    print("Authenticated as", user)

    api.create_repo(DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "data"), path_in_repo="data",
                      repo_id=DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(ROOT / "hf-dataset-card.md"),
                    path_in_repo="README.md", repo_id=DATASET_REPO, repo_type="dataset")
    print("DATASET", f"https://huggingface.co/datasets/{DATASET_REPO}")

    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "hf-model"), path_in_repo="",
                      repo_id=MODEL_REPO)
    api.upload_file(path_or_fileobj=str(ROOT / "hf-model-card.md"),
                    path_in_repo="README.md", repo_id=MODEL_REPO)
    print("MODEL", f"https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
