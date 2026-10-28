"""
Download the LeRobot MetaWorld MT50 dataset to a local directory.

The output directory is the LeRobot dataset root (meta/, data/, videos/), so point
MT50_DATA_ROOT at it for training:

    python dataset_downloader.py
    export MT50_DATA_ROOT=/zfsauton/scratch/avenugo2/mt50
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download lerobot/metaworld_mt50 from the HuggingFace Hub.")
    parser.add_argument("--repo-id", default="lerobot/metaworld_mt50")
    parser.add_argument("--output-dir", default="/zfsauton/scratch/avenugo2/mt50")
    parser.add_argument("--revision", default=None, help="Branch/tag/commit. Defaults to main.")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {output_dir}")
    path = snapshot_download(
        repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
        local_dir=str(output_dir), max_workers=args.max_workers,
    )

    info = output_dir / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"Download finished but {info} is missing; not a valid LeRobot dataset root.")
    print(f"Done: {path}")
    print(f"export MT50_DATA_ROOT={output_dir}")


if __name__ == "__main__":
    main()
