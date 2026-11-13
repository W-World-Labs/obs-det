import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import urllib.request
from urllib.error import HTTPError, URLError

from huggingface_hub import HfApi, snapshot_download

from omnidet.data.hf_visdrone import export_fiftyone_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Voxel51/VisDrone2019-DET")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("/data/omnidet_datasets/visdrone_hf/train"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/omnidet_datasets/hf_cache"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--samples-json", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--manual-download", action="store_true", default=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.samples_json:
        metadata_path = args.samples_json.parent
        samples = json.loads(args.samples_json.read_text(encoding="utf-8"))["samples"]
        dataset_sha = "local-samples-json"
    else:
        api = HfApi(endpoint=args.hf_endpoint) if args.hf_endpoint else HfApi()
        info = api.dataset_info(args.dataset)
        metadata_path = Path(
            snapshot_download(
                args.dataset,
                repo_type="dataset",
                allow_patterns=["samples.json", "metadata.json", "README.md"],
                cache_dir=str(args.cache_dir),
                endpoint=args.hf_endpoint,
            )
        )
        samples = json.loads((metadata_path / "samples.json").read_text(encoding="utf-8"))["samples"]
        dataset_sha = info.sha
    selected = [sample for sample in samples if args.split in sample.get("tags", [])]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if args.metadata_only:
        print(f"dataset={args.dataset} sha={dataset_sha} split={args.split} samples={len(selected)}")
        return

    if args.manual_download:
        snapshot_dir = args.cache_dir / "manual" / args.dataset.replace("/", "--")
        try:
            _download_images(
                args.dataset,
                selected,
                snapshot_dir,
                args.hf_endpoint,
                args.workers,
                args.retries,
            )
        except Exception:
            if not args.allow_partial:
                raise
            print("download_incomplete=true; exporting cached subset")
        for filename in ("samples.json", "metadata.json", "README.md"):
            source = metadata_path / filename
            if source.exists():
                target = snapshot_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    else:
        image_patterns = [sample["filepath"] for sample in selected]
        snapshot_dir = Path(
            snapshot_download(
                args.dataset,
                repo_type="dataset",
                allow_patterns=["samples.json", "metadata.json", "README.md", *image_patterns],
                cache_dir=str(args.cache_dir),
                endpoint=args.hf_endpoint,
            )
        )
    exported = 0
    for index, sample in enumerate(selected):
        if export_fiftyone_sample(sample, snapshot_dir, args.out, index=index):
            exported += 1
        if index % 500 == 0:
            print(f"processed={index + 1} exported={exported}")
    print(f"done samples={len(selected)} exported={exported} out={args.out}")


def _download_images(
    dataset: str,
    samples: list[dict],
    snapshot_dir: Path,
    endpoint: str | None,
    workers: int,
    retries: int,
) -> None:
    base = (endpoint or "https://huggingface.co").rstrip("/")
    repo = dataset
    targets = []
    for sample in samples:
        rel = sample["filepath"]
        target = snapshot_dir / rel
        if not target.exists():
            targets.append((rel, target))
    print(f"download_missing={len(targets)} workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_one, f"{base}/datasets/{repo}/resolve/main/{rel}", target, retries)
            for rel, target in targets
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 500 == 0 or index == len(futures):
                print(f"downloaded={index}/{len(futures)}")


def _download_one(url: str, target: Path, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "omnidet-dataset-downloader"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                tmp.write_bytes(response.read())
            tmp.replace(target)
            return
        except (HTTPError, URLError, TimeoutError):
            if attempt >= retries:
                raise
            sleep_seconds = min(60, 2**attempt)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
