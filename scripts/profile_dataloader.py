"""Profile the VisDrone data pipeline to find what caps training throughput.

Breaks the per-sample cost into stages (raw read / decode / resize / full
__getitem__) and sweeps DataLoader worker counts, so you can see whether the
GPU is starved by disk I/O, JPEG decode, resize, or just too few workers.

CPU-only and read-only: safe to run alongside a training job, though the
worker sweep will compete for CPU cores. The per-stage single-process numbers
barely perturb a running job; keep --workers modest if one is active.
"""

import argparse
import io
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from omnidet.data.dataset import VisDroneDetectionDataset, collate_detections

# DataLoader workers sharing tensors over the default 'file_descriptor' strategy
# hit the kernel's open-fd limit at modest worker counts and crash with
# "received 0 items of ancdata". 'file_system' shares via /dev/shm and scales.
torch.multiprocessing.set_sharing_strategy("file_system")


def bench(label: str, fn, count: int) -> float:
    start = time.perf_counter()
    total_bytes = fn()
    elapsed = time.perf_counter() - start
    rate = count / elapsed if elapsed > 0 else 0.0
    line = f"  {label:<32} {rate:9.1f} img/s   ({elapsed:.2f}s for {count})"
    if total_bytes:
        line += f"   {total_bytes / elapsed / 1e6:8.1f} MB/s"
    print(line, flush=True)
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--sample", type=int, default=512, help="images to benchmark")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 8, 16, 24])
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()

    dataset = VisDroneDetectionDataset(args.images, args.annotations, image_size=args.image_size)
    count = min(args.sample, len(dataset))
    paths = dataset.image_paths[:count]
    size = args.image_size
    print(f"dataset={len(dataset)} images  |  benchmarking {count}  |  image_size={size}\n", flush=True)

    print("per-stage, single process (each stage = the previous one + more work):", flush=True)
    blobs: list[bytes] = []

    def stage_read() -> int:
        total = 0
        for path in paths:
            data = path.read_bytes()
            total += len(data)
            blobs.append(data)
        return total

    def stage_decode() -> int:
        for data in blobs:
            Image.open(io.BytesIO(data)).convert("RGB").load()
        return 0

    def stage_resize() -> int:
        for data in blobs:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image.resize((size, size), Image.Resampling.BILINEAR).load()
        return 0

    def stage_getitem() -> int:
        for index in range(count):
            dataset[index]
        return 0

    bench("1. raw file read", stage_read, count)
    bench("2. + JPEG decode", stage_decode, count)
    bench(f"3. + resize to {size}", stage_resize, count)
    bench("4. full __getitem__ (1 proc)", stage_getitem, count)

    print(
        f"\nDataLoader throughput (batch_size={args.batch_size}, "
        f"prefetch_factor={args.prefetch_factor}):",
        flush=True,
    )
    for workers in args.workers:
        extra: dict = {}
        if workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_detections,
            pin_memory=False,
            drop_last=True,
            **extra,
        )
        iterator = iter(loader)
        next(iterator)  # warm up: pay worker spawn before timing
        seen = 0
        start = time.perf_counter()
        for images, _ in iterator:
            seen += images.shape[0]
            if seen >= count:
                break
        elapsed = time.perf_counter() - start
        del iterator, loader
        rate = seen / elapsed if elapsed > 0 else 0.0
        print(f"  num_workers={workers:<3} {rate:9.1f} img/s   ({seen} imgs in {elapsed:.2f}s)", flush=True)

    print(
        "\nread top-down: the stage where img/s drops sharply is the wall.\n"
        "the DataLoader rate is the hard ceiling on (training step rate x batch_size).",
        flush=True,
    )


if __name__ == "__main__":
    main()
