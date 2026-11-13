import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitStats:
    source_images: int
    train_images: int
    val_images: int
    copied: int
    skipped_missing_annotation: int


def build_paths(root: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root.iterdir() if path.suffix.lower() in allowed and path.is_file())


def _copy_file(src: Path, dst: Path, *, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Fallback for cross-device or permission issues.
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_pair(
    image_path: Path,
    annotations_dir: Path,
    out_images: Path,
    out_annotations: Path,
    *,
    skip_missing_annotation: bool,
    use_symlink: bool,
) -> bool:
    annotation_path = annotations_dir / f"{image_path.stem}.txt"
    if not annotation_path.exists():
        if skip_missing_annotation:
            return False
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    target_image = out_images / image_path.name
    target_annotation = out_annotations / annotation_path.name
    _copy_file(image_path, target_image, use_symlink=use_symlink)
    _copy_file(annotation_path, target_annotation, use_symlink=use_symlink)
    return True


def create_split(
    image_paths: list[Path],
    annotations_dir: Path,
    out_images: Path,
    out_annotations: Path,
    *,
    train_count: int,
    val_count: int,
    seed: int,
    skip_missing_annotation: bool,
    use_symlink: bool,
) -> SplitStats:
    if train_count < 0 or val_count < 0:
        raise ValueError("train_count and val_count must be non-negative")

    rng = random.Random(seed)
    ordered = list(image_paths)
    rng.shuffle(ordered)

    train_targets = ordered[:train_count] if train_count else []
    remaining = ordered[train_count:]
    if train_count >= len(ordered):
        remaining = []
    val_targets = remaining[:val_count] if val_count else []

    copied = 0
    missing = 0

    def fill(split_images: list[Path]) -> int:
        nonlocal copied, missing
        count = 0
        train_dir_images = out_images / "train"
        train_dir_annotations = out_annotations / "train"
        val_dir_images = out_images / "val"
        val_dir_annotations = out_annotations / "val"
        is_train = split_images is train_targets
        target_image_dir = train_dir_images if is_train else val_dir_images
        target_ann_dir = train_dir_annotations if is_train else val_dir_annotations
        for image_path in split_images:
            try:
                if _copy_pair(
                    image_path,
                    annotations_dir,
                    target_image_dir,
                    target_ann_dir,
                    skip_missing_annotation=skip_missing_annotation,
                    use_symlink=use_symlink,
                ):
                    count += 1
                    copied += 1
                else:
                    missing += 1
            except FileNotFoundError:
                missing += 1
        return count

    train_ok = fill(train_targets)
    val_ok = fill(val_targets)
    return SplitStats(
        source_images=len(ordered),
        train_images=train_ok,
        val_images=val_ok,
        copied=copied,
        skipped_missing_annotation=missing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny VisDrone train/val subset")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--skip-missing-annotation", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    use_symlink = args.symlink and not args.copy
    image_paths = build_paths(args.images_dir)
    if not image_paths:
        raise ValueError(f"No image files found in {args.images_dir}")

    out_root = args.out_dir
    # Keep compatibility with current README command style.
    for path in (args.out_dir / "train" / "images", args.out_dir / "train" / "annotations", args.out_dir / "val" / "images", args.out_dir / "val" / "annotations"):
        path.mkdir(parents=True, exist_ok=True)

    max_train = min(args.train_count, len(image_paths))
    max_val = min(args.val_count, max(0, len(image_paths) - max_train))
    if args.val_count > max_val:
        print(f"warn: val_count clipped from {args.val_count} to {max_val} due to source size", flush=True)
    if args.train_count > max_train:
        print(f"warn: train_count clipped from {args.train_count} to {max_train} due to source size", flush=True)

    stats = create_split(
        image_paths,
        args.annotations_dir,
        out_root,
        out_root,
        train_count=max_train,
        val_count=max_val,
        seed=args.seed,
        skip_missing_annotation=args.skip_missing_annotation,
        use_symlink=use_symlink,
    )

    manifest = {
        "source_images": stats.source_images,
        "train_count_requested": args.train_count,
        "val_count_requested": args.val_count,
        "train_count_used": max_train,
        "val_count_used": max_val,
        "train_images_copied": stats.train_images,
        "val_images_copied": stats.val_images,
        "copied_pairs": stats.copied,
        "skipped_missing_annotation": stats.skipped_missing_annotation,
        "seed": args.seed,
        "images_dir": str(args.images_dir),
        "annotations_dir": str(args.annotations_dir),
        "out_dir": str(args.out_dir),
        "symlink": use_symlink,
        "skip_missing_annotation": args.skip_missing_annotation,
    }
    print(
        "done "
        f"source_images={stats.source_images} "
        f"train={stats.train_images} "
        f"val={stats.val_images} "
        f"missing_annotation={stats.skipped_missing_annotation} "
        f"out={args.out_dir}"
    )

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        manifest_path = args.out_dir / "tiny_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
