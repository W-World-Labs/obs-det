import argparse
import time

import torch

from omnidet.config import ModelConfig
from omnidet.models.detector import OmniDet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--neck-variant", choices=["conv_gated", "ssm_lite"], default="conv_gated")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RTX 5090 profiling")

    cfg = ModelConfig(input_size=args.input_size, neck_variant=args.neck_variant)
    model = OmniDet(cfg).cuda().eval().half()
    image = torch.randn(1, 3, args.input_size, args.input_size, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / args.iters
    fps = 1000.0 / latency_ms
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"latency_ms={latency_ms:.3f}")
    print(f"fps={fps:.2f}")
    print(f"max_memory_mb={memory_mb:.1f}")


if __name__ == "__main__":
    main()
