import sys

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    print(f"python {sys.version.split()[0]}")
    print(f"torch {torch.__version__}")
    print(f"mps {torch.backends.mps.is_available()}")

    x = torch.randn(512, 512)
    y = torch.randn(512, 512)
    z = x @ y
    print(f"matmul_shape {tuple(z.shape)}")

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _ = x @ y

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))


if __name__ == "__main__":
    main()
