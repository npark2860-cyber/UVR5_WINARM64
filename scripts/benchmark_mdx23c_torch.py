from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

import torch
import yaml
from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_v5.tfc_tdf_v3 import TFC_TDF_net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "models"
            / "MDX_Net_Models"
            / "model_data"
            / "mdx_c_configs"
            / "model_2_stem_full_band_8k.yaml"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 12))

    with open(args.config, "r", encoding="utf-8") as handle:
        config = ConfigDict(yaml.safe_load(handle))

    model = TFC_TDF_net(config).eval()
    chunk = int(config.audio.hop_length * (config.inference.dim_t - 1))
    x = torch.randn(1, 2, chunk, dtype=torch.float32)

    print(f"machine={platform.machine()}")
    print(f"torch={torch.__version__}")
    print(f"threads={torch.get_num_threads()}")
    print(f"config={args.config}")
    print(f"input_shape={tuple(x.shape)}")
    print(torch.__config__.show())

    times = []
    with torch.inference_mode():
        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            y = model(x)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            print(
                f"run={run} elapsed={elapsed:.3f}s output_shape={tuple(y.shape)}",
                flush=True,
            )

    print(f"best={min(times):.3f}s")
    print(f"average={sum(times)/len(times):.3f}s")


if __name__ == "__main__":
    main()
