from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import yaml
from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_v5.tfc_tdf_v3 import TFC_TDF_net


class MDX23CCore(torch.nn.Module):
    def __init__(self, model: TFC_TDF_net):
        super().__init__()
        self.model = model

    def forward(self, spectrogram):
        return self.model.forward_spectrogram(spectrogram)


def load_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as handle:
        return ConfigDict(yaml.safe_load(handle))


def export_core(config: ConfigDict, output_path: Path, export_frames: int) -> None:
    print(f"[1/3] Building MDX23C core for ONNX export")
    model = TFC_TDF_net(config).eval()
    wrapper = MDX23CCore(model).eval()

    input_channels = int(config.audio.num_channels) * 2
    dummy = torch.randn(
        1,
        input_channels,
        int(config.audio.dim_f),
        export_frames,
        dtype=torch.float32,
    )

    if wrapper.model.num_target_instruments > 1:
        dynamic_axes = {
            "spectrogram": {3: "frames"},
            "estimated_spectrogram": {4: "frames"},
        }
    else:
        dynamic_axes = {
            "spectrogram": {3: "frames"},
            "estimated_spectrogram": {3: "frames"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["spectrogram"],
        output_names=["estimated_spectrogram"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    elapsed = time.perf_counter() - started
    print(f"ONNX export: {elapsed:.3f}s")
    print(f"ONNX file: {output_path}")


def benchmark_ort(
    config: ConfigDict,
    onnx_path: Path,
    frames: int,
    threads: int,
    runs: int,
) -> None:
    print("[2/3] Creating ONNX Runtime session")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_started = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_elapsed = time.perf_counter() - session_started

    input_channels = int(config.audio.num_channels) * 2
    x = np.random.default_rng(0).standard_normal(
        (1, input_channels, int(config.audio.dim_f), frames),
        dtype=np.float32,
    )

    print(f"machine={platform.machine()}")
    print(f"onnxruntime={ort.__version__}")
    print(f"providers={session.get_providers()}")
    print(f"threads={threads}")
    print(f"input_shape={x.shape}")
    print(f"session_create={session_elapsed:.3f}s")

    times = []
    output = None
    for run in range(1, runs + 1):
        started = time.perf_counter()
        output = session.run(None, {"spectrogram": x})[0]
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"ORT run={run} elapsed={elapsed:.3f}s", flush=True)

    assert output is not None
    print(f"output_shape={output.shape}")
    print(f"best={min(times):.3f}s")
    print(f"average={sum(times)/len(times):.3f}s")
    print("[3/3] PASS: MDX23C ONNX Runtime CPU benchmark completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "models"
            / "MDX_Net_Models"
            / "model_data"
            / "mdx_c_configs"
            / "model_2_stem_full_band_8k.yaml"
        ),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / "bench" / "mdx23c_8k_core_random.onnx",
    )
    parser.add_argument("--export-frames", type=int, default=32)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.export_frames % (2 ** int(config.model.num_scales)) != 0:
        raise SystemExit("export-frames must be divisible by 2**num_scales")

    if args.frames % (2 ** int(config.model.num_scales)) != 0:
        raise SystemExit("frames must be divisible by 2**num_scales")

    if args.force_export or not args.onnx.exists():
        export_core(config, args.onnx, args.export_frames)
    else:
        print(f"[1/3] Reusing ONNX: {args.onnx}")

    benchmark_ort(
        config=config,
        onnx_path=args.onnx,
        frames=args.frames,
        threads=args.threads,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
