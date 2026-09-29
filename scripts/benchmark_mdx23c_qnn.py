from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import torch
import yaml
from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib_v5.tfc_tdf_v3 import TFC_TDF_Core, TFC_TDF_net


def load_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as handle:
        return ConfigDict(yaml.safe_load(handle))


def export_dynamic_core(config: ConfigDict, output_path: Path) -> None:
    model = TFC_TDF_net(config).eval()
    wrapper = TFC_TDF_Core(model).eval()

    input_channels = int(config.audio.num_channels) * 2
    export_frames = 32
    dummy = torch.zeros(
        1,
        input_channels,
        int(config.audio.dim_f),
        export_frames,
        dtype=torch.float32,
    )

    output_time_axis = 4 if model.num_target_instruments > 1 else 3
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Exporting compact dynamic MDX23C core ({export_frames} frames)")
    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["spectrogram"],
        output_names=["estimated_spectrogram"],
        dynamic_axes={
            "spectrogram": {3: "frames"},
            "estimated_spectrogram": {output_time_axis: "frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"export_elapsed={time.perf_counter() - started:.3f}s")


def make_frames_static(source: Path, destination: Path, frames: int) -> None:
    print(f"[2/5] Fixing QNN time axis to frames={frames}")
    model = onnx.load(str(source))

    replacements = 0
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param == "frames":
                dim.dim_value = frames
                replacements += 1

    if replacements == 0:
        raise RuntimeError("No symbolic 'frames' dimensions were found.")

    onnx.save(model, str(destination))
    print(f"fixed_symbolic_dims={replacements}")


def create_qnn_session(model_path: Path):
    print("[3/5] Compiling model for Snapdragon NPU (QNN HTP)")

    ep_name = "QNNExecutionProvider"
    ep_lib_path = qnn_ep.get_library_path()
    htp_path = qnn_ep.get_qnn_htp_path()

    print(f"qnn_plugin={ep_lib_path}")
    print(f"qnn_htp={htp_path}")

    ort.register_execution_provider_library(ep_name, ep_lib_path)
    devices = ort.get_ep_devices()
    qnn_devices = [device for device in devices if device.ep_name == ep_name]
    print(f"qnn_devices={qnn_devices}")

    if not qnn_devices:
        raise RuntimeError("QNN plugin registered, but no QNN EP device was found.")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    ep_options = {
        "backend_path": htp_path,
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3",
    }
    options.add_provider_for_devices(qnn_devices, ep_options)

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
    )
    elapsed = time.perf_counter() - started

    print(f"qnn_session_create={elapsed:.3f}s")
    return session, ep_name


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
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    print(f"machine={platform.machine()}")
    print(f"onnxruntime={ort.__version__}")

    config = load_config(args.config)
    work_dir = REPO_ROOT / "bench" / "qnn"
    dynamic_path = work_dir / "mdx23c_core_dynamic.onnx"
    static_path = work_dir / f"mdx23c_core_static_f{args.frames}.onnx"

    if args.force_export or not dynamic_path.exists():
        export_dynamic_core(config, dynamic_path)
    else:
        print(f"[1/5] Reusing dynamic ONNX: {dynamic_path}")

    if args.force_export or not static_path.exists():
        make_frames_static(dynamic_path, static_path, args.frames)
    else:
        print(f"[2/5] Reusing static ONNX: {static_path}")

    session, qnn_ep_name = create_qnn_session(static_path)

    print("[4/5] Running NPU inference")
    input_channels = int(config.audio.num_channels) * 2
    x = np.random.default_rng(0).standard_normal(
        (1, input_channels, int(config.audio.dim_f), args.frames),
        dtype=np.float32,
    )

    times = []
    output = None
    for run in range(1, args.runs + 1):
        started = time.perf_counter()
        output = session.run(None, {"spectrogram": x})[0]
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"QNN NPU run={run} elapsed={elapsed:.3f}s", flush=True)

    assert output is not None
    print("[5/5] PASS: QNN HTP completed with CPU fallback disabled")
    print(f"output_shape={output.shape}")
    print(f"best={min(times):.3f}s")
    print(f"average={sum(times) / len(times):.3f}s")

    del session
    ort.unregister_execution_provider_library(qnn_ep_name)


if __name__ == "__main__":
    main()
