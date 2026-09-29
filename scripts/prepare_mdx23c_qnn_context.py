from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import time

import onnx
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_model(pattern: str) -> Path:
    matches = sorted(
        REPO_ROOT.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    matches = [p for p in matches if p.suffix.lower() == ".onnx"]
    if not matches:
        raise FileNotFoundError(f"No ONNX model matched: {pattern}")
    return matches[0]


def register_qnn():
    ep_name = "QNNExecutionProvider"
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    if not devices:
        ort.register_execution_provider_library(
            ep_name,
            qnn_ep.get_library_path(),
        )
        devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    if not devices:
        raise RuntimeError("QNN device not found after plugin registration.")
    return ep_name, devices


def make_static_model(
    dynamic_path: Path,
    batch_size: int,
    frames: int,
) -> Path:
    marker = f".winarm64_core_t{frames}.onnx"
    if dynamic_path.name.endswith(marker):
        stem = dynamic_path.name[: -len(marker)]
        static_name = (
            f"{stem}.winarm64_qnn_b{batch_size}_t{frames}.onnx"
        )
    else:
        static_name = (
            f"{dynamic_path.stem}.winarm64_qnn_b{batch_size}_"
            f"t{frames}.onnx"
        )

    static_path = dynamic_path.with_name(static_name)

    if (
        static_path.exists()
        and static_path.stat().st_mtime >= dynamic_path.stat().st_mtime
    ):
        print(f"Static QNN graph is up to date: {static_path}")
        return static_path

    print(
        f"Creating static QNN graph batch={batch_size} "
        f"frames={frames}"
    )
    model = onnx.load(str(dynamic_path))
    replacements = 0

    for value in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_param"):
                continue

            if dim.dim_param == "batch":
                dim.dim_value = int(batch_size)
                replacements += 1
            elif dim.dim_param == "frames":
                dim.dim_value = int(frames)
                replacements += 1

    if replacements == 0:
        raise RuntimeError(
            "Dynamic MDX23C graph has no batch/frames symbolic dimensions."
        )

    onnx.checker.check_model(model)
    temp_path = Path(str(static_path) + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        onnx.save(model, str(temp_path))
        os.replace(temp_path, static_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    print(f"fixed_symbolic_dims={replacements}")
    print(f"static_model={static_path}")
    return static_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Already-static MDX23C QNN ONNX model path.",
    )
    parser.add_argument(
        "--dynamic-pattern",
        default=(
            "models/MDX_Net_Models/"
            "*MDX23C-8KFFT-InstVoc_HQ*.winarm64_core_t256.onnx"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument(
        "--optimization-mode",
        default="1",
        choices=["0", "1", "2", "3"],
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("batch-size must be >= 1")

    if args.model is not None:
        static_path = args.model.resolve()
    else:
        dynamic_path = find_model(args.dynamic_pattern).resolve()
        static_path = make_static_model(
            dynamic_path,
            args.batch_size,
            args.frames,
        )

    root = static_path.with_suffix("")
    context_path = Path(str(root) + ".ctx.onnx")

    print(f"machine={platform.machine()}")
    print(f"onnxruntime={ort.__version__}")
    print(f"onnxruntime_qnn={qnn_ep.__version__}")
    print(f"static_model={static_path}")
    print(f"context_model={context_path}")
    print(f"batch_size={args.batch_size}")
    print(f"frames={args.frames}")
    print(f"optimization_mode={args.optimization_mode}")

    if (
        context_path.exists()
        and context_path.stat().st_mtime >= static_path.stat().st_mtime
    ):
        print("PASS: QNN context cache is already up to date.")
        return

    if context_path.exists():
        context_path.unlink()

    ep_name, qnn_devices = register_qnn()
    print(f"qnn_devices={qnn_devices}")

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry(
        "session.disable_cpu_ep_fallback",
        "1",
    )
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("ep.context_embed_mode", "1")
    options.add_session_config_entry(
        "ep.context_file_path",
        str(context_path),
    )

    options.add_provider_for_devices(
        qnn_devices,
        {
            "backend_path": qnn_ep.get_qnn_htp_path(),
            "htp_graph_finalization_optimization_mode": (
                args.optimization_mode
            ),
            "enable_htp_prepare_only": "1",
        },
    )

    print(
        "Compiling real MDX23C weights into QNN context cache. "
        "NPU utilization may remain low during graph preparation.",
        flush=True,
    )
    started = time.perf_counter()

    session = ort.InferenceSession(
        str(static_path),
        sess_options=options,
    )

    elapsed = time.perf_counter() - started
    print(f"prepare_elapsed={elapsed:.3f}s")

    del session

    if not context_path.exists():
        raise RuntimeError(
            "QNN session preparation completed but context file was not created."
        )

    print(
        f"context_size="
        f"{context_path.stat().st_size / (1024 * 1024):.1f} MiB"
    )
    print("PASS: precompiled MDX23C QNN context created.")
    print(
        "UVR will prefer this batch context and fall back to batch=1 "
        "if needed."
    )


if __name__ == "__main__":
    main()
