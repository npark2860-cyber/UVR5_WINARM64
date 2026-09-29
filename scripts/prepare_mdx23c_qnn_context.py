from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_static_model(pattern: str) -> Path:
    matches = sorted(
        REPO_ROOT.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    matches = [
        p for p in matches
        if p.suffix.lower() == ".onnx"
        and not p.name.endswith(".ctx.onnx")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No static QNN model matched: {pattern}"
        )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Static MDX23C QNN ONNX model path.",
    )
    parser.add_argument(
        "--pattern",
        default="models/MDX_Net_Models/*MDX23C-8KFFT-InstVoc_HQ*.winarm64_qnn_b1_t256.onnx",
    )
    parser.add_argument(
        "--optimization-mode",
        default="1",
        choices=["0", "1", "2", "3"],
    )
    args = parser.parse_args()

    static_path = (
        args.model.resolve()
        if args.model is not None
        else find_static_model(args.pattern).resolve()
    )

    root = static_path.with_suffix("")
    context_path = Path(str(root) + ".ctx.onnx")

    print(f"machine={platform.machine()}")
    print(f"onnxruntime={ort.__version__}")
    print(f"onnxruntime_qnn={qnn_ep.__version__}")
    print(f"static_model={static_path}")
    print(f"context_model={context_path}")
    print(f"optimization_mode={args.optimization_mode}")

    if context_path.exists() and context_path.stat().st_mtime >= static_path.stat().st_mtime:
        print("PASS: QNN context cache is already up to date.")
        return

    if context_path.exists():
        context_path.unlink()

    ep_name, qnn_devices = register_qnn()
    print(f"qnn_devices={qnn_devices}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("ep.context_embed_mode", "1")
    options.add_session_config_entry("ep.context_file_path", str(context_path))

    options.add_provider_for_devices(
        qnn_devices,
        {
            "backend_path": qnn_ep.get_qnn_htp_path(),
            "htp_graph_finalization_optimization_mode": args.optimization_mode,
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

    print(f"context_size={context_path.stat().st_size / (1024 * 1024):.1f} MiB")
    print("PASS: precompiled MDX23C QNN context created.")
    print("Now start UVR normally; it will load this cache instead of compiling.")


if __name__ == "__main__":
    main()
