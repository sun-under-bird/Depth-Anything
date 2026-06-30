import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args():
    """解析 RKNN 转换所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="将 Depth Anything V2 ONNX 模型转换为 RKNN。")
    parser.add_argument("--onnx", required=True, help="输入 ONNX 模型路径。")
    parser.add_argument("--output", required=True, help="输出 RKNN 模型路径。")
    parser.add_argument("--target-platform", default="rk3588", help="RK3588S2 使用 rk3588 目标平台。")
    parser.add_argument("--quantize", action="store_true", help="启用 INT8 量化。")
    parser.add_argument("--dataset", default=None, help="INT8 量化校准集列表。")
    return parser.parse_args()


def build_rknn(onnx_path, output_path, target_platform, quantize, dataset):
    """将 ONNX 模型转换为 RK3588S2 可用的 RKNN 模型。"""
    if quantize and not dataset:
        raise ValueError("--dataset is required when --quantize is enabled")

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        # 先保留 FP/FP16 精度，INT8 量化误差需要单独评估。
        if quantize:
            ret = rknn.build(do_quantization=True, dataset=dataset)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()


def main():
    """执行完整的 RKNN 转换流程。"""
    args = parse_args()
    build_rknn(args.onnx, args.output, args.target_platform, args.quantize, args.dataset)
    print(f"RKNN exported: {args.output}")


if __name__ == "__main__":
    main()
