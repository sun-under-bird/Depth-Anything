import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from depth_anything_v2.dinov2_layers import attention, block  # noqa: E402


# 即使环境中安装了 xFormers，也强制走标准 PyTorch attention，方便导出 ONNX。
attention.XFORMERS_AVAILABLE = False
block.XFORMERS_AVAILABLE = False


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args():
    """解析相对深度模型导出 ONNX 所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="将 Depth Anything V2 相对深度模型导出为 RKNN 可转换的 ONNX。")
    parser.add_argument("--encoder", default="vits", choices=MODEL_CONFIGS.keys(), help="模型编码器类型。")
    parser.add_argument("--checkpoint", required=True, help="相对深度 .pth 权重路径。")
    parser.add_argument("--output", required=True, help="输出 ONNX 文件路径。")
    parser.add_argument("--height", type=int, default=322, help="固定 ONNX 输入高度，必须能被 14 整除。")
    parser.add_argument("--width", type=int, default=322, help="固定 ONNX 输入宽度，必须能被 14 整除。")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本。")
    return parser.parse_args()


def check_input_size(height, width):
    """检查固定输入尺寸是否满足 DINOv2 patch 大小约束。"""
    # DINOv2 的 patch size 是 14，RKNN 静态输入高宽必须都能被 14 整除。
    if height % 14 != 0 or width % 14 != 0:
        raise ValueError(f"height and width must be divisible by 14, got {height}x{width}")


def build_model(encoder, checkpoint):
    """加载相对深度模型和权重。"""
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model, output, height, width, opset):
    """使用静态 NCHW 输入将 PyTorch 模型导出为 ONNX。"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["depth"],
        opset_version=opset,
        do_constant_folding=True,
    )


def main():
    """执行完整的相对深度 ONNX 导出流程。"""
    args = parse_args()
    check_input_size(args.height, args.width)
    model = build_model(args.encoder, args.checkpoint)
    export_onnx(model, args.output, args.height, args.width, args.opset)
    print(f"ONNX exported: {args.output}")


if __name__ == "__main__":
    main()
