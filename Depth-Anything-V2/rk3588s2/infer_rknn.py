import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    """解析 RK3588S2 板端推理参数。"""
    parser = argparse.ArgumentParser(description="在 RK3588S2 上运行 Depth Anything V2 RKNN 推理。")
    parser.add_argument("--model", required=True, help="RKNN 模型路径。")
    parser.add_argument("--img-path", required=True, help="输入图片、图片目录或 txt 图片列表。")
    parser.add_argument("--outdir", default="rk3588s2_outputs", help="输出目录。")
    parser.add_argument("--height", type=int, default=322, help="RKNN 输入高度，必须和 ONNX 导出一致。")
    parser.add_argument("--width", type=int, default=322, help="RKNN 输入宽度，必须和 ONNX 导出一致。")
    parser.add_argument("--save-numpy", action="store_true", help="保存原始深度 .npy。")
    parser.add_argument("--grayscale", action="store_true", help="保存灰度深度可视化图。")
    return parser.parse_args()


def iter_image_paths(img_path):
    """从图片文件、目录或 txt 列表中收集输入图片路径。"""
    path = Path(img_path)
    if path.is_file() and path.suffix.lower() == ".txt":
        # txt 列表适合批量对比 PyTorch 和 RKNN 的输出。
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_file():
        return [path]

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in suffixes)


def load_rknn(model_path):
    """加载 RKNN 模型并初始化 NPU runtime。"""
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    core_mask = getattr(RKNNLite, "NPU_CORE_AUTO", None)
    ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    return rknn


def preprocess_image(image_bgr, height, width):
    """将 OpenCV BGR 图片转换为归一化 NCHW float32 输入。"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    image_rgb = (image_rgb - MEAN) / STD
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    return np.expand_dims(np.ascontiguousarray(image_chw), axis=0).astype(np.float32)


def run_inference(rknn, image_bgr, height, width):
    """执行 RKNN 推理，并将深度图缩放回原图大小。"""
    input_tensor = preprocess_image(image_bgr, height, width)
    outputs = rknn.inference(inputs=[input_tensor])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("rknn.inference returned no outputs")

    depth = np.squeeze(outputs[0]).astype(np.float32)
    original_h, original_w = image_bgr.shape[:2]

    # 模型输出是静态导出尺寸下的深度图，需要还原到原图尺寸。
    return cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_CUBIC)


def depth_to_visual(depth, grayscale):
    """将原始深度转换为 8-bit 可视化图片。"""
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-6:
        depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
    else:
        depth_u8 = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)

    if grayscale:
        return depth_u8
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def save_result(depth, image_path, outdir, save_numpy, grayscale):
    """保存单张图片的深度推理结果。"""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    if save_numpy:
        np.save(outdir_path / f"{stem}_raw_depth.npy", depth)

    visual = depth_to_visual(depth, grayscale)
    cv2.imwrite(str(outdir_path / f"{stem}_depth.png"), visual)


def main():
    """在 RK3588S2 上执行批量图片推理。"""
    args = parse_args()
    image_paths = iter_image_paths(args.img_path)
    if not image_paths:
        raise FileNotFoundError(f"No input images found: {args.img_path}")

    rknn = load_rknn(args.model)
    try:
        for index, image_path in enumerate(image_paths, start=1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Skip unreadable image: {image_path}")
                continue

            print(f"Progress {index}/{len(image_paths)}: {image_path}")
            depth = run_inference(rknn, image_bgr, args.height, args.width)
            save_result(depth, image_path, args.outdir, args.save_numpy, args.grayscale)
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
