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
    parser.add_argument("--data-format", default="nchw", choices=["nchw", "nhwc", "none"], help="送入 RKNNLite 的输入布局。")
    parser.add_argument("--save-numpy", action="store_true", help="保存原始深度 .npy。")
    parser.add_argument("--grayscale", action="store_true", help="保存灰度深度可视化图。")
    parser.add_argument("--print-stats", action="store_true", help="打印深度图的数值范围，便于和 PyTorch 输出对比。")
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


def resize_and_crop_image(image_rgb, target_height, target_width):
    """将 RGB 图像等比例缩放后裁剪到 RKNN 固定输入尺寸。"""
    original_height, original_width = image_rgb.shape[:2]
    scale = max(target_width / original_width, target_height / original_height)
    resized_width = max(target_width, int(np.ceil(original_width * scale)))
    resized_height = max(target_height, int(np.ceil(original_height * scale)))

    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

    crop_x = max(0, (resized_width - target_width) // 2)
    crop_y = max(0, resized_height - target_height)
    cropped = resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "target_height": target_height,
        "target_width": target_width,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "scale": scale,
    }
    return cropped, metadata


def preprocess_image(image_bgr, height, width):
    """将 OpenCV BGR 图片转换为归一化 NCHW float32 输入，并返回裁剪元信息。"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb, metadata = resize_and_crop_image(image_rgb, height, width)

    image_rgb = (image_rgb - MEAN) / STD
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    input_tensor = np.expand_dims(np.ascontiguousarray(image_chw), axis=0).astype(np.float32)
    return input_tensor, metadata


def restore_depth_to_original(depth, metadata):
    """根据预处理裁剪元信息，将 RKNN 输出深度图还原到原图尺寸。"""
    target_height = metadata["target_height"]
    target_width = metadata["target_width"]
    if depth.shape != (target_height, target_width):
        depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    top = metadata["crop_y"]
    left = metadata["crop_x"]
    bottom = metadata["resized_height"] - top - target_height
    right = metadata["resized_width"] - left - target_width

    # 被裁掉的区域没有真实推理结果，只能用边缘深度补齐，避免还原图出现空洞。
    restored_resized = cv2.copyMakeBorder(
        depth,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(
        restored_resized,
        (metadata["original_width"], metadata["original_height"]),
        interpolation=cv2.INTER_CUBIC,
    )


def run_inference(rknn, image_bgr, height, width, data_format):
    """执行 RKNN 推理，并将深度图缩放回原图大小。"""
    input_tensor, metadata = preprocess_image(image_bgr, height, width)
    if data_format == "none":
        outputs = rknn.inference(inputs=[input_tensor])
    else:
        outputs = rknn.inference(inputs=[input_tensor], data_format=data_format)
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("rknn.inference returned no outputs")

    depth = np.squeeze(outputs[0]).astype(np.float32)
    return restore_depth_to_original(depth, metadata)


def print_depth_stats(depth, image_path):
    """打印深度图统计信息，用于和本机 PyTorch 的 .npy 输出做数值对比。"""
    percentiles = np.percentile(depth, [1, 50, 99])
    print("depth_stats:")
    print(f"  image: {image_path}")
    print(f"  shape: {depth.shape}")
    print(f"  min: {float(np.min(depth)):.6f}")
    print(f"  max: {float(np.max(depth)):.6f}")
    print(f"  mean: {float(np.mean(depth)):.6f}")
    print(f"  p01: {float(percentiles[0]):.6f}")
    print(f"  p50: {float(percentiles[1]):.6f}")
    print(f"  p99: {float(percentiles[2]):.6f}")


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
            depth = run_inference(rknn, image_bgr, args.height, args.width, args.data_format)
            if args.print_stats:
                print_depth_stats(depth, image_path)
            save_result(depth, image_path, args.outdir, args.save_numpy, args.grayscale)
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
