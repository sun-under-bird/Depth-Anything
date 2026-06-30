import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
WINDOW_NAME = "Depth Anything V2 preview"
COMBINED_SPLIT_WIDTH = 16


def parse_args():
    """解析视频 FPS 测试需要的命令行参数。"""
    parser = argparse.ArgumentParser(description="测试 Depth Anything V2 metric 模型的视频流 FPS。")
    parser.add_argument("--source", required=True, help="视频文件路径或摄像头编号，例如 0。")
    parser.add_argument("--encoder", default="vits", choices=MODEL_CONFIGS.keys(), help="模型编码器类型。")
    parser.add_argument("--load-from", required=True, help="metric depth 权重路径。")
    parser.add_argument("--max-depth", type=float, default=20.0, help="metric 模型最大深度，单位米。")
    parser.add_argument("--input-size", type=int, default=322, help="模型预处理输入尺寸。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备。")
    parser.add_argument("--resize-width", type=int, default=0, help="推理前缩放后的视频帧宽度。")
    parser.add_argument("--resize-height", type=int, default=0, help="推理前缩放后的视频帧高度。")
    parser.add_argument("--max-frames", type=int, default=60, help="处理总帧数，0 表示不限制。")
    parser.add_argument("--warmup-frames", type=int, default=5, help="不计入 FPS 统计的预热帧数。")
    parser.add_argument("--camera-width", type=int, default=0, help="请求的摄像头采集宽度。")
    parser.add_argument("--camera-height", type=int, default=0, help="请求的摄像头采集高度。")
    parser.add_argument("--camera-fps", type=float, default=0.0, help="请求的摄像头采集 FPS。")
    parser.add_argument("--show", action="store_true", help="打开实时预览窗口。")
    parser.add_argument("--save-preview", default=None, help="保存预览视频的路径。")
    parser.add_argument("--preview-mode", default="combined", choices=["combined", "depth"], help="预览布局。")
    parser.add_argument("--preview-normalize", default="fixed", choices=["fixed", "frame"], help="深度颜色映射方式。")
    parser.add_argument("--depth-hud", action="store_true", help="在预览图上绘制帧号和 FPS 信息。")
    return parser.parse_args()


def select_device(requested_device):
    """根据用户参数选择实际使用的 PyTorch 推理设备。"""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but it is not available.")
    return torch.device(requested_device)


def synchronize_device(device):
    """在读取耗时前同步异步加速设备，保证计时准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_model(args, device):
    """加载 Depth Anything V2 metric 模型和权重。"""
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], "max_depth": args.max_depth})
    state_dict = torch.load(args.load_from, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def make_transform(input_size):
    """创建与 metric_depth 单图推理一致的图像预处理流程。"""
    return Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )


def image_to_tensor(raw_image, transform, device):
    """把一帧 OpenCV BGR 图像转换成模型需要的归一化张量。"""
    height, width = raw_image.shape[:2]
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    return image, (height, width)


@torch.inference_mode()
def infer_depth(model, frame, transform, device):
    """执行单帧深度推理，并返回 CPU 上的 numpy 深度图。"""
    image, (height, width) = image_to_tensor(frame, transform, device)

    synchronize_device(device)
    depth = model(image)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    synchronize_device(device)
    return depth.cpu().numpy()


def parse_video_source(source):
    """把数字形式的摄像头编号字符串转换为整数。"""
    if source.isdigit():
        return int(source)
    return source


def open_capture(args):
    """打开视频文件或摄像头视频流。"""
    capture = cv2.VideoCapture(parse_video_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    if args.source.isdigit():
        if args.camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    return capture


def resize_frame(frame, resize_width, resize_height):
    """在指定宽高时缩放视频帧。"""
    if resize_width > 0 and resize_height > 0:
        return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return frame


def depth_to_color(depth, max_depth, normalize_mode):
    """把深度图转换成彩色预览图。"""
    if normalize_mode == "fixed":
        depth_u8 = np.clip(depth / max_depth, 0.0, 1.0)
        depth_u8 = (depth_u8 * 255.0).astype(np.uint8)
    else:
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min < 1e-6:
            depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        else:
            depth_u8 = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)

    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def draw_hud(preview, frame_index, measured_frames, infer_times, total_times, device):
    """在预览画面上绘制简洁的运行状态信息。"""
    infer_fps = measured_frames / sum(infer_times) if infer_times else 0.0
    total_fps = measured_frames / sum(total_times) if total_times else 0.0
    lines = [
        f"frame: {frame_index}",
        f"device: {device.type}",
        f"end-to-end fps: {total_fps:.2f}",
        f"inference fps: {infer_fps:.2f}",
    ]

    # 先绘制深色背景，避免状态文字在亮色画面上看不清。
    cv2.rectangle(preview, (8, 8), (270, 112), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(preview, line, (18, 34 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return preview


def create_depth_inspector():
    """创建鼠标深度取点需要的可变状态。"""
    return {
        "depth": None,
        "frame_shape": None,
        "preview_mode": "combined",
        "hover": None,
        "selected": None,
    }


def update_depth_inspector_frame(inspector, depth, frame_shape, preview_mode):
    """更新鼠标回调使用的最新深度图和画面信息。"""
    if inspector is None:
        return

    # OpenCV 鼠标回调会异步读取这些状态，所以统一放在字典里更新。
    inspector["depth"] = depth
    inspector["frame_shape"] = frame_shape
    inspector["preview_mode"] = preview_mode


def preview_to_depth_point(x, y, frame_shape, preview_mode):
    """把预览窗口坐标映射到深度图坐标。"""
    frame_height, frame_width = frame_shape[:2]
    if y < 0 or y >= frame_height:
        return None

    if preview_mode == "depth":
        if 0 <= x < frame_width:
            return x, y, "depth"
        return None

    if 0 <= x < frame_width:
        return x, y, "image"

    depth_start_x = frame_width + COMBINED_SPLIT_WIDTH
    depth_end_x = depth_start_x + frame_width
    if depth_start_x <= x < depth_end_x:
        return x - depth_start_x, y, "depth"
    return None


def update_depth_inspector_point(event, x, y, flags, inspector):
    """根据 OpenCV 鼠标事件更新悬停点或选中点的深度值。"""
    depth = inspector.get("depth")
    frame_shape = inspector.get("frame_shape")
    if depth is None or frame_shape is None:
        return

    mapped_point = preview_to_depth_point(x, y, frame_shape, inspector.get("preview_mode", "combined"))
    if mapped_point is None:
        inspector["hover"] = None
        return

    depth_x, depth_y, region = mapped_point
    depth_value = float(depth[depth_y, depth_x])
    inspector["hover"] = (depth_x, depth_y, depth_value, region)

    if event == cv2.EVENT_LBUTTONDOWN:
        inspector["selected"] = (depth_x, depth_y, depth_value, region)
        print(f"depth_probe: x={depth_x}, y={depth_y}, depth_m={depth_value:.3f}, region={region}")


def depth_point_to_preview_points(depth_x, depth_y, frame_shape, preview_mode):
    """把一个深度图坐标映射到预览图上的标记位置。"""
    frame_width = frame_shape[1]
    if preview_mode == "depth":
        return [(depth_x, depth_y)]

    # combined 预览模式下，同一个点同时存在于左侧原图和右侧深度图上。
    return [(depth_x, depth_y), (depth_x + frame_width + COMBINED_SPLIT_WIDTH, depth_y)]


def draw_depth_probe(preview, inspector):
    """在预览图上绘制当前鼠标取点的深度信息。"""
    if inspector is None:
        return preview

    point = inspector.get("selected") or inspector.get("hover")
    frame_shape = inspector.get("frame_shape")
    if point is None or frame_shape is None:
        return preview

    depth_x, depth_y, depth_value, region = point
    marker_points = depth_point_to_preview_points(depth_x, depth_y, frame_shape, inspector.get("preview_mode", "combined"))
    for marker_x, marker_y in marker_points:
        cv2.drawMarker(preview, (marker_x, marker_y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(preview, (marker_x, marker_y), 5, (0, 0, 0), 2)

    label = f"x={depth_x} y={depth_y} depth={depth_value:.3f}m"
    label_x = min(max(marker_points[0][0] + 12, 8), max(preview.shape[1] - 340, 8))
    label_y = min(max(marker_points[0][1] - 12, 28), max(preview.shape[0] - 12, 28))

    # 给标签绘制纯色背景，避免文字被复杂视频内容淹没。
    cv2.rectangle(preview, (label_x - 6, label_y - 22), (label_x + 330, label_y + 8), (0, 0, 0), thickness=-1)
    cv2.putText(preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    if region == "depth":
        cv2.putText(preview, "depth view", (label_x, label_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return preview


def build_preview(frame, depth, args, frame_index, measured_frames, infer_times, total_times, device, inspector=None):
    """创建用于显示或保存的视频预览帧。"""
    depth_color = depth_to_color(depth, args.max_depth, args.preview_normalize)
    if args.preview_mode == "depth":
        preview = depth_color
    else:
        split = np.full((frame.shape[0], COMBINED_SPLIT_WIDTH, 3), 255, dtype=np.uint8)
        preview = cv2.hconcat([frame, split, depth_color])

    if args.depth_hud:
        preview = draw_hud(preview, frame_index, measured_frames, infer_times, total_times, device)
    preview = draw_depth_probe(preview, inspector)
    return preview


def create_preview_writer(save_path, preview_frame, capture, args):
    """创建与预览帧尺寸匹配的视频写入器。"""
    if not save_path:
        return None

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = args.camera_fps if args.camera_fps > 0 else 30.0

    height, width = preview_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open preview writer: {save_path}")
    return writer


def percentile_ms(values, percentile):
    """根据秒级耗时列表计算毫秒级分位数。"""
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64) * 1000.0, percentile))


def print_environment(args, capture, device):
    """打印测试环境和输入视频信息。"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("benchmark_config:")
    print(f"  source: {args.source}")
    print(f"  encoder: {args.encoder}")
    print(f"  checkpoint: {args.load_from}")
    print(f"  input_size: {args.input_size}")
    print(f"  device: {device.type}")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  capture_resolution: {width}x{height}")
    print(f"  capture_fps: {fps:.2f}")
    if args.resize_width > 0 and args.resize_height > 0:
        print(f"  resize: {args.resize_width}x{args.resize_height}")
    else:
        print("  resize: original")

    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")


def print_results(processed_frames, warmup_frames, read_times, infer_times, total_times):
    """打印最终 FPS 和延迟统计结果。"""
    measured_frames = len(infer_times)
    read_sum = sum(read_times)
    infer_sum = sum(infer_times)
    total_sum = sum(total_times)

    print("benchmark_result:")
    print(f"  processed_frames: {processed_frames}")
    print(f"  warmup_frames: {min(warmup_frames, processed_frames)}")
    print(f"  measured_frames: {measured_frames}")
    print(f"  end_to_end_fps: {measured_frames / total_sum if total_sum > 0 else 0.0:.2f}")
    print(f"  inference_fps: {measured_frames / infer_sum if infer_sum > 0 else 0.0:.2f}")
    print(f"  read_fps: {measured_frames / read_sum if read_sum > 0 else 0.0:.2f}")
    print(f"  avg_total_ms: {(total_sum / measured_frames * 1000.0) if measured_frames else 0.0:.2f}")
    print(f"  avg_infer_ms: {(infer_sum / measured_frames * 1000.0) if measured_frames else 0.0:.2f}")
    print(f"  p50_infer_ms: {percentile_ms(infer_times, 50):.2f}")
    print(f"  p90_infer_ms: {percentile_ms(infer_times, 90):.2f}")


def run_benchmark(args):
    """执行视频文件或摄像头的 FPS 测试。"""
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(args, device)
    transform = make_transform(args.input_size)
    capture = open_capture(args)
    print_environment(args, capture, device)

    processed_frames = 0
    read_times = []
    infer_times = []
    total_times = []
    writer = None
    inspector = create_depth_inspector() if args.show else None
    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, update_depth_inspector_point, inspector)

    try:
        while args.max_frames <= 0 or processed_frames < args.max_frames:
            total_start = time.perf_counter()
            read_start = time.perf_counter()
            ok, frame = capture.read()
            read_time = time.perf_counter() - read_start
            if not ok:
                break

            frame = resize_frame(frame, args.resize_width, args.resize_height)
            infer_start = time.perf_counter()
            depth = infer_depth(model, frame, transform, device)
            infer_time = time.perf_counter() - infer_start

            processed_frames += 1
            measured = processed_frames > args.warmup_frames
            if measured:
                read_times.append(read_time)
                infer_times.append(infer_time)

            if args.show or args.save_preview:
                update_depth_inspector_frame(inspector, depth, frame.shape, args.preview_mode)
                preview = build_preview(
                    frame,
                    depth,
                    args,
                    processed_frames,
                    len(infer_times),
                    infer_times,
                    total_times,
                    device,
                    inspector,
                )
                if args.save_preview:
                    if writer is None:
                        writer = create_preview_writer(args.save_preview, preview, capture, args)
                    writer.write(preview)
                if args.show:
                    cv2.imshow(WINDOW_NAME, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        if measured:
                            total_times.append(time.perf_counter() - total_start)
                        break

            if measured:
                total_times.append(time.perf_counter() - total_start)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print_results(processed_frames, args.warmup_frames, read_times, infer_times, total_times)


def main():
    """程序入口。"""
    args = parse_args()
    if not os.path.exists(args.load_from):
        raise FileNotFoundError(f"Checkpoint not found: {args.load_from}")
    run_benchmark(args)


if __name__ == "__main__":
    main()
