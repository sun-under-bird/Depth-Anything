import argparse
from datetime import datetime
from pathlib import Path
import time
import webbrowser

import cv2
import numpy as np
import torch

from benchmark_video_fps import (
    COMBINED_SPLIT_WIDTH,
    build_model,
    create_depth_inspector,
    depth_to_color,
    draw_depth_probe,
    infer_depth,
    make_transform,
    open_capture,
    resize_frame,
    select_device,
    update_depth_inspector_frame,
    update_depth_inspector_point,
)
from depth_to_pointcloud import (
    create_open3d_pointcloud,
    depth_to_points,
    get_camera_intrinsics,
    o3d,
    prepare_pointcloud_arrays,
    save_html_viewer,
    save_pointcloud_ply,
)


WINDOW_NAME = "Depth Anything V2 camera point cloud"


def parse_args():
    """解析实时摄像头转点云需要的命令行参数。"""
    parser = argparse.ArgumentParser(description="使用摄像头实时预览深度，并把当前帧保存为点云。")
    parser.add_argument("--source", default="0", help="摄像头编号或视频源，默认 0。")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"], help="模型编码器类型。")
    parser.add_argument("--load-from", required=True, help="Depth Anything V2 metric 权重路径。")
    parser.add_argument("--max-depth", default=20.0, type=float, help="metric 模型最大深度，单位米。")
    parser.add_argument("--input-size", default=322, type=int, help="模型推理输入尺寸。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备。")
    parser.add_argument("--camera-width", default=640, type=int, help="请求的摄像头采集宽度。")
    parser.add_argument("--camera-height", default=480, type=int, help="请求的摄像头采集高度。")
    parser.add_argument("--camera-fps", default=30.0, type=float, help="请求的摄像头采集 FPS。")
    parser.add_argument("--resize-width", default=0, type=int, help="推理和点云保存前缩放到的宽度，0 表示不缩放。")
    parser.add_argument("--resize-height", default=0, type=int, help="推理和点云保存前缩放到的高度，0 表示不缩放。")
    parser.add_argument("--max-frames", default=0, type=int, help="最多处理多少帧，0 表示一直运行。")
    parser.add_argument("--outdir", default="metric_depth/outputs_camera_pointcloud", help="点云输出目录。")
    parser.add_argument("--focal-length-x", default=470.4, type=float, help="相机 x 方向焦距，单位像素。")
    parser.add_argument("--focal-length-y", default=470.4, type=float, help="相机 y 方向焦距，单位像素。")
    parser.add_argument("--principal-point-x", default=None, type=float, help="相机主点 x 坐标，默认使用图像中心。")
    parser.add_argument("--principal-point-y", default=None, type=float, help="相机主点 y 坐标，默认使用图像中心。")
    parser.add_argument("--voxel-size", default=0.02, type=float, help="体素下采样尺寸，0 表示不下采样。")
    parser.add_argument("--max-points", default=300000, type=int, help="PLY 最多保存多少个点，0 表示不限。")
    parser.add_argument("--save-html", action="store_true", help="保存点云时同时生成浏览器 HTML 查看器。")
    parser.add_argument("--open-html", action="store_true", help="保存 HTML 后自动用浏览器打开。")
    parser.add_argument("--html-max-points", default=120000, type=int, help="HTML 查看器最多写入多少个点。")
    parser.add_argument("--open-open3d", action="store_true", help="保存点云后自动用 Open3D 打开。")
    parser.add_argument("--open3d-point-size", default=2.0, type=float, help="Open3D 窗口中的点大小。")
    parser.add_argument("--open3d-axis-size", default=0.5, type=float, help="Open3D 坐标轴尺寸，0 表示不显示坐标轴。")
    parser.add_argument("--save-depth-numpy", action="store_true", help="保存当前帧对应的米制深度 .npy。")
    parser.add_argument("--save-rgb", action="store_true", help="保存当前帧 RGB 图片，便于和点云结果对应。")
    parser.add_argument("--preview-mode", default="combined", choices=["combined", "depth"], help="预览模式。")
    parser.add_argument("--preview-normalize", default="fixed", choices=["fixed", "frame"], help="深度颜色映射方式。")
    return parser.parse_args()


def print_environment(args, capture, device):
    """打印摄像头、模型和快捷键配置，方便确认当前运行环境。"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("camera_pointcloud_config:")
    print(f"  source: {args.source}")
    print(f"  encoder: {args.encoder}")
    print(f"  checkpoint: {args.load_from}")
    print(f"  input_size: {args.input_size}")
    print(f"  device: {device.type}")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  capture_resolution: {width}x{height}")
    print(f"  capture_fps: {fps:.2f}")
    print(f"  focal_length: fx={args.focal_length_x:.2f}, fy={args.focal_length_y:.2f}")
    if args.resize_width > 0 and args.resize_height > 0:
        print(f"  resize: {args.resize_width}x{args.resize_height}")
    else:
        print("  resize: original")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print("controls:")
    print("  s / p: save current frame as point cloud")
    print("  o: save current frame and open HTML viewer")
    print("  v: save current frame and open Open3D viewer")
    print("  q / Esc: quit")


def make_output_stem(frame_index):
    """根据时间和帧号生成稳定且不容易重名的输出文件名前缀。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"camera_frame_{timestamp}_{frame_index:06d}"


def frame_depth_to_pointcloud(frame, depth, args):
    """把当前摄像头帧和深度图反投影成彩色点云数组。"""
    height, width = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fx, fy, cx, cy = get_camera_intrinsics(width, height, args)

    # 深度图和 RGB 图必须同尺寸，颜色才能按同一个像素索引取出来。
    points, valid_mask = depth_to_points(depth, fx, fy, cx, cy)
    colors = image_rgb.reshape(-1, 3)[valid_mask] / 255.0
    points, colors = prepare_pointcloud_arrays(points, colors, args)
    return points, colors, (fx, fy, cx, cy)


def open_open3d_viewer(points, colors, window_name, args):
    """用 Open3D 打开彩色点云窗口，关闭窗口后会回到摄像头预览。"""
    if o3d is None:
        raise RuntimeError("当前环境未安装 open3d，无法打开 Open3D 点云窗口。")

    pcd = create_open3d_pointcloud(points, colors)
    geometries = [pcd]
    if args.open3d_axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.open3d_axis_size)
        geometries.append(axis)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name=window_name, width=1280, height=720)
    for geometry in geometries:
        viewer.add_geometry(geometry)

    # 点云默认点太小时不容易看清，这里按参数调整渲染效果。
    render_option = viewer.get_render_option()
    render_option.point_size = max(1.0, args.open3d_point_size)
    render_option.background_color = np.array([0.02, 0.02, 0.02])
    viewer.run()
    viewer.destroy_window()


def save_current_pointcloud(frame, depth, args, frame_index, force_open_html=False, force_open_open3d=False):
    """保存当前帧点云，并按参数决定是否打开 HTML 或 Open3D 查看器。"""
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, colors, intrinsics = frame_depth_to_pointcloud(frame, depth, args)
    if len(points) == 0:
        print("pointcloud_save_skipped: no valid depth points")
        return "save skipped: no valid points"

    stem = make_output_stem(frame_index)
    ply_path = output_dir / f"{stem}.ply"
    save_pointcloud_ply(points, colors, ply_path)

    html_path = None
    should_save_html = args.save_html or args.open_html or force_open_html
    should_open_html = args.open_html or force_open_html
    if should_save_html:
        html_path = output_dir / f"{stem}_viewer.html"
        save_html_viewer(points, colors, html_path, args.html_max_points)
        if should_open_html:
            webbrowser.open(html_path.resolve().as_uri())

    should_open_open3d = args.open_open3d or force_open_open3d
    open3d_error = ""
    if should_open_open3d:
        try:
            open_open3d_viewer(points, colors, f"Open3D PointCloud: {stem}", args)
        except Exception as exc:
            open3d_error = repr(exc)
            print(f"open3d_viewer_failed: {open3d_error}")

    if args.save_depth_numpy:
        np.save(output_dir / f"{stem}_raw_depth_meter.npy", depth)

    if args.save_rgb:
        cv2.imwrite(str(output_dir / f"{stem}_rgb.png"), frame)

    fx, fy, cx, cy = intrinsics
    print("pointcloud_saved:")
    print(f"  ply: {ply_path}")
    if html_path is not None:
        print(f"  html: {html_path}")
    print(f"  points: {len(points)}")
    print(f"  resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    if open3d_error:
        return f"saved {ply_path.name}, open3d failed"
    if should_open_open3d:
        return f"saved+open3d {ply_path.name}, points={len(points)}"
    return f"saved {ply_path.name}, points={len(points)}"


def draw_text_panel(preview, lines, origin=(10, 10)):
    """在预览图左上角绘制半透明信息面板。"""
    x, y = origin
    line_height = 22
    width = 430
    height = 18 + line_height * len(lines)

    # OpenCV 直接画不透明背景，能保证文字在复杂画面上可读。
    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (x + 10, y + 26 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    return preview


def build_preview(frame, depth, args, frame_index, device, fps, last_message, inspector):
    """生成实时显示的 RGB/深度预览画面。"""
    depth_color = depth_to_color(depth, args.max_depth, args.preview_normalize)
    if args.preview_mode == "depth":
        preview = depth_color.copy()
    else:
        split = np.full((frame.shape[0], COMBINED_SPLIT_WIDTH, 3), 255, dtype=np.uint8)
        preview = cv2.hconcat([frame, split, depth_color])

    lines = [
        f"frame: {frame_index}",
        f"device: {device.type}",
        f"fps: {fps:.2f}",
        "S/P save PLY, V Open3D, O HTML, Q/Esc quit",
    ]
    if last_message:
        lines.append(last_message[:68])

    preview = draw_text_panel(preview, lines)
    return draw_depth_probe(preview, inspector)


def handle_key(key, frame, depth, args, frame_index):
    """处理预览窗口快捷键，并返回是否需要退出以及最新状态消息。"""
    if key in (27, ord("q")):
        return True, ""

    if key in (ord("s"), ord("p")):
        message = save_current_pointcloud(frame, depth, args, frame_index)
        return False, message

    if key == ord("o"):
        message = save_current_pointcloud(frame, depth, args, frame_index, force_open_html=True)
        return False, message

    if key == ord("v"):
        message = save_current_pointcloud(frame, depth, args, frame_index, force_open_open3d=True)
        return False, message

    return False, ""


def run_camera_pointcloud(args):
    """执行摄像头实时深度预览，并支持按键保存当前帧点云。"""
    if not Path(args.load_from).exists():
        raise FileNotFoundError(f"权重文件不存在: {args.load_from}")

    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(args, device)
    transform = make_transform(args.input_size)
    capture = open_capture(args)
    print_environment(args, capture, device)

    inspector = create_depth_inspector()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, update_depth_inspector_point, inspector)

    frame_index = 0
    last_message = ""
    last_tick = time.perf_counter()
    fps = 0.0

    try:
        while args.max_frames <= 0 or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                print(f"camera_read_failed: source={args.source}")
                break

            frame = resize_frame(frame, args.resize_width, args.resize_height)
            depth = infer_depth(model, frame, transform, device)
            frame_index += 1

            now = time.perf_counter()
            elapsed = now - last_tick
            fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
            last_tick = now

            update_depth_inspector_frame(inspector, depth, frame.shape, args.preview_mode)
            preview = build_preview(frame, depth, args, frame_index, device, fps, last_message, inspector)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            should_quit, message = handle_key(key, frame, depth, args, frame_index)
            if message:
                last_message = message
            if should_quit:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    """程序入口。"""
    args = parse_args()
    run_camera_pointcloud(args)


if __name__ == "__main__":
    main()
