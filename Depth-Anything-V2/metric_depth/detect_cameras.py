import argparse
import json
import platform
import subprocess
import sys

import cv2


BACKEND_IDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def parse_args():
    """解析摄像头编号扫描和单个编号探测所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="检测 OpenCV 可用的摄像头编号。")
    parser.add_argument("--start-index", default=0, type=int, help="开始扫描的摄像头编号。")
    parser.add_argument("--max-index", default=6, type=int, help="结束扫描的摄像头编号，包含这个编号。")
    parser.add_argument("--timeout", default=2.5, type=float, help="每个编号的最大探测秒数。")
    parser.add_argument("--backend", default="auto", choices=["auto", "any", "dshow", "msmf"], help="OpenCV 摄像头后端。")
    parser.add_argument("--width", default=640, type=int, help="探测时请求的采集宽度。")
    parser.add_argument("--height", default=480, type=int, help="探测时请求的采集高度。")
    parser.add_argument("--fps", default=30.0, type=float, help="探测时请求的采集 FPS。")
    parser.add_argument("--read-frames", default=3, type=int, help="每个编号最多读取多少帧确认画面可用。")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出完整扫描结果。")
    parser.add_argument("--probe-index", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-backend", default=None, choices=["any", "dshow", "msmf"], help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_backend_name(backend_name):
    """把 auto 后端转换成当前系统上更合适的 OpenCV 后端名称。"""
    if backend_name != "auto":
        return backend_name
    if platform.system().lower() == "windows":
        return "dshow"
    return "any"


def get_capture_backend_name(capture):
    """读取 OpenCV 实际使用的后端名称，失败时返回 unknown。"""
    try:
        return capture.getBackendName()
    except cv2.error:
        return "unknown"


def probe_camera(index, backend_name, width, height, fps, read_frames):
    """在子进程中探测单个摄像头编号，避免坏编号卡住整个扫描。"""
    backend_id = BACKEND_IDS[backend_name]
    capture = cv2.VideoCapture(index, backend_id)
    result = {
        "index": index,
        "backend": backend_name,
        "opened": False,
        "read": False,
        "frame_shape": None,
        "reported_width": 0,
        "reported_height": 0,
        "reported_fps": 0.0,
        "actual_backend": "unknown",
        "error": "",
    }

    try:
        result["opened"] = bool(capture.isOpened())
        if not result["opened"]:
            return result

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)

        result["actual_backend"] = get_capture_backend_name(capture)
        result["reported_width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["reported_height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["reported_fps"] = float(capture.get(cv2.CAP_PROP_FPS))

        # 有些摄像头第一帧可能为空，多读几帧能减少误判。
        for _ in range(max(1, read_frames)):
            ok, frame = capture.read()
            if ok and frame is not None:
                result["read"] = True
                result["frame_shape"] = list(frame.shape)
                break
        return result
    except Exception as exc:
        result["error"] = repr(exc)
        return result
    finally:
        capture.release()


def run_probe_child(args):
    """执行单个编号探测子进程，并把结果以一行 JSON 打印给父进程。"""
    backend_name = args.probe_backend or resolve_backend_name(args.backend)
    result = probe_camera(args.probe_index, backend_name, args.width, args.height, args.fps, args.read_frames)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def parse_probe_stdout(stdout):
    """从子进程输出中解析最后一行 JSON 结果。"""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_probe_subprocess(index, backend_name, args):
    """用独立 Python 子进程探测一个摄像头编号，并处理超时和异常。"""
    command = [
        sys.executable,
        __file__,
        "--probe-index",
        str(index),
        "--probe-backend",
        backend_name,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--read-frames",
        str(args.read_frames),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "index": index,
            "backend": backend_name,
            "opened": False,
            "read": False,
            "status": "timeout",
            "error": f"timeout after {args.timeout:.1f}s",
        }

    result = parse_probe_stdout(completed.stdout)
    if result is None:
        return {
            "index": index,
            "backend": backend_name,
            "opened": False,
            "read": False,
            "status": "error",
            "error": (completed.stderr or completed.stdout or "no probe output").strip(),
        }

    result["status"] = "ok" if result.get("opened") and result.get("read") else "unavailable"
    if completed.stderr.strip() and not result.get("error"):
        result["error"] = completed.stderr.strip()
    return result


def scan_cameras(args):
    """按编号顺序扫描摄像头，并返回每个编号的探测结果。"""
    backend_name = resolve_backend_name(args.backend)
    results = []
    for index in range(args.start_index, args.max_index + 1):
        result = run_probe_subprocess(index, backend_name, args)
        results.append(result)
        print_probe_result(result)
    return {
        "backend": backend_name,
        "start_index": args.start_index,
        "max_index": args.max_index,
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "timeout": args.timeout,
        "results": results,
    }


def format_frame_shape(frame_shape):
    """把 OpenCV 的 HWC 形状转换成更直观的 宽x高x通道 文本。"""
    if not frame_shape or len(frame_shape) < 2:
        return "none"
    height, width = frame_shape[:2]
    channels = frame_shape[2] if len(frame_shape) > 2 else 1
    return f"{width}x{height}x{channels}"


def print_probe_result(result):
    """打印单个摄像头编号的探测结果。"""
    index = result.get("index")
    backend = result.get("backend")
    status = result.get("status", "unknown")
    if status == "ok":
        frame_text = format_frame_shape(result.get("frame_shape"))
        reported_width = result.get("reported_width", 0)
        reported_height = result.get("reported_height", 0)
        reported_fps = result.get("reported_fps", 0.0)
        actual_backend = result.get("actual_backend", "unknown")
        print(
            f"index {index}: OK, backend={backend}/{actual_backend}, "
            f"frame={frame_text}, reported={reported_width}x{reported_height}@{reported_fps:.2f}"
        )
        return

    error = result.get("error", "")
    if status == "timeout":
        print(f"index {index}: TIMEOUT, backend={backend}, {error}")
    else:
        print(f"index {index}: unavailable, backend={backend}")


def print_summary(scan_result):
    """打印可直接用于 --source 的摄像头编号汇总。"""
    available = [item["index"] for item in scan_result["results"] if item.get("status") == "ok"]
    print("camera_scan_summary:")
    if available:
        print("  available_indices: " + ", ".join(str(index) for index in available))
        print(f"  second_available_index: {available[1] if len(available) > 1 else 'none'}")
    else:
        print("  available_indices: none")
        print("  second_available_index: none")


def main():
    """程序入口，根据参数选择父进程扫描模式或子进程探测模式。"""
    args = parse_args()
    if args.probe_index is not None:
        return run_probe_child(args)

    backend_name = resolve_backend_name(args.backend)
    print("camera_scan_config:")
    print(f"  index_range: {args.start_index}..{args.max_index}")
    print(f"  backend: {backend_name}")
    print(f"  request: {args.width}x{args.height}@{args.fps:.2f}")
    print(f"  timeout_per_index: {args.timeout:.1f}s")
    scan_result = scan_cameras(args)
    print_summary(scan_result)
    if args.json:
        print(json.dumps(scan_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
