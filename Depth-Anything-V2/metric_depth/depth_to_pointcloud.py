import argparse
import base64
import glob
import os
from pathlib import Path
import webbrowser

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


try:
    import open3d as o3d
except ImportError:
    o3d = None


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args():
    """解析图片转点云所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 Depth Anything V2 metric 深度图生成彩色点云。")
    parser.add_argument("--encoder", default="vits", choices=MODEL_CONFIGS.keys(), help="模型编码器类型。")
    parser.add_argument("--load-from", required=True, help="metric depth 权重路径。")
    parser.add_argument("--max-depth", default=20.0, type=float, help="metric 模型最大深度，单位米。")
    parser.add_argument("--img-path", required=True, help="输入图片、图片目录或 txt 图片列表。")
    parser.add_argument("--outdir", default="metric_depth/outputs_pointcloud", help="点云输出目录。")
    parser.add_argument("--input-size", default=518, type=int, help="模型推理输入尺寸。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备。")
    parser.add_argument("--focal-length-x", default=470.4, type=float, help="相机 x 方向焦距，单位像素。")
    parser.add_argument("--focal-length-y", default=470.4, type=float, help="相机 y 方向焦距，单位像素。")
    parser.add_argument("--principal-point-x", default=None, type=float, help="相机主点 x 坐标，默认使用图像中心。")
    parser.add_argument("--principal-point-y", default=None, type=float, help="相机主点 y 坐标，默认使用图像中心。")
    parser.add_argument("--voxel-size", default=0.0, type=float, help="体素下采样尺寸，0 表示不下采样。")
    parser.add_argument("--max-points", default=0, type=int, help="最多保留的点数，0 表示不限制。")
    parser.add_argument("--save-depth-numpy", action="store_true", help="同时保存原始米制深度 .npy。")
    parser.add_argument("--save-html", action="store_true", help="同时保存浏览器可打开的 HTML 点云预览。")
    parser.add_argument("--open-html", action="store_true", help="生成 HTML 后自动用浏览器打开。")
    parser.add_argument("--html-max-points", default=120000, type=int, help="HTML 预览最多写入的点数。")
    parser.add_argument("--visualize", action="store_true", help="生成后用 Open3D 打开点云窗口。")
    return parser.parse_args()


def select_device(requested_device):
    """根据用户参数选择实际推理设备。"""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 cuda，但当前 PyTorch 不可用 cuda。")
    if requested_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("请求使用 mps，但当前 PyTorch 不可用 mps。")
    return torch.device(requested_device)


def build_model(args, device):
    """加载 Depth Anything V2 metric 模型和权重。"""
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], "max_depth": args.max_depth})
    state_dict = torch.load(args.load_from, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def collect_image_paths(img_path):
    """从图片文件、目录或 txt 列表中收集输入图片路径。"""
    path = Path(img_path)
    if path.is_file() and path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_file():
        return [str(path)]

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        item for item in glob.glob(os.path.join(img_path, "**/*"), recursive=True)
        if Path(item).suffix.lower() in image_suffixes
    )


def get_camera_intrinsics(width, height, args):
    """根据图片尺寸和参数生成相机内参。"""
    cx = args.principal_point_x if args.principal_point_x is not None else width / 2.0
    cy = args.principal_point_y if args.principal_point_y is not None else height / 2.0
    return args.focal_length_x, args.focal_length_y, cx, cy


def depth_to_points(depth, fx, fy, cx, cy):
    """将米制深度图反投影为相机坐标系下的三维点。"""
    height, width = depth.shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    valid_mask = np.isfinite(depth) & (depth > 0)
    z = depth[valid_mask]
    x = (x_grid[valid_mask] - cx) * z / fx
    y = (y_grid[valid_mask] - cy) * z / fy
    return np.stack((x, y, z), axis=-1), valid_mask.reshape(-1)


def limit_points(points, colors, max_points):
    """按最大点数限制随机抽样点云。"""
    if max_points <= 0 or len(points) <= max_points:
        return points, colors

    indices = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def voxel_downsample(points, colors, voxel_size):
    """使用简单体素网格对点云下采样。"""
    if voxel_size <= 0 or len(points) == 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices], colors[unique_indices]


def prepare_pointcloud_arrays(points, colors, args):
    """根据参数处理点云数组，返回可保存的点和颜色。"""
    points, colors = voxel_downsample(points, colors, args.voxel_size)
    points, colors = limit_points(points, colors, args.max_points)
    return points.astype(np.float32), colors.astype(np.float32)


def create_open3d_pointcloud(points, colors):
    """根据三维点和颜色创建 Open3D 点云对象。"""
    if o3d is None:
        raise RuntimeError("当前环境未安装 open3d，无法打开 Open3D 可视化窗口。")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_pointcloud_ply(points, colors, output_path):
    """使用二进制 PLY 格式保存彩色点云。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    ply_data = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    ply_data["x"] = points[:, 0]
    ply_data["y"] = points[:, 1]
    ply_data["z"] = points[:, 2]
    ply_data["red"] = colors_u8[:, 0]
    ply_data["green"] = colors_u8[:, 1]
    ply_data["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(output_path, "wb") as file:
        file.write(header.encode("ascii"))
        ply_data.tofile(file)


def encode_array_base64(array):
    """将 numpy 数组编码为 HTML 中可嵌入的 base64 字符串。"""
    return base64.b64encode(array.tobytes()).decode("ascii")


def save_html_viewer(points, colors, output_path, max_points):
    """保存一个不依赖外部库的浏览器 WebGL 点云查看器。"""
    preview_points, preview_colors = limit_points(points, colors, max_points)
    if len(preview_points) == 0:
        raise RuntimeError("没有可写入 HTML 预览的有效点。")

    preview_points = preview_points.astype("<f4", copy=False)
    preview_colors = np.clip(preview_colors * 255.0, 0, 255).astype(np.uint8)

    center = preview_points.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(preview_points - center, axis=1).max()) if len(preview_points) else 1.0
    radius = max(radius, 1e-6)

    points_b64 = encode_array_base64(preview_points)
    colors_b64 = encode_array_base64(preview_colors)
    center_json = "[" + ",".join(f"{value:.6f}" for value in center.tolist()) + "]"

    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Depth Anything Point Cloud Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: Arial, sans-serif; }
    #hud { position: fixed; left: 12px; top: 12px; padding: 10px 12px; background: rgba(0,0,0,.65); border: 1px solid #444; line-height: 1.55; font-size: 13px; }
    canvas { display: block; width: 100vw; height: 100vh; }
  </style>
</head>
<body>
<canvas id="view"></canvas>
<div id="hud">
  <div>points: __POINT_COUNT__</div>
  <div>drag to rotate, wheel to zoom</div>
  <div>WebGL preview, no Open3D required</div>
</div>
<script>
const POINT_COUNT = __POINT_COUNT__;
const POINTS_B64 = "__POINTS_B64__";
const COLORS_B64 = "__COLORS_B64__";
const CENTER = __CENTER_JSON__;
const RADIUS = __RADIUS__;

function decodeBase64ToBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function makeShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function makeProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0
  ]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function viewMatrix(yaw, pitch, distance) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const tx = -CENTER[0], ty = -CENTER[1], tz = -CENTER[2] - distance;
  return new Float32Array([
    cy, sy * sp, sy * cp, 0,
    0, cp, -sp, 0,
    -sy, cy * sp, cy * cp, 0,
    cy * tx - sy * tz, sy * sp * tx + cp * ty + cy * sp * tz, sy * cp * tx - sp * ty + cy * cp * tz, 1
  ]);
}

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl");
if (!gl) alert("WebGL is not supported by this browser");

const vertexSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uMvp;
varying vec3 vColor;
void main() {
  gl_Position = uMvp * vec4(aPosition, 1.0);
  gl_PointSize = 2.0;
  vColor = aColor;
}`;
const fragmentSource = `
precision mediump float;
varying vec3 vColor;
void main() {
  gl_FragColor = vec4(vColor, 1.0);
}`;

const program = makeProgram(gl, vertexSource, fragmentSource);
gl.useProgram(program);

const pointBytes = decodeBase64ToBytes(POINTS_B64);
const colorBytes = decodeBase64ToBytes(COLORS_B64);
const positions = new Float32Array(pointBytes.buffer);
const colorsU8 = new Uint8Array(colorBytes.buffer);
const colors = new Float32Array(colorsU8.length);
for (let i = 0; i < colorsU8.length; i++) colors[i] = colorsU8[i] / 255;

function bindBuffer(name, data, size) {
  const location = gl.getAttribLocation(program, name);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
}

bindBuffer("aPosition", positions, 3);
bindBuffer("aColor", colors, 3);

let yaw = 0.0;
let pitch = -0.25;
let distance = RADIUS * 2.5;
let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", event => { dragging = true; lastX = event.clientX; lastY = event.clientY; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", event => {
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.005;
  pitch += (event.clientY - lastY) * 0.005;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
});
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  distance *= Math.exp(event.deltaY * 0.001);
  distance = Math.max(RADIUS * 0.08, Math.min(RADIUS * 20, distance));
}, { passive: false });

function render() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.06, 0.06, 0.07, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  const proj = perspective(Math.PI / 4, canvas.width / canvas.height, RADIUS * 0.001, RADIUS * 50);
  const view = viewMatrix(yaw, pitch, distance);
  const mvp = multiply(proj, view);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "uMvp"), false, mvp);
  gl.drawArrays(gl.POINTS, 0, POINT_COUNT);
  requestAnimationFrame(render);
}
render();
</script>
</body>
</html>
"""

    html = html.replace("__POINT_COUNT__", str(len(preview_points)))
    html = html.replace("__POINTS_B64__", points_b64)
    html = html.replace("__COLORS_B64__", colors_b64)
    html = html.replace("__CENTER_JSON__", center_json)
    html = html.replace("__RADIUS__", f"{radius:.6f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")


def process_image(filename, model, args, device):
    """处理单张图片并生成点云。"""
    print(f"Processing: {filename}")
    image_bgr = cv2.imread(filename)
    if image_bgr is None:
        print(f"Skip unreadable image: {filename}")
        return

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        depth = model.infer_image(image_bgr, args.input_size)

    fx, fy, cx, cy = get_camera_intrinsics(width, height, args)
    points, valid_mask = depth_to_points(depth, fx, fy, cx, cy)
    colors = image_rgb.reshape(-1, 3)[valid_mask] / 255.0
    points, colors = prepare_pointcloud_arrays(points, colors, args)

    stem = Path(filename).stem
    output_path = Path(args.outdir) / f"{stem}.ply"
    save_pointcloud_ply(points, colors, output_path)
    print(f"Saved point cloud: {output_path}")

    if args.save_html or args.open_html:
        html_path = Path(args.outdir) / f"{stem}_viewer.html"
        save_html_viewer(points, colors, html_path, args.html_max_points)
        print(f"Saved HTML viewer: {html_path}")
        if args.open_html:
            webbrowser.open(html_path.resolve().as_uri())

    if args.save_depth_numpy:
        depth_path = Path(args.outdir) / f"{stem}_raw_depth_meter.npy"
        np.save(depth_path, depth)
        print(f"Saved depth: {depth_path}")

    if args.visualize:
        pcd = create_open3d_pointcloud(points, colors)
        o3d.visualization.draw_geometries([pcd], window_name=f"PointCloud: {stem}")


def main():
    """程序入口。"""
    args = parse_args()
    if not os.path.exists(args.load_from):
        raise FileNotFoundError(f"权重文件不存在: {args.load_from}")

    device = select_device(args.device)
    print(f"device: {device.type}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model = build_model(args, device)
    filenames = collect_image_paths(args.img_path)
    if not filenames:
        raise FileNotFoundError(f"没有找到输入图片: {args.img_path}")

    os.makedirs(args.outdir, exist_ok=True)
    for filename in filenames:
        process_image(filename, model, args, device)


if __name__ == "__main__":
    main()
