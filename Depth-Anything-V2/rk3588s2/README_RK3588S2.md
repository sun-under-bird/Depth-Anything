# Depth Anything V2 在 RK3588S2 上部署说明

这个目录用于把 Depth Anything V2 的 PyTorch 权重转换成 RKNN，并在 RK3588S2 的 NPU 上运行。

推荐流程：

```text
.pth 权重 -> ONNX -> RKNN -> RK3588S2 NPU 推理
```

RK3588S2 在 RKNN 里使用的目标平台参数是：

```text
rk3588
```

## 1. 640x480 图片能不能用

可以用 `640x480` 的图片或摄像头画面。

这里要区分两个尺寸：

```text
输入图片尺寸：摄像头或图片本身的尺寸，比如 640x480
模型输入尺寸：ONNX/RKNN 固定输入尺寸，比如 322x434
```

板端 `infer_rknn.py` 的处理方式是：

```text
读取原图 -> resize 到 RKNN 固定输入 -> NPU 推理 -> 深度图 resize 回原图尺寸
```

所以，原始图片可以是 `640x480`。但是当前导出的 Depth Anything V2 ONNX/RKNN 模型输入高宽需要能被 `14` 整除，因为 DINOv2 的 patch size 是 `14`。

因此不建议直接导出 `640x480` 模型输入：

```text
640 不能被 14 整除
480 不能被 14 整除
```

如果摄像头是 `640x480`，推荐先用：

```text
322x434
```

原因是：

```text
322 = 14 x 23
434 = 14 x 31
322x434 接近 640x480 的 4:3 比例
速度比接近全分辨率输入更容易接受
```

如果你想更接近 `640x480` 的细节，可以尝试：

```text
476x630
```

原因是：

```text
476 = 14 x 34
630 = 14 x 45
476x630 接近 640x480 的 4:3 比例
```

但 `476x630` 会明显更慢，占用内存也更多。建议先跑通 `322x434`，再对比更大的输入尺寸。

## 2. 准备权重

建议先使用最小的 `vits` 模型。

相对深度模型常见权重路径：

```text
metric_depth/checkpoints/depth_anything_v2_vits.pth
```

相对深度输出不是米制深度，只能表示远近关系。

如果要米制室内深度，使用 Hypersim metric 权重：

```text
metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

室内 Hypersim metric 模型建议：

```text
--max-depth 20
```

室外 VKITTI metric 模型建议：

```text
--max-depth 80
```

## 3. 在 PC 上导出相对深度 ONNX

如果你使用的是：

```text
depth_anything_v2_vits.pth
```

从 `Depth-Anything-V2` 项目根目录运行：

```bash
python rk3588s2/export_relative_onnx.py \
  --encoder vits \
  --checkpoint metric_depth/checkpoints/depth_anything_v2_vits.pth \
  --height 322 \
  --width 434 \
  --output rk3588s2/models/depth_anything_v2_vits_322x434.onnx
```

如果只是先快速验证，也可以用正方形输入：

```bash
python rk3588s2/export_relative_onnx.py \
  --encoder vits \
  --checkpoint metric_depth/checkpoints/depth_anything_v2_vits.pth \
  --height 322 \
  --width 322 \
  --output rk3588s2/models/depth_anything_v2_vits_322x322.onnx
```

注意：`--height` 和 `--width` 必须都能被 `14` 整除。

## 4. 在 PC 上导出 metric 深度 ONNX

如果你使用的是米制深度权重，比如：

```text
depth_anything_v2_metric_hypersim_vits.pth
```

推荐先导出适合 `640x480` 摄像头的 `322x434` 输入模型：

```bash
python rk3588s2/export_metric_onnx.py \
  --encoder vits \
  --checkpoint metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  --max-depth 20 \
  --height 322 \
  --width 434 \
  --output rk3588s2/models/depth_anything_v2_metric_hypersim_vits_322x434.onnx
```

如果你想测试更接近 `640x480` 的模型输入，可以导出：

```bash
python rk3588s2/export_metric_onnx.py \
  --encoder vits \
  --checkpoint metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  --max-depth 20 \
  --height 476 \
  --width 630 \
  --output rk3588s2/models/depth_anything_v2_metric_hypersim_vits_476x630.onnx
```

`476x630` 会比 `322x434` 慢很多，建议只在 `322x434` 跑通后再测试。

## 5. 将 ONNX 转成 RKNN

这一步建议在 Linux/x86_64 环境执行，并安装 `rknn-toolkit2`。

转换 `322x434` metric 模型示例：

```bash
python rk3588s2/convert_onnx_to_rknn.py \
  --onnx rk3588s2/models/depth_anything_v2_metric_hypersim_vits_322x434.onnx \
  --output rk3588s2/models/depth_anything_v2_metric_hypersim_vits_322x434.rknn \
  --target-platform rk3588
```

刚开始不要直接做 INT8 量化。先用非量化模型确认 RKNN 输出和 PyTorch 输出趋势一致，再考虑量化。

## 6. 在 RK3588S2 上运行单张图片

把这些文件复制到板子：

```text
.rknn 模型
rk3588s2/infer_rknn.py
测试图片
```

板端 Python 依赖：

```text
rknn-toolkit-lite2
opencv-python
numpy
```

检查 NPU 设备：

```bash
ls /dev/rknpu*
```

运行 `640x480` 图片示例：

```bash
python3 rk3588s2/infer_rknn.py \
  --model rk3588s2/models/depth_anything_v2_metric_hypersim_vits_322x434.rknn \
  --img-path assets/examples/demo_640x480.jpg \
  --outdir rk3588s2_outputs \
  --height 322 \
  --width 434 \
  --save-numpy
```

这里的 `--height 322 --width 434` 是 RKNN 模型输入尺寸，不是原始图片尺寸。原始图片可以是 `640x480`，脚本会自动 resize。

如果你导出的是 `476x630` 模型，板端也必须使用相同尺寸：

```bash
python3 rk3588s2/infer_rknn.py \
  --model rk3588s2/models/depth_anything_v2_metric_hypersim_vits_476x630.rknn \
  --img-path assets/examples/demo_640x480.jpg \
  --outdir rk3588s2_outputs \
  --height 476 \
  --width 630 \
  --save-numpy
```

## 7. 摄像头输入建议

如果摄像头输出是 `640x480`，建议保持采集分辨率为：

```text
640x480
```

然后在推理前 resize 到 RKNN 模型输入：

```text
322x434
```

这样做的好处是：

```text
摄像头内参仍然按 640x480 使用
深度图最后可以 resize 回 640x480
点云反投影时可以用 640x480 对应的 fx/fy/cx/cy
```

如果你把图像先裁剪、拉伸或换成其他尺寸，内参也必须同步换算，否则点云会变形。

## 8. 尺寸和内参关系

如果原始内参是按 `1280x720` 标定的，而你把图像缩放到 `640x480`，需要分别按宽高比例换算：

```text
fx_new = fx_old x 640 / 1280
fy_new = fy_old x 480 / 720
cx_new = cx_old x 640 / 1280
cy_new = cy_old x 480 / 720
```

你现在给的 `640x480` 内参是：

```text
fx = 423.1229070051419
fy = 562.8160548615311
cx = 286.1811834652652
cy = 255.7353546318091
```

这些内参只适用于最终用于点云反投影的深度图也是 `640x480` 的情况。

## 9. 常见注意事项

- 优先用 `vits`，`vitb` 和 `vitl` 更占内存，转换难度也更高。
- ONNX 导出尺寸和板端 `infer_rknn.py` 的 `--height/--width` 必须一致。
- RKNN 固定输入高宽必须能被 `14` 整除，不要直接使用 `640x480` 作为模型输入。
- 原始图片或摄像头画面可以是 `640x480`。
- 当前 ONNX 输入是已经归一化后的 `NCHW float32`，所以 RKNN 转换脚本里没有配置 mean/std。
- 如果要做摄像头实时推理，可以复用 `infer_rknn.py` 里的 `load_rknn`、`run_inference` 和 `depth_to_visual`，再接 `cv2.VideoCapture`。
- 如果要做点云，使用深度图对应分辨率的相机内参，单位是像素。
