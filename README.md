# Robocon OCR

这是一个面向 RoboCon 四则运算题目的离线识别项目骨架。

当前阶段只实现：

- `dataset` 图片作为输入
- `pix2tex` 离线识别题目
- `USB` 摄像头实时取流并持续送入 OCR
- 对识别结果做规则化、表达式求值、答案输出
- 用标注文件评估识别准确率

暂未实现：

- 视觉处理层复杂透视矫正、去反光、去模糊

## 项目分层

```text
src/robocon_ocr/
├── vision_capture/       # 视觉采集层，占位接口
├── vision_processing/    # 视觉处理层，占位接口
├── image_recognition/    # 图片识别层，当前核心
├── result/               # 结果层：规范化、求值、评估、输出
├── config.py
└── pipeline.py
```

当前主链路：

```text
dataset/*.png
  -> 定位白色题板 ROI（优先矩形，回退四边形）
  -> 透视拉正 + Otsu 二值化
  -> OCR文本规则化
  -> 表达式求值
  -> 输出 expression / answer / confidence
```

## 为什么这个方案适合你现在的阶段

你的题目图片有几个非常重要的优势：

- 正对屏幕，没有透视畸变
- 单行显示，不换行
- 白底黑字，背景稳定
- 字体固定为 `Times New Roman`
- 字符集合固定：数字、`+ - × ÷ ( ) =`

所以第一阶段没必要先做复杂检测网络，可以直接：

1. 先找大块高对比白色题板区域，长宽比约束为 `16:9`
2. 如果矩形不稳定，再找 4 个角点构成的四边形题板
3. 对 ROI 做透视拉正
4. 在 ROI 内做 Otsu 二值化和放大
5. 送给 `pix2tex`
6. 用规则层把 OCR 输出收敛为 `= ( ) 0-9 + - × ÷`
7. 再独立计算答案

这样系统会比“直接信 OCR 原始文本”稳很多。

## 安装建议

建议单独创建虚拟环境，并固定 `numpy<2`，避免目前很多 OCR 相关包与 `numpy 2.x` 的兼容问题。
同时建议优先使用 `Python 3.11` 或 `Python 3.12`。
当前一些 `pix2tex` 相关下游依赖在 `Python 3.14` 上还不稳定，容易退回源码编译。



```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`pix2tex` 依赖 `torch`，首次运行时还可能自动下载模型权重，所以第一次启动会比后续慢一些。
当前默认配置按 `CPU-only Linux` 规划，不需要额外安装 `tesseract` 系统命令。
如果你使用的是 `Python 3.14`，安装时又看到 `stringzilla`、`gcc-12` 之类的编译错误，优先重新创建 `Python 3.11/3.12` 虚拟环境，而不是继续硬补编译链。

## 运行离线识别

识别一个数据集目录：

```bash
python3 -m robocon_ocr dataset/num_100_com_4
```

保存调试图：

```bash
python3 -m robocon_ocr dataset/num_100_com_8 \
  --debug-dir debug_outputs
```

如需手动覆盖标注文件路径：

```bash
python3 -m robocon_ocr dataset/num_100_com_4 \
  --label-file dataset/num_100_com_4/problems_and_answers.txt
```

兼容旧入口：

```bash
python3 scripts/run_offline_pipeline.py dataset/num_100_com_4
```

## 运行 USB 摄像头实时识别

持续取流并实时输出识别结果：

```bash
python3 -m robocon_ocr camera \
  --show-window
```

如果 `--show-window` 报 OpenCV `cv2.imshow` 不支持，一般是环境里装成了 `opencv-python-headless`。
可用下面这组命令修复：

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python==4.13.0.92
```

如果你当前只想先跑识别，也可以先去掉 `--show-window`。

默认只在识别结果发生变化时打印新结果，按 `Ctrl+C` 停止。
实时模式现在是“采集线程持续抓图 + OCR 线程后台处理最新帧”的异步模型。
默认摄像头参数是 `device-index=2`、`1280x720`、`30fps`、`MJPG`、`interval-ms=0`。
默认固定参数写在 [robocon_ocr/camera_tuning.py](/home/angela/Robocon/OCR/robocon_ocr/camera_tuning.py) 里，程序启动时会按固定顺序自动尝试应用曝光、白平衡、对焦、增益等控制项。
白色题板 ROI 检测参数单独写在 [robocon_ocr/roi_tuning.py](/home/angela/Robocon/OCR/robocon_ocr/roi_tuning.py) 里，方便单独调试阈值、面积和比例限制。
OCR 输出会被严格限制在 `= ( ) 0-9 + - × ÷` 这组字符内，超出范围会直接判定失败。

如果你想限制抓图帧数，或者保存拍摄原图和预处理调试图：

```bash
python3 -m robocon_ocr camera \
  --device-index 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --pixel-format MJPG \
  --warmup-frames 8 \
  --max-frames 50 \
  --show-window \
  --save-frame captures/latest_frame.png \
  --debug-dir debug_outputs
```

打开窗口调试后，会显示：

- 彩色原图，并叠加 ROI 四边形
- 灰度原图，并叠加 ROI 四边形
- 真正送入 OCR 的输入图
- OCR 结果和调试信息

在彩色原图旁边，还会显示当前 ROI 判定信息，包括：

- 当前阈值
- 当前最佳候选的面积 / 边缘 / 比例测量值
- 当前失败原因，方便判断是哪个参数限制了检测

在窗口里按 `q` 或 `Esc` 可以退出实时识别。

支持的关键参数：

- `--device-index`：USB 设备号，对应常见的 `/dev/video0`、`/dev/video1`
- `--fps`：请求摄像头输出帧率
- `--pixel-format`：请求视频格式，当前推荐固定为 `MJPG`
- `--width` / `--height`：请求分辨率
- `--warmup-frames`：丢弃前几帧，等曝光稳定后再识别
- `--interval-ms`：保留的兼容参数，异步最新帧模式下默认不主动节流
- `--max-frames`：最多识别多少帧后自动退出，便于调试
- `--print-all`：即使识别结果没变也持续打印每次 OCR 输出
- `--show-window`：用 `cv2` 打开实时调试窗口
- `--window-scale`：调试窗口缩放比例，适合高分辨率画面
- `--save-frame`：保存抓拍到的原始 RGB 图片

## 摄像头固定参数调优

推荐直接编辑 [robocon_ocr/camera_tuning.py](/home/angela/Robocon/OCR/robocon_ocr/camera_tuning.py)。

这份文件专门服务于“电子显示屏中的白色题板 + 黑色题目 OCR”场景，包含：

- 固定采集参数：`device_index`、`width`、`height`、`fps`、`pixel_format`
- 固定控制参数：曝光、白平衡、对焦、增益、对比度、锐度等
- 每个参数的中文注释
- 每个参数的调试建议和常见失效现象

推荐调参顺序：

1. 先调 `exposure_time_absolute`
2. 再调 `focus_absolute`
3. 再调 `contrast`
4. 最后才考虑 `gain` 和 `brightness`

如果某个控制项当前设备不支持，程序会在初始化阶段打印 `[camera-init]` 警告，但不会中断取流。

## ROI 参数调优

推荐直接编辑 [robocon_ocr/roi_tuning.py](/home/angela/Robocon/OCR/robocon_ocr/roi_tuning.py)。

这份文件专门维护白色题板 ROI 检测参数，包含：

- `white_threshold`
- `edge_threshold`
- `min_roi_area_ratio`
- `rectangle_ratio_tolerance`
- `quadrilateral_ratio_tolerance`
- `target_aspect_ratio`
- `roi_padding`
- `perspective_width`
- `perspective_height`
- `scale_factor`

推荐调参顺序：

1. 先调 `min_roi_area_ratio`
2. 再调 `edge_threshold`
3. 再调 `white_threshold`
4. 最后调 `rectangle_ratio_tolerance` 和 `quadrilateral_ratio_tolerance`

## 输出内容

脚本会输出每张图的：

- 图片名
- OCR 原始文本
- 规范化后的表达式
- 解析得到的答案
- OCR 置信度
- 是否与标注一致

最后会给出汇总统计：

- 表达式完全匹配率
- 答案匹配率
- 失败样例数量

## 推荐的迭代路线

### 第 1 阶段：现在

- 用 `dataset` 跑通离线 OCR 基线
- 建立评测指标
- 明确常见误识别类型

### 第 2 阶段：接入视觉处理层

- 屏幕定位
- 透视矫正
- 自适应阈值
- 去反光和锐化

### 第 3 阶段：接入视觉采集层

- USB 摄像头取流
- 多帧去抖
- 按置信度投票

### 第 4 阶段：比赛联调

- 限时识别
- 异常回退
- 结果缓存
- 与控制系统联动
