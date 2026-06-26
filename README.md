# Robocon OCR

面向 RoboCon 数学题目的离线 OCR 识别，支持数据集批量识别和 USB 摄像头实时识别。

## 常用命令

### 安装

```bash
git clone <repo-url> && cd OCR
python3 -m venv .venv-onnx && source .venv-onnx/bin/activate
pip install -r requirements.txt
```

### 摄像头实时识别

```bash
# 带调试窗口（最常用）
python3 -m robocon_ocr camera --show-window --show-stage-debug

# 无窗口纯识别
python3 -m robocon_ocr camera

# 指定摄像头设备
python3 -m robocon_ocr camera --device-index 0 --show-window --show-stage-debug
```

按 `q` 或 `Esc` 退出。默认只在识别结果变化时打印，`Ctrl+C` 停止。

### 生成测试数据并验证环境

```bash
# 生成 20 张难度 2 的四则运算题目
python3 question_generate/generate_expression_image.py -n 20 --complexity 2 -o test_dataset

# 用默认 ONNX 后端识别
python3 -m robocon_ocr test_dataset
```

如果表达式和答案与 `test_dataset/problems_and_answers.txt` 标注一致，说明环境正常。

### 数据集批量识别

```bash
# 默认 ONNX 后端
python3 -m robocon_ocr dataset/num_100_com_4

# 显式使用 PaddleOCR
python3 -m robocon_ocr dataset/num_100_com_4 --ocr-backend lightweight
```

### 分阶段调试

```bash
# 只跑到表达式区域提取（调题板和 ROI 常用）
python3 -m robocon_ocr camera --show-window --show-stage-debug \
  --stop-after-stage expression_region

# 只跑到 OCR 阶段
python3 -m robocon_ocr camera --show-window --show-stage-debug \
  --stop-after-stage ocr

# 保存各阶段调试图到磁盘
python3 -m robocon_ocr camera --show-window --show-stage-debug \
  --stop-after-stage expression_region \
  --debug-dir debug_outputs --debug-save-stages
```

可用阶段：`board_detection` → `rectification` → `expression_region` → `enhancement` → `segmentation` → `ocr` → `postprocess`

### 摄像头关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device-index` | USB 设备号 (`/dev/video0`=0) | 2 |
| `--show-window` | 打开实时调试窗口 | 关闭 |
| `--show-stage-debug` | 显示 2x2 分阶段调试面板 | 关闭 |
| `--stop-after-stage` | 在指定阶段截停 | 不截停 |
| `--max-frames` | 最多识别帧数后自动退出 | 不限 |
| `--print-all` | 结果没变也打印每次输出 | 仅变化时打印 |
| `--save-frame` | 保存抓拍原图到指定路径 | 不保存 |
| `--window-scale` | 调试窗口缩放比例 | 0.75 |

如果 `--show-window` 报 `cv2.imshow` 不支持，说明环境装的是 `opencv-python-headless`：

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python
```

## OCR 后端

| 后端 | 引擎 | 延迟 | 依赖 | 适用场景 |
|------|------|------|------|----------|
| `onnx`（默认） | ONNX Runtime | ~10-50ms | onnxruntime | Jetson / ARM64 / 跨平台 |
| `lightweight` | PaddleOCR TextRecognition | ~50-200ms | PaddlePaddle | x86 GPU 加速 |
| `api` | 联网多模态大模型 | ~1-3s | requests | 离线识别兜底、复杂题目 |

`dataset` 和 `camera` 模式均默认 `onnx`，无需 `--ocr-backend` 参数。ONNX 后端精度与 PaddleOCR 一致（200 样本 0 差异）。

切换到其他后端：

```bash
# PaddleOCR
source .venv-paddle/bin/activate
python3 -m robocon_ocr camera --ocr-backend lightweight --show-window --show-stage-debug

# 联网大模型（Moonshot / OpenAI / Qwen / Gemini）
python3 -m robocon_ocr camera --ocr-backend api --show-window --show-stage-debug
```

### 联网大模型 API 后端

适用于本地 OCR 识别失败时的兜底方案。摄像头模式下 API 调用**非阻塞**，流水线持续运行题板检测和图像纠正，OCR 结果 1-3 秒后回填，不影响实时画面流畅度。

#### 配置 API 密钥

编辑项目根目录 `.env` 文件：

```bash
# Moonshot（默认）
MOONSHOT_API_KEY=sk-your-key-here
MOONSHOT_MODEL=moonshot-v1-8k

# 切换其他提供商时修改 api_provider 字段或设对应环境变量
# 支持: moonshot / openai / qwen / gemini / custom
```

也可通过命令行参数或代码中 `OCRConfig.api_provider` 切换提供商。环境变量优先级：CLI 参数 > 环境变量 > 默认值。

#### 支持的提供商

| 提供商 | `api_provider` | 默认模型 | 协议 |
|--------|---------------|----------|------|
| Moonshot / Kimi | `moonshot` | moonshot-v1-8k | OpenAI 兼容 |
| OpenAI | `openai` | gpt-4o-mini | OpenAI 兼容 |
| 通义千问 VL | `qwen` | qwen-vl-max | OpenAI 兼容 |
| Gemini | `gemini` | gemini-2.5-flash | Gemini 原生 |
| 自定义 | `custom` | — | OpenAI 兼容 |

#### 添加新提供商

需要代码级修改时，只需在 [`api_recognizer.py`](robocon_ocr/image_recognition/api_recognizer.py) 的 `_PROVIDERS` 字典中追加一条记录，其余逻辑自动复用。

## 项目架构

```text
robocon_ocr/
├── vision_capture/       # USB 摄像头采集
├── vision_processing/    # 题板检测、透视矫正、增强、表达式区域提取
├── image_recognition/    # OCR 识别层（ONNX / PaddleOCR / 联网大模型 后端）
├── result/               # 表达式规范化、求值、结果汇总
├── config.py             # 可调参数（OCR、预处理、摄像头）
├── pipeline.py           # 单张/批量推理入口
└── staged_pipeline.py    # 7 阶段流水线
```

流水线：`board_detection → rectification → expression_region → enhancement → segmentation → ocr → postprocess`

## 参数调优

### 摄像头参数

编辑 [robocon_ocr/camera_tuning.py](robocon_ocr/camera_tuning.py)，包含设备号、分辨率、帧率、曝光、白平衡、对焦、增益等。推荐调参顺序：`exposure_time_absolute` → `focus_absolute` → `contrast` → `gain`/`brightness`。

如果某项不支持，程序会打印 `[camera-init]` 警告但不会中断。

### ROI 检测参数

编辑 [robocon_ocr/roi_tuning.py](robocon_ocr/roi_tuning.py)，包含 `white_threshold`、`edge_threshold`、`min_roi_area_ratio`、比例容差等。推荐调参顺序：`min_roi_area_ratio` → `edge_threshold` → `white_threshold` → 比例容差。

### 表达式区域提取参数

编辑 [robocon_ocr/roi_tuning.py](robocon_ocr/roi_tuning.py) 中 `expression_` 开头的参数。关键参数：

| 参数 | 作用 |
|------|------|
| `expression_bbox_padding_x/y` | 表达式区域留白，裁字符时增大 |
| `expression_otsu_bias` | 二值化偏移，抓不到字符时增大 |
| `expression_enter_ratio` | 进入前景的阈值比例 |
| `expression_min_consecutive_rows/cols` | 最小连续行/列，边缘抖动大时增大 |

## 调试指南

### 调试顺序

按阶段顺序调试，不要跳步：

1. **board_detection** — 题板四边形是否稳定
2. **rectification** — 透视图是否拉正
3. **expression_region** — 表达式 bbox 是否完整
4. **enhancement** — 二值图是否清晰
5. **ocr** — 识别结果是否稳定

### 表达式区域提取

打开 `--show-stage-debug` 观察：
- 表达式首尾字符是否完整保留
- bbox 是否明显小于整张透视图
- 正视和斜视时区域是否稳定

失败时回看两类原因：`rectification` 不稳定，或表达式区域提取参数太紧/太松。

### OCR 识别

前序阶段稳定后再调 OCR。重点关注：
- `raw_text` 是否经常为空 → 检查 enhancement 输出
- 数字混淆（2/3/5/8）→ 检查增强是否让开口/闭口变形
- 运算符混淆（÷/+、×/x）→ 检查除号小点是否清楚分离
- 同一题轻微抖动结果是否乱跳 → OCR 稳定性问题

OCR 输出经过规范化（`x/*→×`、`/→÷`、中文括号→英文括号）后，再通过字符集限制（只接受 `0-9 + - × ÷ ( ) =`）和表达式解析。

## PaddleOCR 部署（可选，仅 x86 GPU 需要）

### 安装

```bash
python3 -m venv .venv-paddle && source .venv-paddle/bin/activate
pip install -r requirements-paddle.txt

# GPU 用户额外安装对应 CUDA 版本的 PaddlePaddle
# RTX 4060 / CUDA 12.6 示例：
pip install --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ paddlepaddle-gpu==3.3.1
```

### 从 Paddle 模型导出 ONNX（PC 端操作）

```bash
source .venv-paddle/bin/activate
pip install paddle2onnx
paddle2onnx --model_dir ~/.paddlex/official_models/PP-OCRv5_server_rec/ \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file models/PP-OCRv5_server_rec.onnx \
            --opset_version 14
```

## Jetson 部署

推荐使用 ONNX 后端。Jetson 若需 PaddlePaddle：

**JetPack 5.x:**
```bash
wget https://paddle-inference-lib.bj.bcebos.com/2.5.0/jetson/jetpack5.0.2/all/paddlepaddle_gpu-2.5.0-cp38-cp38-linux_aarch64.whl
pip install paddlepaddle_gpu-2.5.0-cp38-cp38-linux_aarch64.whl
pip install -r requirements-paddle.txt
```

**JetPack 6.x:** 优先 ONNX 后端，避免 PaddlePaddle 兼容问题。

### 常见问题

| 问题 | 解决 |
|------|------|
| `import paddle` 报 `libcublas.so` 找不到 | PaddlePaddle 与 JetPack CUDA 版本不匹配，换 ONNX 后端 |
| `cv2.imshow` 报错 | `pip install opencv-python` 替换 headless 版本 |
| 摄像头打不开 | `ls /dev/video*` 确认设备号；`sudo usermod -aG video $USER` |
