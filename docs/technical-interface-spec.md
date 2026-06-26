# Robocon OCR 技术接口交接文档

## 1. 设计目标

本项目为 Robocon 比赛中的 **实时摄像头数学表达式识别** 模块。父项目（机器人主控程序）通过统一的 Python API 调用本模块，无需关心内部流水线细节。

核心设计原则：

- **三后端统一接口** — ONNX、PaddleOCR、联网大模型三种后端，调用方式完全一致，父项目只需改一个参数即可切换。
- **信号量 + 回调双通知机制** — 识别结果确认后，主动通知父项目，无需轮询。
- **本地模型 3 帧共识滤波** — ONNX / PaddleOCR 本地模型需要连续 3 帧（表达式 + 答案）一致才输出，防止抖动误判。联网大模型准确率足够高，直通不滤波。

---

## 2. 模块结构

```
robocon_ocr/
├── recognition_output.py   # 标准化输出数据结构 RecognitionOutput
├── recognition_filter.py   # 共识滤波器 RecognitionFilter
├── camera_session.py       # 摄像头识别会话 CameraRecognitionSession
├── __init__.py             # 公开 API 导出
├── config.py               # 配置数据类（OCRConfig, CameraConfig, PipelineConfig）
├── staged_pipeline.py      # 7 阶段识别流水线
├── image_recognition/      # 三个 OCR 后端实现
│   ├── onnx_recognizer.py
│   ├── lightweight_recognizer.py
│   └── api_recognizer.py
├── vision_capture/         # USB 摄像头采集
└── vision_processing/      # 图像预处理（题板检测、矫正、增强等）
```

父项目只需导入顶层 `robocon_ocr` 包，使用三个公开类：

| 类名 | 职责 |
|---|---|
| `RecognitionOutput` | 标准化识别结果，包含答案和 `answer % 4` |
| `RecognitionFilter` | 共识滤波器（高级用法，通常不直接使用） |
| `CameraRecognitionSession` | **主入口**，封装摄像头采集 → 流水线 → 滤波 → 通知 |

---

## 3. RecognitionOutput — 标准化输出

### 3.1 字段说明

```python
from robocon_ocr import RecognitionOutput

@dataclass(slots=True)
class RecognitionOutput:
    expression: str          # 识别出的数学表达式，如 "12+34×2"
    answer: int | None       # 计算结果，如 80
    answer_mod_4: int | None # answer % 4，用于决策（如 80 % 4 = 0）
    is_valid: bool           # 是否成功识别并正确求值
    confidence: float        # OCR 置信度，范围 [0.0, 1.0]
    backend: str             # 来源后端："onnx" / "lightweight" / "api"
    error: str | None        # 错误描述，成功时为 None
```

### 3.2 重要约定

- **`is_valid == True`** 时，`answer` 和 `answer_mod_4` **一定非 None**，可直接用于决策。
- **`is_valid == False`** 时，应查看 `error` 字段了解原因，`answer` 为 None。
- `confidence` 的含义因后端而异：
  - ONNX / PaddleOCR：基于 CTC 解码的概率值。
  - API（联网大模型）：固定为 `1.0`（大模型不返回 token 级概率）。

### 3.3 可能的 error 值

| error 值 | 含义 |
|---|---|
| `None` | 正常 |
| `"no text detected by OCR"` | OCR 引擎未检测到文字 |
| `"unsupported symbol outside arithmetic charset"` | 检测到非算术字符 |
| `"empty expression"` | 表达式为空 |
| `"model could not recognize the expression"` | 大模型返回 `<UNKNOWN>` |
| `"non-integer division"` | 表达式含非整除 |
| `"division by zero"` | 表达式含除零 |

---

## 4. RecognitionFilter — 共识滤波器

### 4.1 滤波策略

| 后端 | 策略 | 说明 |
|---|---|---|
| `"api"` | 共识数 = 1，直通 | 联网大模型单次识别已经足够准确 |
| `"onnx"` | 共识数 = 3 | 需连续 3 帧 `(expression, answer)` 完全相同 |
| `"lightweight"` | 共识数 = 3 | 同上 |

### 4.2 滤波逻辑细节

对于本地模型（共识数 = 3）：

1. 每帧识别结果 `is_valid == False` → 清空缓冲区，不输出。
2. 每帧识别结果 `is_valid == True`：
   - 若缓冲区为空，或当前 `(expression, answer)` 与缓冲区最后一条相同 → 追加到缓冲区。
   - 若 `(expression, answer)` 与缓冲区最后一条 **不同** → 清空缓冲区，以当前帧作为新的第一条。
3. 缓冲区累积到 3 条 → **输出**（触发信号量 / 回调），然后清空缓冲区。
4. 下一次输出需要再次累积 3 条，防止同一个结果被反复输出。

### 4.3 直接使用（通常不需要）

```python
from robocon_ocr import RecognitionFilter, RecognitionOutput

f = RecognitionFilter(backend="onnx", consensus=3)

for each_frame:
    output = RecognitionOutput(...)
    emitted = f.feed(output)
    if emitted is not None:
        print(f"共识达成: {emitted.answer}")

# 切换场景时重置
f.reset()
```

---

## 5. CameraRecognitionSession — 主入口

### 5.1 构造函数

```python
class CameraRecognitionSession:
    def __init__(
        self,
        backend: str = "onnx",                           # 后端选择
        on_result: Callable[[RecognitionOutput], None] | None = None,  # 回调
        filter_consensus: int = 3,                       # 共识帧数（api 自动为 1）
        camera_config: CameraConfig | None = None,       # 完整摄像头配置
        **camera_overrides,                              # 单项摄像头参数覆盖
    ):
```

### 5.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | `str` | `"onnx"` | `"onnx"` / `"lightweight"` / `"api"` |
| `on_result` | `Callable \| None` | `None` | 回调函数，收到 `RecognitionOutput` 时调用 |
| `filter_consensus` | `int` | `3` | 共识帧数，仅对本地模型生效，api 自动忽略 |
| `camera_config` | `CameraConfig \| None` | `None` | 完整摄像头配置，默认使用 `DEFAULT_CAMERA_TUNING` |
| `**camera_overrides` | — | — | 单项覆盖，如 `device_index=2` |

### 5.3 公共方法

| 方法 | 说明 |
|---|---|
| `start()` | **阻塞启动**。直到 `stop()` 或 `Ctrl+C` 才返回。适合主线程直接调用。 |
| `start_async()` | **异步启动**。在后台守护线程中运行，立即返回。适合需要同时做其他事情的场景。 |
| `stop()` | 通知会话停止，等待后台线程退出（最多等 5 秒）。 |

### 5.4 公共属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `result_ready` | `threading.Event` | **信号量**。有新结果确认时自动 set。父项目可 `wait()` 阻塞等待。 |
| `latest_output` | `RecognitionOutput \| None` | 最近一次确认的识别结果。线程安全。 |

---

## 6. 使用示例

### 6.1 回调模式（推荐）

最简洁的方式，识别结果到达时自动回调。

```python
from robocon_ocr import CameraRecognitionSession, RecognitionOutput

def on_result(output: RecognitionOutput) -> None:
    """识别确认后的回调 — 在这里做决策。"""
    print(f"[识别到] 表达式: {output.expression}")
    print(f"[识别到] 答案: {output.answer}")
    print(f"[决策用] answer % 4 = {output.answer_mod_4}")
    # 发送给机器人控制模块
    # robot.execute(output.answer_mod_4)

# 启动会话（阻塞）
session = CameraRecognitionSession(
    backend="onnx",
    on_result=on_result,
    device_index=2,
)
session.start()  # 阻塞直到 Ctrl+C
```

### 6.2 信号量模式

父项目在自有的主循环中检查识别结果。

```python
from robocon_ocr import CameraRecognitionSession

session = CameraRecognitionSession(backend="api", device_index=2)
session.start_async()  # 后台运行，立即返回

while robot.is_running():
    # 等待识别结果，超时 1 秒去检查机器人状态
    if session.result_ready.wait(timeout=1.0):
        output = session.latest_output
        if output.is_valid:
            robot.navigate_to(output.answer_mod_4)
        session.result_ready.clear()  # 消费信号量

# 比赛结束
session.stop()
```

### 6.3 切换后端

只需改 `backend` 参数，代码完全不变：

```python
# 方案 A：本地 ONNX 推理（默认，延迟 ~10-50ms）
session = CameraRecognitionSession(backend="onnx", on_result=on_result)

# 方案 B：本地 PaddleOCR（延迟 ~50-200ms，需 .venv-paddle 环境）
session = CameraRecognitionSession(backend="lightweight", on_result=on_result)

# 方案 C：联网大模型（延迟 ~1-3s，需配置 API Key）
session = CameraRecognitionSession(backend="api", on_result=on_result)
```

### 6.4 摄像头参数覆盖

```python
session = CameraRecognitionSession(
    backend="onnx",
    on_result=on_result,
    device_index=0,        # 覆盖默认设备号
    width=1920,            # 覆盖分辨率
    height=1080,
    exposure_time_absolute=150,  # 覆盖曝光时间
    contrast=25,           # 覆盖对比度
)
session.start()
```

默认值来自 `robocon_ocr/camera_tuning.py` 中的 `DEFAULT_CAMERA_TUNING`：

| 参数 | 默认值 |
|---|---|
| `device_index` | 2 |
| `width` | 1280 |
| `height` | 720 |
| `fps` | 30.0 |
| `pixel_format` | MJPG |
| `warmup_frames` | 5 |
| `capture_timeout_ms` | 3000 |
| `exposure_time_absolute` | 200 |
| `contrast` | 20 |
| `sharpness` | 4 |
| `white_balance_temperature` | 4600 |
| `focus_absolute` | 190 |

---

## 7. 流水线内部阶段

虽然父项目无需关心流水线内部，但了解以下阶段有助于调试：

```
原始帧 → board_detection → rectification → expression_region
       → enhancement → segmentation → ocr → postprocess
```

每个阶段的作用：

| 阶段 | 输入 | 输出 | 说明 |
|---|---|---|---|
| board_detection | 原始彩色帧 | ROI 四边形 | 定位白色题板区域 |
| rectification | 原始帧 + ROI | 矫正后的题板图 | 透视变换拉正 |
| expression_region | 矫正图 | 表达式裁剪区域 | 定位题目所在行列 |
| enhancement | 裁剪区域 | 二值化增强图 | CLAHE + 降噪 + 自适应阈值 |
| segmentation | 增强图 | 字符片段列表 | 当前为整行输入占位 |
| ocr | 待识别图 | OCR 原始文本 | 三个后端在这里分叉 |
| postprocess | OCR 文本 | 表达式 + 答案 | 字符清洗 → 表达式解析 → 求值 → 修正 |

---

## 8. API 后端环境配置

使用联网大模型时，需配置对应提供商的 API Key。在项目根目录创建 `.env` 文件：

```bash
# Moonshot (Kimi) — 默认提供商
MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxx

# 或 OpenAI
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 或 通义千问 VL
# QWEN_API_KEY=sk-xxxxxxxxxxxxxxxx

# 或 Gemini
# GEMINI_API_KEY=xxxxxxxxxxxxxxxx

# 可选：覆盖默认模型
# MOONSHOT_MODEL=moonshot-v1-8k-vision-preview
```

通过 `OCRConfig.api_provider` 切换提供商（当前通过 `.env` 隐式选择，将来可扩展为显式参数）。

API 后端的冷却间隔为 **3 秒**（`api_recognizer.py` 中的 `_min_interval`），防止频繁调用消耗配额。

---

## 9. 线程模型

`CameraRecognitionSession` 内部使用 **双线程异步架构**：

```
┌─ 采集线程 (cam-cap) ──────────────────┐
│  USB 摄像头持续读取原始帧              │
│  发布到 LatestFrameBuffer（单槽缓冲）  │
└────────────────┬───────────────────────┘
                 │ 只保留最新帧
                 ▼
┌─ OCR 线程 (cam-ocr) ───────────────────┐
│  等待最新帧 → 7 阶段流水线 → 滤波     │
│  共识达成 → set result_ready 信号量    │
│  共识达成 → 调用 on_result 回调        │
└────────────────────────────────────────┘
```

`LatestFrameBuffer` 是一个单槽线程安全缓冲：
- 采集线程发布帧时覆盖旧帧。
- OCR 线程取出时拷贝一份，不阻塞采集线程。
- 如果 OCR 处理速度跟不上采集帧率，中间的帧会被**自动丢弃**，OCR 总是处理最新帧。

---

## 10. 关键行为约定

1. **本地模型首次启动有预热延迟**：ONNX 模型加载 ~2-5 秒，PaddleOCR 模型加载 ~5-15 秒。`warmup()` 在 `start()` 时自动调用。

2. **API 后端首次结果延迟**：第 1 次 API 调用约 1-3 秒后返回。调用期间不阻塞采集和流水线（`submit` / `poll` 异步模式）。

3. **无题板时不输出**：如果画面中没有检测到白色题板（`roi_found == False`），该帧被静默跳过，不会产生 `RecognitionOutput`。

4. **连续多帧稳定输出后切换场景**：如果题目变了，滤波器的缓冲区会被新结果覆盖并重新计数。旧结果不会输出。

5. **`result_ready` 信号量**：每次共识达成时会被 `set()`。父项目读取 `latest_output` 后应调用 `result_ready.clear()` 以便下次等待。

6. **线程安全**：`latest_output` 的读写受内部锁保护。但不要在回调函数中长时间阻塞，这会拖慢 OCR 线程。

---

## 11. CLI 调试入口（不变）

原有的命令行接口继续可用，适合调试和单独测试：

```bash
# ONNX 实时识别
python3 -m robocon_ocr camera --device-index 2

# 指定后端
python3 -m robocon_ocr camera --ocr-backend api

# 显示调试窗口
python3 -m robocon_ocr camera --show-window --show-stage-debug

# 停在中途看中间结果
python3 -m robocon_ocr camera --stop-after-stage enhancement --show-window

# 打印每一帧结果（而非仅变化时）
python3 -m robocon_ocr camera --print-all
```
