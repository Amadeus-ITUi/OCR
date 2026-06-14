# OCR 项目清理与 ONNX 迁移计划

## 任务一：清理 pix2tex 链路 ✅ 已完成

### 清理内容

| 操作 | 文件/目录 |
|------|----------|
| 删除 | `.venv/` (Python 3.14 pix2tex 虚拟环境) |
| 删除 | `requirements.txt` (pix2tex 专用依赖) |
| 删除 | `robocon_ocr/image_recognition/pix2tex_recognizer.py` |
| 修改 | `robocon_ocr/image_recognition/factory.py` — 移除 `"pix2tex"` 分支 |
| 修改 | `robocon_ocr/config.py` — 移除 `model_path` 字段，`backend` 默认值改为 `"lightweight"` |
| 修改 | `robocon_ocr/cli.py` — `OCR_BACKEND_CHOICES` 移除 `"pix2tex"`，两模式均默认 `"lightweight"` |
| 修改 | `robocon_ocr/staged_pipeline.py` — 移除 `recognize_with_fallback_variants` 和 `_select_best_result`，`_OCRStage` 简化为单次识别 |
| 修改 | `robocon_ocr/pipeline.py` — 移除 `_recognize_with_fallback_variants`、`_select_best_result`、`_record_from_preprocess` 等死代码，删除无用 import |
| 删除 | `tests/test_pix2tex_recognizer.py` |
| 修改 | `tests/test_ocr_backends.py` — 移除 pix2tex 测试用例和 import |
| 修改 | `tests/test_cli.py` — 更新 pix2tex 引用，子进程测试 skipif 从 `pix2tex` 改为 `paddleocr` |
| 修改 | `tests/test_pipeline_integration.py` — 移除 fallback 变体测试和 `_recognize_with_fallback_variants` import |
| 修改 | `README.md` — 移除所有 pix2tex 相关说明，简化为纯 PaddleOCR 文档 |

### 验证结果

```
pytest tests/ -v  →  55 passed, 0 failures
dataset num_10_com_1  →  10/10 expression_match, 10/10 answer_match
```

PaddleOCR 链路完全不受影响。

---

## 任务二：PaddleOCR 转 ONNX 及推理封装（待执行）

### 目标

将 `PP-OCRv5_server_rec` 模型从 PaddlePaddle 格式转换为 ONNX，编写包含预处理 + ONNX 推理 + CTC 解码的推理封装，作为新的 OCR 后端（`onnx` backend），在离线数据集上验证精度与 PaddleOCR 原生一致。

### 技术要点

**PaddleOCR `TextRecognition` 内部流程：**

```
输入 PIL Image (RGB)
  → DecodeImage: 保持 HWC BGR
  → RecResizeImg: resize 到 48px 高，保持宽高比，最大宽 320px
  → NormalizeImage: (pixel - 127.5) / 127.5
  → HWC → CHW
  → 模型前向 → CTC logits [T, batch, num_classes]
  → CTCLabelDecode: argmax per frame → 去重连续相同 → 去 blank → 输出文本
```

导出 ONNX 仅包含模型前向部分，预处理和 CTC 解码需自行实现且必须精确对齐。

### 操作步骤

#### 阶段 A：导出 ONNX 模型

1. 在 `.venv-paddle` 环境中安装 `paddle2onnx`
2. 模型文件位置：`~/.paddlex/official_models/PP-OCRv5_server_rec/`
   - `inference.pdmodel` — 模型结构
   - `inference.pdiparams` — 权重
3. 导出：
   ```bash
   paddle2onnx --model_dir ~/.paddlex/official_models/PP-OCRv5_server_rec/ \
               --model_filename inference.pdmodel \
               --params_filename inference.pdiparams \
               --save_file models/PP-OCRv5_server_rec.onnx \
               --opset_version 14
   ```
4. 验证 ONNX 模型

#### 阶段 B：编写 ONNX 推理封装

1. **新建 `robocon_ocr/image_recognition/onnx_recognizer.py`**
   - `OnnxMathRecognizer` 类，`backend_name = "onnx"`, `supports_fallback_variants = False`

2. **预处理模块**（与 PaddleOCR 精确对齐）：
   - PIL RGB → BGR numpy
   - Resize：高 48px，等比缩放，宽上限 320px
   - 归一化：`(x - 127.5) / 127.5`
   - HWC → CHW + batch dim

3. **CTC 解码模块**：
   - squeeze batch → 逐帧 argmax → 合并相邻相同 → 过滤 blank → 查字典
   - 置信度 = 非 blank 帧的最大概率均值

4. **字符字典**：从模型目录提取，导出到 `models/dict.txt`

5. **注册**：factory.py 添加 `"onnx"` 分支，config.py 添加 `onnx_model_path` / `onnx_dict_path`，cli.py 添加 `"onnx"` 选项

#### 阶段 C：精度验证

- 用 `num_100_com_4` 和 `num_100_com_8` 对比 PaddleOCR 原生与 ONNX 后端
- 目标：表达式匹配率和答案匹配率差异 < 1%
- 结果保存到 `docs/onnx-vs-paddle-benchmark.txt`

#### 阶段 D：测试

- 新建 `tests/test_onnx_recognizer.py`
- 测试预处理 shape、CTC 解码、工厂分发、标准化 / charset 验证

### 关键文件

| 文件 | 操作 |
|------|------|
| `robocon_ocr/image_recognition/onnx_recognizer.py` | **新建** |
| `robocon_ocr/image_recognition/factory.py` | 修改：添加 `"onnx"` 分支 |
| `robocon_ocr/config.py` | 修改：添加 `onnx_model_path`、`onnx_dict_path` |
| `robocon_ocr/cli.py` | 修改：添加 `"onnx"` 选项 |
| `models/PP-OCRv5_server_rec.onnx` | **新建** |
| `models/dict.txt` | **新建** |
| `tests/test_onnx_recognizer.py` | **新建** |
| `docs/onnx-vs-paddle-benchmark.txt` | **新建** |

---

## 后续部署路线

```
x86: PaddleOCR → ONNX 导出 → ONNX RT 验证精度
  ↓
Jetson Docker: ONNX RT CPU 先跑通
  ↓
Jetson Docker: ONNX → TensorRT FP16 GPU 加速
  ↓
ROS2 节点封装 → 合入主项目
```
