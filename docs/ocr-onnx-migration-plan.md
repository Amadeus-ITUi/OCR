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

## 任务二：PaddleOCR 转 ONNX 及推理封装 ✅ 已完成

### 实施摘要

PP-OCRv5_server_rec 模型已成功导出为 ONNX 格式，OnnxMathRecognizer 已实现并集成到项目中。

### 实际技术发现

模型是 PaddleX 3.x PIR 格式（`inference.json` + `inference.pdiparams`），非传统 `.pdmodel` 格式。paddle2onnx v2.1.0 原生支持 PIR 格式。

ONNX 模型输出 class 0 为 CTC blank token，class i (1-based) 映射到 character_dict[i-1]。模型有 18385 个输出类，字典有 18383 个字符。

### 执行结果

| 步骤 | 状态 |
|------|------|
| paddle2onnx 导出（opset 14） | 完成，模型从 1133 → 511 节点（常量折叠） |
| `onnx_recognizer.py` 实现 | 完成：预处理 + ONNX RT 推理 + CTC greedy decode |
| 字符字典提取 | 完成：`models/dict.txt`（18383 字符，index 0 = blank） |
| 工厂/配置/CLI 注册 | 完成：`--ocr-backend onnx` 可用 |
| 精度验证 | 完成：200/200 精确匹配（100%） |
| 测试覆盖 | 完成：15 个新增测试，70/70 全通过 |

### 精度对比

见 [docs/onnx-vs-paddle-benchmark.txt](onnx-vs-paddle-benchmark.txt)

| 数据集 | 图片数 | 匹配率 |
|--------|--------|--------|
| num_100_com_4 | 100 | 100% |
| num_100_com_8 | 100 | 100% |
| **合计** | **200** | **100%** |

### 修改的文件清单

| 文件 | 操作 |
|------|------|
| `robocon_ocr/image_recognition/onnx_recognizer.py` | **新建** |
| `robocon_ocr/image_recognition/factory.py` | 修改：添加 `"onnx"` 分支 |
| `robocon_ocr/config.py` | 修改：添加 `onnx_model_path`、`onnx_dict_path` |
| `robocon_ocr/cli.py` | 修改：`OCR_BACKEND_CHOICES` 添加 `"onnx"` |
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
