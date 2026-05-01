# Robocon OCR

这是一个面向 RoboCon 四则运算题目的离线识别项目骨架。

当前阶段只实现：

- `dataset` 图片作为输入
- `Tesseract OCR` 离线识别题目
- 对识别结果做规则化、表达式求值、答案输出
- 用标注文件评估识别准确率

暂未实现：

- 视觉采集层真实摄像头接入
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
  -> 图片识别层
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

1. 从整张图里裁出公式前景区域
2. 做轻量二值化和放大
3. 送给 `Tesseract OCR`
4. 用规则层把 OCR 输出修正为合法算式
5. 再独立计算答案

这样系统会比“直接信 OCR 原始文本”稳很多。

## 安装建议

建议单独创建虚拟环境，并固定 `numpy<2`，避免目前很多 OCR 相关包与 `numpy 2.x` 的兼容问题。



```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

除了 Python 依赖，还需要在系统里安装 `tesseract` 可执行程序。

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

macOS:

```bash
brew install tesseract
```

如果 `tesseract` 不在系统 `PATH` 里，可以在 `OCRConfig.tesseract_cmd` 里手动指定可执行文件路径。
语言建议使用 Tesseract 的语言码，例如英文应写成 `eng`，不是 `en`。

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
