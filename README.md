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

## 当前建议：开始调表达式区域提取

如果题板检测和透视矫正都已经稳定，现在下一步建议只盯 `expression_region`。

原因很简单：

- 表达式区域提取决定了后面增强和 OCR 真正看到的内容
- 如果区域截太大，会把白边和噪声一起带入
- 如果区域截太小，会直接丢字符、断字符
- 现在项目已经支持按阶段截停，可以先把表达式区域单独调准

推荐先用这条命令做实时联调：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug \
  --stop-after-stage expression_region
```

这时程序只会执行到表达式区域提取阶段：

- 会持续抓摄像头画面
- 会在原图上画出当前题板四边形
- 会显示题板检测调试图
- 会显示当前提取出的表达式 bbox 裁切图
- 不会继续跑增强、分割和 OCR

推荐先观察这几件事：

1. 表达式区域是否完整包含首尾字符
2. 左右边界是否贴近表达式，而不是退回整张题板宽度
3. 上下左右是否只保留少量安全白边
4. 正视和斜视时，bbox 位置是否稳定
5. 失败时是不是候选太小、或者不够像单行文本

如果你想保存这一阶段的调试输出，推荐直接加上：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug \
  --stop-after-stage expression_region \
  --debug-dir debug_outputs \
  --debug-save-stages
```

如果你发现透视图本身不稳定，那就先退回上一阶段：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug \
  --stop-after-stage rectification
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

如果你当前目标是“开始调表达式区域提取”，优先用下面这条，而不是直接跑完整 OCR：

```bash
python3 -m robocon_ocr dataset/num_100_com_8 \
  --stop-after-stage expression_region \
  --debug-dir debug_outputs \
  --debug-save-stages
```

这样可以批量检查数据集里每张图的表达式 bbox 提取结果。

识别一个数据集目录：

```bash
python3 -m robocon_ocr dataset/num_100_com_4
```

保存调试图：

```bash
python3 -m robocon_ocr dataset/num_100_com_8 \
  --debug-dir debug_outputs
```

只跑到表达式区域提取阶段，并把各阶段调试图按阶段落盘：

```bash
python3 -m robocon_ocr dataset/num_100_com_8 \
  --stop-after-stage expression_region \
  --debug-dir debug_outputs \
  --debug-save-stages
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

如果你当前只想调表达式区域提取，最推荐的启动方式是：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug \
  --stop-after-stage expression_region
```

只有在表达式区域已经稳定之后，再去掉 `--stop-after-stage expression_region` 往后放开。

持续取流并实时输出识别结果：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug
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

如果你想限制抓图帧数，或者保存拍摄原图和分阶段调试图：

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
  --show-stage-debug \
  --stop-after-stage expression_region \
  --save-frame captures/latest_frame.png \
  --debug-dir debug_outputs \
  --debug-save-stages
```

打开窗口调试后，会显示：

- 彩色原图，并叠加 ROI 四边形
- 题板检测调试图
- 当前阶段主输出图
- 阶段状态 / OCR / 失败原因信息

其中题板检测阶段的调试信息会包含：

- 当前阈值
- 当前最佳候选的面积 / 边缘 / 比例测量值
- 当前失败原因，方便判断是哪个参数限制了检测

如果加上 `--stop-after-stage expression_region`：

- 只会执行到表达式区域提取
- 后续增强、分割、OCR 都不会调用
- 窗口左下角会显示当前提取出的表达式 bbox 裁切图
- 右下角会持续显示 bbox、候选数量、搜索窗和失败原因

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
- `--show-stage-debug`：显示分阶段 2x2 调试面板
- `--stop-after-stage`：在指定阶段后截停，当前支持 `board_detection / rectification / expression_region / enhancement / segmentation / ocr / postprocess`
- `--debug-save-stages`：将每个阶段的主输出图保存到 `--debug-dir`
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

如果你现在还在微调题板检测，优先编辑 [robocon_ocr/roi_tuning.py](/home/angela/Robocon/OCR/robocon_ocr/roi_tuning.py)。

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

调题板检测时的经验建议：

- 远距离明显漏检时，先降 `min_roi_area_ratio`
- 肉眼能看到边界但总说边缘太弱时，先降 `edge_threshold`
- 白板整体发灰、亮度不够时，再降 `white_threshold`
- 题板已经找到了，但总因为形状不够规整被拒绝，再放宽比例容差
- 如果透视图总带到题板外缘，检查 `roi_padding`；当前默认是 `-2`，表示向内收 2 像素

## 表达式区域提取阶段怎么看

开始调 `expression_region` 时，当前最重要的不是 OCR 对不对，而是“表达式区域 bbox 是否被完整、稳定地截出来”。

这一阶段会先尝试剥离透视图四周连续的黑色边框，再在内部白底区域上做上下左右状态机扫描。
黑像素判定不再用固定阈值，而是先在搜索窗内做两轮 Otsu，再用第二轮结果驱动四边状态机。

这一阶段建议重点观察：

- 表达式首尾字符是否都被保留
- 上下左右是否完整包住表达式
- bbox 是否明显小于整张透视图
- 四周是否只保留少量安全白边，量级约 10px
- 正视和斜视时，区域是否都落在正确位置
- 失败时是不是 `row top not found`、`row bottom not found`、`column left not found` 或 `column right not found`

如果表达式区域不对，优先回看两类原因：

1. `rectification` 本身图像内容就不稳定
2. 表达式区域提取参数太紧或太松

当前与表达式区域提取最相关的参数是：

- `expression_search_top_ratio`
- `expression_search_bottom_ratio`
- `expression_search_left_ratio`
- `expression_search_right_ratio`
- `expression_otsu_bias`
- `expression_enter_ratio`
- `expression_exit_ratio`
- `expression_min_consecutive_rows`
- `expression_min_consecutive_cols`
- `expression_bbox_padding_x`
- `expression_bbox_padding_y`

经验上：

- 如果上下经常裁掉字符，先增大 `expression_bbox_padding_y`
- 如果左右经常裁掉字符，先增大 `expression_bbox_padding_x`
- 如果总抓不到字符，先增大 `expression_otsu_bias` 或降低 `expression_enter_ratio`
- 如果边缘噪点老是提前触发，先提高 `expression_enter_ratio` 或增大 `expression_min_consecutive_rows`
- 如果左右边界抖动大，先增大 `expression_min_consecutive_cols`
- 如果需要看前景掩码和有效搜索窗，打开 `--debug-save-stages`，会额外保存 `*_expression_region_mask.png`

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

## 推荐的优化顺序

当前代码已经拆成严格串联的阶段流水线：

1. `board_detection`
2. `rectification`
3. `expression_region`
4. `enhancement`
5. `segmentation`
6. `ocr`
7. `postprocess`

推荐按下面顺序优化，不要跳步：

1. 先把题板四边形稳定提取出来
2. 再做透视矫正
3. 再把表达式区域稳定提取出来
4. 再做去摩尔纹 / 图像增强
5. 再尝试字符串或字符区域分割
6. 最后再压 OCR 识别率

这样后面的每一步都严格基于前一步的结构化结果，不会把前序问题藏到后面的 OCR 补偿逻辑里。

在联调时，建议现在先用：

```bash
python3 -m robocon_ocr camera \
  --show-window \
  --show-stage-debug \
  --stop-after-stage expression_region
```

先把实时单帧表达式区域提取调稳，再逐阶段往后放开。
