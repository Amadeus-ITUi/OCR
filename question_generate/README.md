# Question Generate

这个目录用于批量生成四则运算题目图片，并同时输出题目与答案清单。

脚本文件：

- `generate_expression_image.py`

## 功能说明

这个脚本会批量生成满足以下规则的题目图片：

- 字体使用 `Times New Roman`
- 图片整体为黑色底板，中间为白色 16:9 显示区
- 题目为单行显示，不换行
- 文字为黑色
- 题目只包含整数，不包含小数和负数
- 支持加、减、乘、除和小括号
- 题目末尾带等号 `=`
- 乘号显示为 `×`
- 除号显示为 `÷`
- 同时输出一个文本清单，保存图片文件名、算式、答案和实际字号

## 输出内容

执行脚本后，指定输出目录下会生成：

- `problem_0001.png`、`problem_0002.png` 等题目图片
- `problems_and_answers.txt` 题目与答案清单

`problems_and_answers.txt` 的列格式如下：

```text
filename    expression    answer    font_size_px
```

## 复杂度参数

脚本支持 `--complexity` 参数，范围是 `1-10`。

- 数值越小，题目越简单
- 数值越大，题目越复杂
- 复杂度升高后，会同时提升：
  - 表达式层数
  - 小括号出现概率
  - 乘号和除号出现概率

例如：

- `--complexity 2` 更接近简单加减题
- `--complexity 8` 会更容易出现多层括号和乘除混合运算

## 主要代码说明

脚本的核心逻辑主要分为四部分：

### 1. 字体查找

`find_times_new_roman()` 会查找系统中的 `Times New Roman` 字体文件。如果没有找到，会直接报错，避免误用其他字体。

### 2. 题目生成

`generate_node()` 和 `generate_problem()` 负责递归构造表达式树，并生成最终算式。

规则包括：

- 只生成整数结果
- 除法必须整除
- 减法不会产生负数
- 表达式末尾统一补上等号

### 3. 复杂度控制

`build_complexity_config()` 会根据 `--complexity` 生成一组内部配置，用来控制：

- 递归深度范围
- 提前停止扩展为数字叶子的概率
- 额外添加小括号的概率
- 乘除与加减的权重比例

### 4. 图片渲染

`render_problem_image()` 会把题目渲染到图片中。

渲染方式：

- 外层黑色底板
- 中间白色显示区
- 自动计算字号
- 保证题目尽量在单行内放下
- 文字在显示区中居中

## 常用命令

### 生成 20 道题

```bash
python3 /home/terrisa/Robocon/OCR/question_generate/generate_expression_image.py \
  -n 20 \
  -o /home/terrisa/Robocon/OCR/question_generate/output_20
```

### 生成 100 道中等复杂度题目

```bash
python3 /home/terrisa/Robocon/OCR/question_generate/generate_expression_image.py \
  -n 100 \
  -o /home/terrisa/Robocon/OCR/question_generate/output_100 \
  --complexity 5
```

### 生成高复杂度题目

```bash
python3 /home/terrisa/Robocon/OCR/question_generate/generate_expression_image.py \
  -n 100 \
  -o /home/terrisa/Robocon/OCR/dataset/num_100_com_8 \
  --complexity 8
```

### 指定随机种子，便于复现

```bash
python3 /home/terrisa/Robocon/OCR/question_generate/generate_expression_image.py \
  -n 30 \
  -o /home/terrisa/Robocon/OCR/question_generate/output_seed_42 \
  --complexity 6 \
  --seed 42
```

### 只测试题目生成，不输出图片

```bash
python3 /home/terrisa/Robocon/OCR/question_generate/generate_expression_image.py \
  -n 20 \
  -o /home/terrisa/Robocon/OCR/question_generate/output_dry_run \
  --complexity 7 \
  --dry-run
```

## 可选参数补充

除了 `--complexity`，还支持一些版式相关参数：

- `--outer-width` / `--outer-height`：黑色底板尺寸
- `--screen-width` / `--screen-height`：白色显示区尺寸
- `--inner-padding-x` / `--inner-padding-y`：显示区内边距
- `--font-path`：手动指定 `Times New Roman` 字体文件路径
- `--min-depth` / `--max-depth`：手动覆盖自动复杂度深度范围

## 目录建议

推荐使用方式：

- 脚本放在 `question_generate/`
- 每次生成题目时把输出放到单独的子目录中，便于管理不同批次结果

例如：

```text
question_generate/
├── generate_expression_image.py
├── README.md
├── output_20/
├── output_100/
└── output_hard/
```
