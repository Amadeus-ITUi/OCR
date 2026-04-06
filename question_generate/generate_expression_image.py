#!/usr/bin/env python3
import argparse
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    "Times New Roman",
    "Times New Roman Regular",
]

DISPLAY_WIDTH_MM = 1600
DISPLAY_HEIGHT_MM = 900

OUTER_WIDTH_PX = 1920
OUTER_HEIGHT_PX = 1080
SCREEN_WIDTH_PX = 1600
SCREEN_HEIGHT_PX = 900

MM_TO_PX = SCREEN_HEIGHT_PX / DISPLAY_HEIGHT_MM
MIN_FONT_MM = 30
MIN_FONT_PX = math.ceil(MIN_FONT_MM * MM_TO_PX)
START_FONT_PX = 120
MAX_LEAF_VALUE = 20
MAX_NODE_VALUE = 400

SYMBOL_MAP = {
    "+": "+",
    "-": "-",
    "*": "×",
    "/": "÷",
}

PRECEDENCE = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
}


@dataclass
class Node:
    value: int
    text: str
    precedence: int
    is_leaf: bool
    operator: str | None = None


@dataclass
class ComplexityConfig:
    complexity: int
    min_depth: int
    max_depth: int
    leaf_stop_chance: float
    extra_parentheses_chance: float
    mul_div_weight: int
    add_sub_weight: int


def find_times_new_roman(font_path: str | None) -> Path:
    if font_path:
        path = Path(font_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定字体文件不存在: {path}")
        return path

    for family in FONT_CANDIDATES:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            capture_output=True,
            text=True,
            check=False,
        )
        candidate = result.stdout.strip()
        if candidate and Path(candidate).is_file():
            lower_name = Path(candidate).name.lower()
            if "times" in lower_name:
                return Path(candidate)

    raise FileNotFoundError(
        "系统中未找到 Times New Roman 字体。\n"
        "Ubuntu 可尝试执行:\n"
        "  sudo apt-get update\n"
        "  sudo apt-get install -y ttf-mscorefonts-installer\n"
        "也可以通过 --font-path 显式指定 Times New Roman 的 .ttf 文件。"
    )


def maybe_wrap(
    child: Node,
    parent_op: str,
    is_right_child: bool,
    rng: random.Random,
    extra_parentheses_chance: float,
) -> str:
    if child.is_leaf:
        return child.text

    child_prec = child.precedence
    parent_prec = PRECEDENCE[parent_op]
    needs_parentheses = child_prec < parent_prec

    if is_right_child and parent_op in {"-", "/"} and child_prec == parent_prec:
        needs_parentheses = True

    if not needs_parentheses and rng.random() < extra_parentheses_chance:
        needs_parentheses = True

    if needs_parentheses:
        return f"({child.text})"
    return child.text


def combine_nodes(
    left: Node,
    right: Node,
    op: str,
    rng: random.Random,
    extra_parentheses_chance: float,
) -> Node | None:
    if op == "+":
        value = left.value + right.value
    elif op == "-":
        if left.value < right.value:
            return None
        value = left.value - right.value
    elif op == "*":
        value = left.value * right.value
    elif op == "/":
        if right.value == 0 or left.value % right.value != 0:
            return None
        value = left.value // right.value
    else:
        return None

    if value < 0 or value > MAX_NODE_VALUE:
        return None

    left_text = maybe_wrap(left, op, False, rng, extra_parentheses_chance)
    right_text = maybe_wrap(right, op, True, rng, extra_parentheses_chance)
    text = f"{left_text} {SYMBOL_MAP[op]} {right_text}"
    return Node(
        value=value,
        text=text,
        precedence=PRECEDENCE[op],
        is_leaf=False,
        operator=op,
    )


def generate_leaf(rng: random.Random) -> Node:
    value = rng.randint(1, MAX_LEAF_VALUE)
    return Node(value=value, text=str(value), precedence=3, is_leaf=True)


def choose_operator(rng: random.Random, config: ComplexityConfig) -> str:
    operators = ["+", "-", "*", "/"]
    weights = [
        config.add_sub_weight,
        config.add_sub_weight,
        config.mul_div_weight,
        config.mul_div_weight,
    ]
    return rng.choices(operators, weights=weights, k=1)[0]


def build_complexity_config(complexity: int) -> ComplexityConfig:
    clamped = max(1, min(10, complexity))
    min_depth = 1 + clamped // 3
    max_depth = min(5, min_depth + 1 + clamped // 4)
    leaf_stop_chance = max(0.08, 0.48 - clamped * 0.04)
    extra_parentheses_chance = min(0.60, 0.04 + clamped * 0.05)
    mul_div_weight = 1 + clamped
    add_sub_weight = max(1, 12 - clamped)
    return ComplexityConfig(
        complexity=clamped,
        min_depth=min_depth,
        max_depth=max_depth,
        leaf_stop_chance=leaf_stop_chance,
        extra_parentheses_chance=extra_parentheses_chance,
        mul_div_weight=mul_div_weight,
        add_sub_weight=add_sub_weight,
    )


def generate_node(rng: random.Random, depth: int, config: ComplexityConfig) -> Node:
    if depth <= 0 or rng.random() < config.leaf_stop_chance:
        return generate_leaf(rng)

    for _ in range(300):
        left = generate_node(rng, depth - 1, config)
        right = generate_node(rng, depth - 1, config)
        node = combine_nodes(
            left,
            right,
            choose_operator(rng, config),
            rng,
            config.extra_parentheses_chance,
        )
        if node is not None:
            return node

    return generate_leaf(rng)


def generate_problem(
    rng: random.Random,
    min_depth: int,
    max_depth: int,
    config: ComplexityConfig,
) -> tuple[str, int]:
    for _ in range(500):
        depth = rng.randint(min_depth, max_depth)
        node = generate_node(rng, depth, config)
        if node.operator is None:
            continue
        return f"{node.text} =", node.value
    raise RuntimeError("生成算式失败，请放宽参数后重试。")


def fit_font_size(
    text: str,
    font_path: Path,
    text_area_width: int,
    text_area_height: int,
) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int], int]:
    dummy = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy)

    for font_size in range(START_FONT_PX, MIN_FONT_PX - 1, -2):
        font = ImageFont.truetype(str(font_path), font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= text_area_width and height <= text_area_height:
            return font, bbox, font_size

    raise ValueError(
        f"当前算式无法在单行内放入显示区，最小字号 {MIN_FONT_PX}px 也超限: {text}"
    )


def render_problem_image(
    question_text: str,
    output_path: Path,
    font_path: Path,
    outer_width: int,
    outer_height: int,
    screen_width: int,
    screen_height: int,
    border_color: str,
    screen_color: str,
    text_color: str,
    inner_padding_x: int,
    inner_padding_y: int,
) -> int:
    image = Image.new("RGB", (outer_width, outer_height), border_color)
    draw = ImageDraw.Draw(image)

    screen_left = (outer_width - screen_width) // 2
    screen_top = (outer_height - screen_height) // 2
    screen_right = screen_left + screen_width
    screen_bottom = screen_top + screen_height

    draw.rectangle(
        [(screen_left, screen_top), (screen_right, screen_bottom)],
        fill=screen_color,
    )

    text_area_width = screen_width - inner_padding_x * 2
    text_area_height = screen_height - inner_padding_y * 2
    font, bbox, font_size = fit_font_size(
        text=question_text,
        font_path=font_path,
        text_area_width=text_area_width,
        text_area_height=text_area_height,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = screen_left + (screen_width - text_width) / 2 - bbox[0]
    text_y = screen_top + (screen_height - text_height) / 2 - bbox[1]
    draw.text((text_x, text_y), question_text, fill=text_color, font=font)

    image.save(output_path)
    return font_size


def write_manifest(entries: list[tuple[str, str, int, int]], output_dir: Path) -> Path:
    manifest_path = output_dir / "problems_and_answers.txt"
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write("filename\texpression\tanswer\tfont_size_px\n")
        for filename, expression, answer, font_size in entries:
            fh.write(f"{filename}\t{expression}\t{answer}\t{font_size}\n")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量生成 Times New Roman 四则运算题目图片和答案清单。"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="generated_problems",
        help="输出目录，默认: generated_problems",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=20,
        help="生成题目数量，默认: 20",
    )
    parser.add_argument(
        "--font-path",
        help="Times New Roman 字体文件路径；未提供时自动从系统查找。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子，便于复现生成结果。",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=3,
        help="题目复杂度，范围 1-10；值越大，括号和乘除越多，结构越深，默认: 3",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        help="算式最小递归深度；未指定时根据 --complexity 自动决定",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="算式最大递归深度；未指定时根据 --complexity 自动决定",
    )
    parser.add_argument(
        "--outer-width",
        type=int,
        default=OUTER_WIDTH_PX,
        help=f"黑色底板宽度，默认: {OUTER_WIDTH_PX}",
    )
    parser.add_argument(
        "--outer-height",
        type=int,
        default=OUTER_HEIGHT_PX,
        help=f"黑色底板高度，默认: {OUTER_HEIGHT_PX}",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=SCREEN_WIDTH_PX,
        help=f"白色显示区宽度，默认: {SCREEN_WIDTH_PX}",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=SCREEN_HEIGHT_PX,
        help=f"白色显示区高度，默认: {SCREEN_HEIGHT_PX}",
    )
    parser.add_argument(
        "--inner-padding-x",
        type=int,
        default=80,
        help="显示区左右留白，默认: 80",
    )
    parser.add_argument(
        "--inner-padding-y",
        type=int,
        default=110,
        help="显示区上下留白，默认: 110",
    )
    parser.add_argument(
        "--border-color",
        default="black",
        help="黑色底板颜色，默认: black",
    )
    parser.add_argument(
        "--screen-color",
        default="white",
        help="显示区颜色，默认: white",
    )
    parser.add_argument(
        "--text-color",
        default="black",
        help="文字颜色，默认: black",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成题目与答案清单，不输出图片，便于先验证题目逻辑。",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count 必须大于 0")
    if args.complexity < 1 or args.complexity > 10:
        raise ValueError("--complexity 必须在 1 到 10 之间")
    if args.min_depth is not None and args.min_depth < 1:
        raise ValueError("--min-depth 必须大于等于 1")
    if args.max_depth is not None and args.min_depth is not None and args.max_depth < args.min_depth:
        raise ValueError("请保证 1 <= --min-depth <= --max-depth")
    if args.screen_width >= args.outer_width or args.screen_height >= args.outer_height:
        raise ValueError("白色显示区必须小于黑色底板尺寸")
    if args.inner_padding_x * 2 >= args.screen_width:
        raise ValueError("左右留白过大，已超过显示区宽度")
    if args.inner_padding_y * 2 >= args.screen_height:
        raise ValueError("上下留白过大，已超过显示区高度")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        font_path = None if args.dry_run else find_times_new_roman(args.font_path)
        rng = random.Random(args.seed)
        complexity_config = build_complexity_config(args.complexity)
        min_depth = (
            args.min_depth if args.min_depth is not None else complexity_config.min_depth
        )
        max_depth = (
            args.max_depth if args.max_depth is not None else complexity_config.max_depth
        )
        if max_depth < min_depth:
            raise ValueError("请保证 1 <= --min-depth <= --max-depth")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[tuple[str, str, int, int]] = []

        for index in range(1, args.count + 1):
            for _ in range(500):
                question_text, answer = generate_problem(
                    rng=rng,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    config=complexity_config,
                )
                filename = f"problem_{index:04d}.png"

                if args.dry_run:
                    manifest_entries.append((filename, question_text, answer, 0))
                    break

                try:
                    font_size = render_problem_image(
                        question_text=question_text,
                        output_path=output_dir / filename,
                        font_path=font_path,
                        outer_width=args.outer_width,
                        outer_height=args.outer_height,
                        screen_width=args.screen_width,
                        screen_height=args.screen_height,
                        border_color=args.border_color,
                        screen_color=args.screen_color,
                        text_color=args.text_color,
                        inner_padding_x=args.inner_padding_x,
                        inner_padding_y=args.inner_padding_y,
                    )
                except ValueError:
                    continue

                manifest_entries.append((filename, question_text, answer, font_size))
                break
            else:
                raise RuntimeError(f"第 {index} 题在当前版式约束下生成失败，请放宽参数后重试。")

        manifest_path = write_manifest(manifest_entries, output_dir)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"输出目录: {output_dir}")
    print(f"清单文件: {manifest_path}")
    print(
        f"复杂度: {complexity_config.complexity} "
        f"(depth {min_depth}-{max_depth}, 额外括号概率 {complexity_config.extra_parentheses_chance:.2f}, "
        f"乘除权重 {complexity_config.mul_div_weight})"
    )
    if not args.dry_run:
        print(f"字体文件: {font_path}")
        print(
            f"版式基准: 显示区 {DISPLAY_WIDTH_MM}mm x {DISPLAY_HEIGHT_MM}mm, "
            f"最小字高约 {MIN_FONT_MM}mm -> {MIN_FONT_PX}px"
        )
    print(f"生成数量: {args.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
