from robocon_ocr.result.expression import normalize_ocr_text, parse_expression, validate_ocr_text


def test_parse_simple_expression():
    parsed = parse_expression("7 + 5 =")
    assert parsed.is_valid
    assert parsed.expression == "7+5"
    assert parsed.answer == 12


def test_parse_times_and_division():
    parsed = parse_expression("(15 × 16 ÷ 2) + 2 - (4 × 15 - 16) =")
    assert parsed.is_valid
    assert parsed.answer == 78


def test_parse_mixed_ocr_symbols():
    parsed = parse_expression("((18 + 6) / (9 + 3) x 8) =")
    assert parsed.is_valid
    assert parsed.expression == "((18+6)÷(9+3)×8)"
    assert parsed.answer == 16


def test_invalid_non_integer_division():
    parsed = parse_expression("5 ÷ 2 =")
    assert not parsed.is_valid
    assert parsed.answer is None


def test_repair_duplicate_operator_sequence():
    parsed = parse_expression("7 + 11 × × 14 =")
    assert parsed.is_valid
    assert parsed.expression == "7+11×14"
    assert parsed.answer == 161


def test_repair_unmatched_parentheses():
    parsed = parse_expression("(6 + 3 + (4 × 3)) × 8) - (4 + 4) + 8 + 18) =")
    assert parsed.is_valid
    assert parsed.expression == "(6+3+(4×3))×8-(4+4)+8+18"
    assert parsed.answer == 186


def test_normalize_latex_math_symbols():
    assert normalize_ocr_text("3 \\times 4 =") == "3×4="
    assert normalize_ocr_text("\\left(1+2\\right) \\div 3") == "(1+2)÷3"


def test_reject_unsupported_symbol_outside_charset():
    normalized = normalize_ocr_text("\\frac{1}{2}")
    assert normalized == "12"

    parsed = parse_expression("\\frac{1}{2}")
    assert not parsed.is_valid
    assert parsed.error == "unsupported symbol outside arithmetic charset"


def test_extract_arithmetic_expression_from_noisy_pix2tex_latex():
    parsed = parse_expression(
        "\\left(\\begin{array}{l l}{{}}&{{}}\\\\ {{}}&{{}}\\\\ {{}}&{{2+9\\times10\\times3\\cdot7=}}\\end{array}\\right)"
    )
    assert parsed.is_valid
    assert parsed.expression == "2+9×10×3×7"
    assert parsed.answer == 1892
