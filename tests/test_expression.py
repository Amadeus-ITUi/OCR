from robocon_ocr.result.expression import parse_expression


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
