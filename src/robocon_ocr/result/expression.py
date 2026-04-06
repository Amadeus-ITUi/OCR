from __future__ import annotations

from dataclasses import dataclass


ALLOWED_CHARS = set("0123456789+-×÷()= ")
SYMBOL_REPLACEMENTS = {
    "x": "×",
    "X": "×",
    "*": "×",
    "✖": "×",
    "✖️": "×",
    "／": "÷",
    "/": "÷",
    "—": "-",
    "−": "-",
    "（": "(",
    "）": ")",
    "[": "(",
    "]": ")",
}


@dataclass(slots=True)
class ParsedExpression:
    normalized_text: str
    expression: str
    answer: int | None
    is_valid: bool
    error: str | None = None


def normalize_ocr_text(text: str) -> str:
    normalized = text.strip()
    for src, dst in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    normalized = "".join(ch for ch in normalized if ch in ALLOWED_CHARS)
    normalized = " ".join(normalized.split())
    return normalized


def to_expression_only(text: str) -> str:
    cleaned = normalize_ocr_text(text)
    cleaned = cleaned.replace(" ", "")
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[0]
    return cleaned


def tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    number = []
    for ch in expression:
        if ch.isdigit():
            number.append(ch)
            continue
        if number:
            tokens.append("".join(number))
            number.clear()
        tokens.append(ch)
    if number:
        tokens.append("".join(number))
    return tokens


class ExpressionParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def current(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def consume(self, expected: str | None = None) -> str:
        token = self.current()
        if token is None:
            raise ValueError("unexpected end of expression")
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.index += 1
        return token

    def parse(self) -> int:
        value = self.parse_add_sub()
        if self.current() is not None:
            raise ValueError(f"unexpected token {self.current()!r}")
        return value

    def parse_add_sub(self) -> int:
        value = self.parse_mul_div()
        while self.current() in {"+", "-"}:
            op = self.consume()
            rhs = self.parse_mul_div()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_mul_div(self) -> int:
        value = self.parse_primary()
        while self.current() in {"×", "÷"}:
            op = self.consume()
            rhs = self.parse_primary()
            if op == "×":
                value *= rhs
            else:
                if rhs == 0:
                    raise ValueError("division by zero")
                if value % rhs != 0:
                    raise ValueError("non-integer division")
                value //= rhs
        return value

    def parse_primary(self) -> int:
        token = self.current()
        if token is None:
            raise ValueError("unexpected end of expression")
        if token == "(":
            self.consume("(")
            value = self.parse_add_sub()
            self.consume(")")
            return value
        if token.isdigit():
            return int(self.consume())
        raise ValueError(f"unexpected token {token!r}")


def parse_expression(text: str) -> ParsedExpression:
    normalized = normalize_ocr_text(text)
    expression = to_expression_only(normalized)

    if not expression:
        return ParsedExpression(
            normalized_text=normalized,
            expression="",
            answer=None,
            is_valid=False,
            error="empty expression",
        )

    try:
        tokens = tokenize(expression)
        answer = ExpressionParser(tokens).parse()
    except ValueError as exc:
        return ParsedExpression(
            normalized_text=normalized,
            expression=expression,
            answer=None,
            is_valid=False,
            error=str(exc),
        )

    return ParsedExpression(
        normalized_text=normalized,
        expression=expression,
        answer=answer,
        is_valid=True,
    )

