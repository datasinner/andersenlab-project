"""Number handling shared by the rule engine and the grounding check.

Tariff documents print numbers in many local styles: "30 960.46" (space as
thousands separator, sometimes doubled by PDF extraction: "2  801.91"),
"1,654.56", "0,65" (decimal comma), "1.234,56", "35%". Everything numeric in
the system goes through parse_number so that "the same number" means the
same thing everywhere.
"""

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

CENT = Decimal("0.01")

# Characters PDFs use as thousands separators: space, no-break space,
# thin space and narrow no-break space.
_GROUP_SPACE_CHARS = "    "
_GROUP_SPACES = re.compile(f"[{_GROUP_SPACE_CHARS}]+")

# A run of digits, separators and group spaces that starts and ends with a
# digit. Newlines are deliberately not included: two numbers on separate
# lines are two numbers.
_NUMBER_RUN = re.compile(rf"\d(?:[\d.,{_GROUP_SPACE_CHARS}]*\d)?")
_THREE_DIGIT_GROUP = re.compile(r"\d{3}(?:[.,]\d+)?")
_COMMA_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+")


class NumberFormatError(ValueError):
    pass


def parse_number(value: str | int | Decimal) -> Decimal:
    """Parse one number as printed in a tariff document.

    A single "." is a decimal point and a single "," followed by exactly
    three digits is a thousands separator ("1,654" is 1654, "0,65" is 0.65).
    When both appear, whichever comes last is the decimal separator.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise NumberFormatError(f"Not a number: {value!r}")
    if isinstance(value, int):
        return Decimal(value)

    text = _GROUP_SPACES.sub("", value.strip())
    if text.endswith("%"):
        text = text[:-1]

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if _COMMA_THOUSANDS.fullmatch(text):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")

    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise NumberFormatError(f"Not a number: {value!r}") from exc
    if not number.is_finite():
        raise NumberFormatError(f"Not a number: {value!r}")
    return number


def extract_numbers(text: str) -> set[Decimal]:
    """Every number that could be read out of `text`.

    Where spacing is ambiguous ("Up to 2 000 0.50" could be 2000 and 0.50, or
    2, 0 and 0.50) all readings are included. The grounding check only asks
    "does this number appear?", so a generous superset is the safe side.
    """
    numbers: set[Decimal] = set()
    for run in _NUMBER_RUN.findall(text):
        pieces = _GROUP_SPACES.split(run)
        for start in range(len(pieces)):
            for end in range(start, len(pieces)):
                candidate = _join_digit_groups(pieces[start : end + 1])
                if candidate is None:
                    continue
                try:
                    numbers.add(parse_number(candidate))
                except NumberFormatError:
                    continue
    return numbers


def _join_digit_groups(pieces: list[str]) -> str | None:
    """Join space-separated pieces into one number if they form valid
    thousands groups ("30" + "960.46"), otherwise None."""
    if len(pieces) == 1:
        return pieces[0]
    for piece in pieces[:-1]:
        if "." in piece or "," in piece:
            return None
    for piece in pieces[1:]:
        if not _THREE_DIGIT_GROUP.fullmatch(piece):
            return None
    return "".join(pieces)


def round_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def format_number(value: Decimal) -> str:
    """Human-readable number for formula traces: thousands separators, no
    trailing zeros ("51,300", "3.39", "100,500.8553")."""
    if value == value.to_integral_value():
        return f"{value.to_integral_value():,f}"
    return f"{value.normalize():,f}"


def format_money(value: Decimal) -> str:
    return f"{round_money(value):,.2f}"
