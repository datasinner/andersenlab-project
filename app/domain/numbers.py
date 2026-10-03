"""Number handling shared by the rule engine and the grounding check.

Tariff documents print numbers in many local styles: "12 345.67" (space as
thousands separator, sometimes doubled by PDF extraction: "4  512.30"),
"1,234.50", "0,75" (decimal comma), "1.234,56", "35%". Everything numeric in
the system goes through parse_number so that "the same number" means the
same thing everywhere. Small numbers are often written as words ("beyond the
fifth day", "within seven days"); extract_numbers reads those too.
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

# Numbers written as words: cardinals and ordinals up to ninety-nine
# ("twenty-four", "twenty-fourth"), optionally times a hundred or thousand.
_UNIT_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS_WORDS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_IRREGULAR_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_LETTERS = re.compile(r"[^\W\d_]+")
_MULTIPLIER_WORDS = {"hundred": 100, "thousand": 1000}


def _ordinal(word: str) -> str:
    if word in _IRREGULAR_ORDINALS:
        return _IRREGULAR_ORDINALS[word]
    return word[:-1] + "ieth" if word.endswith("y") else word + "th"


_UNIT_VALUES = {word: n for n, word in enumerate(_UNIT_WORDS)} | {
    _ordinal(word): n for n, word in enumerate(_UNIT_WORDS) if n
}
_TENS_VALUES = {word: 10 * n for n, word in enumerate(_TENS_WORDS, start=2)}
_WORD_VALUES = _UNIT_VALUES | _TENS_VALUES | {_ordinal(w): v for w, v in _TENS_VALUES.items()}


class NumberFormatError(ValueError):
    pass


def parse_number(value: str | int | Decimal) -> Decimal:
    """Parse one number as printed in a tariff document.

    A single "." is a decimal point and a single "," followed by exactly
    three digits is a thousands separator ("1,234" is 1234, "0,75" is 0.75).
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

    Where spacing is ambiguous ("Up to 3 000 0.75" could be 3000 and 0.75, or
    3, 0 and 0.75) all readings are included, and so are numbers written as
    words ("the fifth day" gives 5). The grounding check only asks "does this
    number appear?", so a generous superset is the safe side.
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
    return numbers | _word_numbers(text)


def _word_numbers(text: str) -> set[Decimal]:
    """Numbers written as words, each word on its own and as a compound
    ("twenty-four" gives 20, 4 and 24; "two hundred" gives 2, 100 and 200).
    casefold() also undoes typographic ligatures ("ﬁfth")."""
    words = _LETTERS.findall(text.casefold())
    numbers = {Decimal(_MULTIPLIER_WORDS[word]) for word in words if word in _MULTIPLIER_WORDS}
    for index, word in enumerate(words):
        if word not in _WORD_VALUES:
            continue
        value = _WORD_VALUES[word]
        numbers.add(Decimal(value))
        following = words[index + 1 : index + 3]
        if word in _TENS_VALUES and following and 0 < _UNIT_VALUES.get(following[0], 0) < 10:
            value += _UNIT_VALUES[following.pop(0)]
            numbers.add(Decimal(value))
        if following and following[0] in _MULTIPLIER_WORDS:
            numbers.add(Decimal(value * _MULTIPLIER_WORDS[following[0]]))
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
