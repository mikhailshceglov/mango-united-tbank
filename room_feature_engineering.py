from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


FEATURE_COLUMNS = [
    "hotel_id",
    "room_level",
    "bathroom_type",
    "single_beds",
    "double_beds",
    "king_beds",
    "twin_beds",
    "bedrooms_count",
    "capacity_persons",
    "has_balcony",
    "view_type",
    "is_family_room",
    "paren_text_len",
    "paren_token_count",
    "unique_token_ratio",
    "core_room_level",
    "hotel_share_balcony",
    "rel_business_attr_vs_hotel",
]

CATEGORICAL_FEATURES = [
    "hotel_id",
    "room_level",
    "bathroom_type",
    "view_type",
    "core_room_level",
]


# ----------------------------
# Нормализация текста
# ----------------------------
def normalize_text(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[|;/,(){}\[\]<>]+", " ", text)
    text = re.sub(r"[-–—_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Числительные
# ----------------------------
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "single": 1,
    "double": 2,
    "twin": 2,
    "triple": 3,
    "quad": 4,
    "quadruple": 4,
    "odin": 1,
    "odna": 1,
    "odno": 1,
    "dva": 2,
    "dve": 2,
    "tri": 3,
    "chetyre": 4,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "две": 2,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
}

NUMBER_TOKEN_REGEX = r"(?:\d+|" + "|".join(
    sorted(map(re.escape, NUMBER_WORDS.keys()), key=len, reverse=True)
) + r")"


def parse_number_token(token: str):
    token = str(token).strip().lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def max_number_from_patterns(text: str, patterns: Iterable[re.Pattern]) -> int:
    values = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = parse_number_token(match.group("n"))
            if value is not None:
                values.append(value)
    return max(values) if values else 0


# ----------------------------
# Уровень комнаты
# ----------------------------
ROOM_LEVEL_PATTERNS = [
    ("presidential", re.compile(r"\b(presidential|президентск\w*)\b")),
    ("executive", re.compile(r"\b(executive|exec|экзекьютив)\b")),
    ("premium", re.compile(r"\b(premium|premier|премиум)\b")),
    ("suite", re.compile(r"\b(suite|люкс)\b")),
    ("junior_suite", re.compile(r"\b(junior suite|junior|полулюкс)\b")),
    ("deluxe", re.compile(r"\b(deluxe|de luxe|делюкс)\b")),
    ("superior", re.compile(r"\b(superior|супериор|улучшенн\w*)\b")),
    ("comfort", re.compile(r"\b(comfort|комфорт)\b")),
    ("business", re.compile(r"\b(business|бизнес)\b")),
    ("standard", re.compile(r"\b(standard|standart|стандарт\w*)\b")),
    ("economy", re.compile(r"\b(economy|econ|эконом\w*)\b")),
]


def extract_room_level(text: str) -> str:
    for label, pattern in ROOM_LEVEL_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


# ----------------------------
# Ванная комната
# ----------------------------
PRIVATE_BATH_PATTERN = re.compile(
    r"\b("
    r"private bathroom|private bath|ensuite|en suite|own bathroom|attached bathroom|"
    r"ванная комната|собственная ванная|своя ванная|собственный санузел|"
    r"санузел в номере|душ в номере|туалет в номере|bath in room"
    r")\b"
)

SHARED_BATH_PATTERN = re.compile(
    r"\b("
    r"shared bathroom|shared bath|shared toilet|common bathroom|common toilet|"
    r"общая ванная|общий санузел|общий туалет|общие удобства|"
    r"ванная на этаже|душ на этаже|туалет на этаже"
    r")\b"
)


def extract_bathroom_type(text: str) -> str:
    if SHARED_BATH_PATTERN.search(text):
        return "shared"
    if PRIVATE_BATH_PATTERN.search(text):
        return "private"
    return "unknown"


# ----------------------------
# Количество и конфигурация кроватей
# ----------------------------
KING_PATTERNS = [
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:x\s*)?(?:king)(?:\s*size)?(?:\s*bed|\s*кроват\w*)?"),
]
DOUBLE_PATTERNS = [
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:x\s*)?(?:double|двуспальн\w*)(?:\s*bed|\s*кроват\w*)?"),
]
SINGLE_PATTERNS = [
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:x\s*)?(?:single|односпальн\w*)(?:\s*bed|\s*кроват\w*)?"),
]
TWIN_PATTERNS = [
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:x\s*)?(?:twin|separate|отдельн\w*|раздельн\w*)(?:\s*beds?|\s*кроват\w*)?"),
]

KING_KEYWORD = re.compile(r"\bking(?:\s*size)?(?:\s*bed)?\b")
DOUBLE_KEYWORD = re.compile(r"\b(double bed|двуспальн\w*)\b")
SINGLE_KEYWORD = re.compile(r"\b(single bed|односпальн\w*)\b")
TWIN_KEYWORD = re.compile(r"\b(twin beds?|separate beds?|отдельн\w* кроват\w*|раздельн\w* кроват\w*)\b")


def extract_bed_count(
    text: str,
    numeric_patterns: list[re.Pattern],
    keyword_pattern: re.Pattern,
    default_if_keyword: int,
) -> int:
    value = max_number_from_patterns(text, numeric_patterns)
    if value > 0:
        return value
    if keyword_pattern.search(text):
        return default_if_keyword
    return 0


# ----------------------------
# Количество спален
# ----------------------------
BEDROOM_PATTERNS = [
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*bedrooms?\b"),
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*bedroom\b"),
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*спальн\w*\b"),
]


def extract_bedrooms_count(text: str) -> int:
    return max_number_from_patterns(text, BEDROOM_PATTERNS)


# ----------------------------
# Вместимость номера
# ----------------------------
CAPACITY_PATTERNS = [
    re.compile(fr"(?:for|up to|sleeps?)\s*(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:guests?|persons?|people|adults?)\b"),
    re.compile(fr"(?P<n>{NUMBER_TOKEN_REGEX})\s*(?:guests?|persons?|people|adults?|чел(?:овек)?|гост\w*|местн\w*)\b"),
]


def extract_capacity(text: str) -> int:
    value = max_number_from_patterns(text, CAPACITY_PATTERNS)
    if value > 0:
        return value

    if re.search(r"\b(single room|одноместн\w*)\b", text):
        return 1
    if re.search(r"\b(double room|двухместн\w*|twin room)\b", text):
        return 2
    if re.search(r"\b(triple|трехместн\w*|трёхместн\w*)\b", text):
        return 3
    if re.search(r"\b(quad|quadruple|четырехместн\w*|четырёхместн\w*)\b", text):
        return 4

    return 0


# ----------------------------
# Балкон
# ----------------------------
BALCONY_PATTERN = re.compile(
    r"\b(balcony|balcon|terrace|loggia|veranda|patio|балкон|терраса|лоджия|веранда)\b"
)


def extract_has_balcony(text: str) -> int:
    return int(bool(BALCONY_PATTERN.search(text)))


# ----------------------------
# Вид из окна
# ----------------------------
VIEW_PATTERNS = [
    ("sea", re.compile(r"\b(sea view|ocean view|sea side|мор[ея]\b|на море|вид на море|океан)\b")),
    ("city", re.compile(r"\b(city view|city side|город\w*|вид на город)\b")),
    ("garden", re.compile(r"\b(garden view|сад\w*|вид на сад)\b")),
    ("mountain", re.compile(r"\b(mountain view|горы|вид на горы|горн\w*)\b")),
    ("pool", re.compile(r"\b(pool view|вид на бассейн|бассейн)\b")),
    ("lake", re.compile(r"\b(lake view|озеро|вид на озеро)\b")),
    ("river", re.compile(r"\b(river view|река|вид на реку)\b")),
    ("park", re.compile(r"\b(park view|парк|вид на парк)\b")),
    ("courtyard", re.compile(r"\b(courtyard view|inner yard|внутренний двор|двор)\b")),
    ("panoramic", re.compile(r"\b(panoramic view|панорам\w*)\b")),
]


def extract_view_type(text: str) -> str:
    for label, pattern in VIEW_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


# ----------------------------
# Семейный номер
# ----------------------------
FAMILY_PATTERN = re.compile(r"\b(family|семейн\w*)\b")


def extract_is_family_room(text: str) -> int:
    return int(bool(FAMILY_PATTERN.search(text)))


# ----------------------------
# Новые текстовые meta-features
# ----------------------------
PAREN_CONTENT_PATTERN = re.compile(r"\((.*?)\)")


def extract_parenthesized_text(room_name: str) -> str:
    parts = PAREN_CONTENT_PATTERN.findall(str(room_name))
    return " ".join(parts).strip()


def extract_paren_text_len(room_name: str) -> int:
    return len(extract_parenthesized_text(room_name))


def extract_paren_token_count(room_name: str) -> int:
    text = normalize_text(extract_parenthesized_text(room_name))
    if not text:
        return 0
    return len(text.split())


def extract_unique_token_ratio(room_name: str) -> float:
    text = normalize_text(room_name)
    if not text:
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def extract_core_text(room_name: str) -> str:
    text = str(room_name)

    # Убираем всё, что внутри скобок
    text = re.sub(r"\(.*?\)", " ", text)

    # Оставляем основную часть до типичных хвостов через дефис
    parts = re.split(r"\s[-–—]\s", text)
    if parts:
        text = parts[0]

    return normalize_text(text)


def extract_core_room_level(room_name: str) -> str:
    core_text = extract_core_text(room_name)
    return extract_room_level(core_text)


# ----------------------------
# Основная функция
# ----------------------------
def extract_room_features(room_name: str) -> pd.Series:
    text = normalize_text(room_name)

    single_beds = extract_bed_count(
        text=text,
        numeric_patterns=SINGLE_PATTERNS,
        keyword_pattern=SINGLE_KEYWORD,
        default_if_keyword=1,
    )
    double_beds = extract_bed_count(
        text=text,
        numeric_patterns=DOUBLE_PATTERNS,
        keyword_pattern=DOUBLE_KEYWORD,
        default_if_keyword=1,
    )
    king_beds = extract_bed_count(
        text=text,
        numeric_patterns=KING_PATTERNS,
        keyword_pattern=KING_KEYWORD,
        default_if_keyword=1,
    )
    twin_beds = extract_bed_count(
        text=text,
        numeric_patterns=TWIN_PATTERNS,
        keyword_pattern=TWIN_KEYWORD,
        default_if_keyword=2,
    )

    return pd.Series(
        {
            "room_level": extract_room_level(text),
            "bathroom_type": extract_bathroom_type(text),
            "single_beds": single_beds,
            "double_beds": double_beds,
            "king_beds": king_beds,
            "twin_beds": twin_beds,
            "bedrooms_count": extract_bedrooms_count(text),
            "capacity_persons": extract_capacity(text),
            "has_balcony": extract_has_balcony(text),
            "view_type": extract_view_type(text),
            "is_family_room": extract_is_family_room(text),
            "paren_text_len": extract_paren_text_len(room_name),
            "paren_token_count": extract_paren_token_count(room_name),
            "unique_token_ratio": extract_unique_token_ratio(room_name),
            "core_room_level": extract_core_room_level(room_name),
        }
    )


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["supplier_room_name"] = result["supplier_room_name"].fillna("").astype(str)
    result["hotel_id"] = result["hotel_id"].fillna("unknown").astype(str)
    return result


def add_hotel_aware_features(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()

    result["business_attr_count"] = (
        (result["room_level"] != "unknown").astype(int)
        + (result["bathroom_type"] != "unknown").astype(int)
        + (result["single_beds"] > 0).astype(int)
        + (result["double_beds"] > 0).astype(int)
        + (result["king_beds"] > 0).astype(int)
        + (result["twin_beds"] > 0).astype(int)
        + (result["bedrooms_count"] > 0).astype(int)
        + (result["capacity_persons"] > 0).astype(int)
        + (result["has_balcony"] > 0).astype(int)
        + (result["view_type"] != "unknown").astype(int)
        + (result["is_family_room"] > 0).astype(int)
        + (result["paren_token_count"] > 0).astype(int)
        + (result["core_room_level"] != "unknown").astype(int)
    )

    hotel_mean_business = result.groupby("hotel_id")["business_attr_count"].transform("mean")
    result["rel_business_attr_vs_hotel"] = result["business_attr_count"] - hotel_mean_business

    result["hotel_share_balcony"] = result.groupby("hotel_id")["has_balcony"].transform("mean")

    result = result.drop(columns=["business_attr_count"])
    return result


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = prepare_base_dataframe(df)
    feature_df = result["supplier_room_name"].apply(extract_room_features)
    result = pd.concat([result, feature_df], axis=1)

    result["room_level"] = result["room_level"].fillna("unknown").astype(str)
    result["bathroom_type"] = result["bathroom_type"].fillna("unknown").astype(str)
    result["view_type"] = result["view_type"].fillna("unknown").astype(str)
    result["core_room_level"] = result["core_room_level"].fillna("unknown").astype(str)

    result = add_hotel_aware_features(result)
    return result


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS.copy()


def get_categorical_features() -> list[str]:
    return CATEGORICAL_FEATURES.copy()
