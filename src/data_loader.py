from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Draw:
    round: int
    date: str
    numbers: tuple[int, int, int, int, int, int]
    bonus: int
    source: str = ""

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "date": self.date,
            "n1": self.numbers[0],
            "n2": self.numbers[1],
            "n3": self.numbers[2],
            "n4": self.numbers[3],
            "n5": self.numbers[4],
            "n6": self.numbers[5],
            "bonus": self.bonus,
            "source": self.source,
        }


def _clean_col(c) -> str:
    return str(c).strip().replace(" ", "")


def _extract_int(value) -> int:
    s = str(value).replace(",", "")
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"정수를 추출할 수 없습니다: {value!r}")
    return int(m.group())


def _read_csv_any(path: str | Path) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 로딩 실패: {path}, 마지막 오류={last}")


def load_lotto_csv(path: str | Path) -> list[Draw]:
    df = _read_csv_any(path)
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round", "draw"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["추첨일", "당첨일자", "date", "Date"]), None)

    # Korean official-like column names first, then positional fallback.
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당첨번호{i}", f"당번{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        numeric_cols = [c for c in df.columns if c not in [draw_col, date_col] and "보" not in c]
        ncols = numeric_cols[:6]
    bonus_col = next((c for c in df.columns if c in ["보너스", "보볼", "bonus", "Bonus"]), None)
    if bonus_col is None:
        candidates = [c for c in df.columns if "보" in c]
        bonus_col = candidates[0] if candidates else df.columns[8]

    draws: list[Draw] = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def load_lotto_html_xls(path: str | Path) -> list[Draw]:
    # The uploaded LottoSpecial file is an UTF-8 HTML table with .xls extension.
    tables = pd.read_html(path, encoding="utf-8")
    if not tables:
        return []
    df = tables[0]
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["당첨일자", "추첨일", "date", "Date"]), None)
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당번{i}", f"당첨번호{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        # Fallback: after draw/date/machine columns
        candidates = [c for c in df.columns if re.search(r"\d", c) and c != draw_col]
        ncols = candidates[:6]
    bonus_col = next((c for c in df.columns if c in ["보볼", "보너스", "bonus", "Bonus"]), None)
    if bonus_col is None:
        bonus_col = next((c for c in df.columns if "보" in c), df.columns[9])
    draws = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def merge_draws(*draw_lists: Iterable[Draw]) -> list[Draw]:
    by_round: dict[int, Draw] = {}
    for draws in draw_lists:
        for d in draws:
            by_round[d.round] = d
    return validate_draws(sorted(by_round.values(), key=lambda d: d.round))


def validate_draws(draws: Iterable[Draw]) -> list[Draw]:
    valid = []
    seen = set()
    for d in sorted(draws, key=lambda x: x.round):
        if d.round in seen:
            continue
        if len(d.numbers) != 6 or len(set(d.numbers)) != 6:
            raise ValueError(f"본번호 중복 또는 개수 오류: {d}")
        if not all(1 <= n <= 45 for n in d.numbers):
            raise ValueError(f"본번호 범위 오류: {d}")
        if not 1 <= d.bonus <= 45:
            raise ValueError(f"보너스 범위 오류: {d}")
        if d.bonus in d.numbers:
            raise ValueError(f"보너스와 본번호 중복: {d}")
        seen.add(d.round)
        valid.append(d)
    return valid


def draws_to_frame(draws: list[Draw]) -> pd.DataFrame:
    return pd.DataFrame([d.as_dict() for d in draws])


def load_history(csv_path: str | Path | None = None, xls_path: str | Path | None = None) -> list[Draw]:
    pieces = []
    if csv_path and Path(csv_path).exists():
        pieces.append(load_lotto_csv(csv_path))
    if xls_path and Path(xls_path).exists():
        pieces.append(load_lotto_html_xls(xls_path))
    if not pieces:
        raise FileNotFoundError("CSV/XLS 데이터 파일을 찾지 못했습니다.")
    return merge_draws(*pieces)
