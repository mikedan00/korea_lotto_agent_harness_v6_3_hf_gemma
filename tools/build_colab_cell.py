"""Colab 단일 셀 생성기.

src/ 의 모든 .py 와 기본 데이터 CSV 를 zip→base64 로 묶어 notebooks/colab_cell_template.py 에 끼워 넣고
  - notebooks/colab_one_cell.py        (Colab 셀에 그대로 붙여넣을 코드)
  - notebooks/korea_lotto_v7_colab.ipynb (코드 셀 1개짜리 노트북)
을 만듭니다. src/ 를 고친 뒤에는 `python tools/build_colab_cell.py` 로 다시 생성하세요.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "colab_cell_template.py"
OUT_PY = ROOT / "notebooks" / "colab_one_cell.py"
OUT_NB = ROOT / "notebooks" / "korea_lotto_v7_colab.ipynb"
DATA = ["data/korealotto_1224V_CSV2.csv"]


def bundle_files() -> list[Path]:
    files = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    return files + [ROOT / d for d in DATA]


def build_bundle() -> tuple[str, str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in bundle_files():
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))  # 결정적 출력
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()[:10]


def render() -> tuple[str, str]:
    b64, version = build_bundle()
    template = TEMPLATE.read_text(encoding="utf-8")
    cell = template.replace('"__BUNDLE_VERSION__"', json.dumps(version)).replace('"__BUNDLE__"', json.dumps(b64))
    nb = {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"cellView": "form", "id": "korea-lotto-v7"},
            "outputs": [],
            "source": cell.splitlines(keepends=True),
        }],
        "metadata": {
            "colab": {"provenance": [], "name": "korea_lotto_v7_colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return cell, json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    cell, nb = render()
    if "--check" in sys.argv:
        stale = [p.name for p, c in ((OUT_PY, cell), (OUT_NB, nb)) if not p.exists() or p.read_text(encoding="utf-8") != c]
        if stale:
            print("다시 생성 필요:", ", ".join(stale))
            return 1
        print("최신 상태")
        return 0
    OUT_PY.write_text(cell, encoding="utf-8")
    OUT_NB.write_text(nb, encoding="utf-8")
    print(f"{OUT_PY.relative_to(ROOT)} ({len(cell)//1024} KB), {OUT_NB.relative_to(ROOT)} 생성")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
