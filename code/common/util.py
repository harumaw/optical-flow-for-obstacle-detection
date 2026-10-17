"""Мелкие общие утилиты."""
from __future__ import annotations

import re


def safe_name(s: str) -> str:
    """Имя метода/источника -> допустимое имя файла."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def unique_names(dets: list) -> None:
    """Если два детектора получили одинаковое имя (разные параметры одного метода),
    добавить суффикс, чтобы результаты не перезаписали друг друга."""
    seen: dict[str, int] = {}
    for d in dets:
        base = d.name
        if base in seen:
            seen[base] += 1
            d.name = f"{base}#{seen[base]}"
        else:
            seen[base] = 1
