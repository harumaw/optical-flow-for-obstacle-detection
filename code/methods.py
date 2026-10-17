"""Реестр методов и разбор строки спецификации. Единственное место, где
скрипты запуска узнают о конкретных пакетах optical_flow/ и neural_net/.

Формат спецификации:  <метод>[:<вариант>][?ключ=значение&ключ=значение]
Примеры:
    flow                      -> оптический поток Фарнебака с настройками по умолчанию
    flow:dis                  -> DIS, пресет fast
    flow:dis?preset=medium    -> DIS, пресет medium
    flow:lk?cell=12&ttc_threshold_s=3
Ключи, совпадающие с полями FlowConfig, идут в конфиг, остальные в бэкенд потока.
"""
from __future__ import annotations

from dataclasses import fields
from urllib.parse import parse_qsl

from common.types import BaseDetector


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def parse_spec(spec: str) -> tuple[str, str | None, dict]:
    head, _, query = spec.partition("?")
    method, _, variant = head.partition(":")
    kwargs = {k: _coerce(v) for k, v in parse_qsl(query)} if query else {}
    return method.strip(), (variant.strip() or None), kwargs


def make_detector(spec: str) -> BaseDetector:
    method, variant, kwargs = parse_spec(spec)
    if method == "flow":
        from optical_flow.detector import FlowConfig, FlowObstacleDetector
        cfg_names = {f.name for f in fields(FlowConfig)}
        cfg_kw = {k: v for k, v in kwargs.items() if k in cfg_names}
        backend_kw = {k: v for k, v in kwargs.items() if k not in cfg_names}
        cfg = FlowConfig(backend=variant or "farneback", backend_kwargs=backend_kw, **cfg_kw)
        return FlowObstacleDetector(cfg)
    if method in ("depth", "yolo"):
        raise ValueError(f"Method {method!r} is not implemented yet (see neural_net/)")
    raise ValueError(f"Unknown method {method!r} in spec {spec!r}")


def list_methods() -> list[str]:
    return ["flow:farneback", "flow:dis", "flow:lk"]
