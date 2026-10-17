"""Метод 1: детекция препятствий по оптическому потоку.

backends.py  - вычисление плотного поля потока (Farneback, DIS, LK на сетке)
detector.py  - дивергенция -> время до столкновения -> область и сторона обхода
synth_check.py - проверка на синтетических последовательностях с известным TTC
"""
from .backends import make_flow_backend, BACKENDS
from .detector import FlowConfig, FlowObstacleDetector

__all__ = ["make_flow_backend", "BACKENDS", "FlowConfig", "FlowObstacleDetector"]
