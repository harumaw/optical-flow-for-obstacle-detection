"""Общая инфраструктура для обоих методов: типы, источники кадров, разметка,
метрики, отрисовка и правило выбора манёвра. Сами методы лежат в пакетах
optical_flow/ и neural_net/ и зависят от common, но не друг от друга."""
from .types import Frame, ObstacleResult, BaseDetector, SIDES

__all__ = ["Frame", "ObstacleResult", "BaseDetector", "SIDES"]
