"""Детектор препятствий по оптическому потоку.

Идея. При движении камеры вперёд на фронтальную поверхность поле потока
расходится из фокуса расширения: u = (x - x0)/tau, v = (y - y0)/tau, где tau -
время до столкновения в кадрах. Дивергенция такого поля div = du/dx + dv/dy = 2/tau,
то есть tau = 2 / div. Дивергенция не зависит от поворота камеры и от бокового
сдвига (они дают почти однородное поле с нулевой дивергенцией), поэтому она
выделяет именно сближение.

Алгоритм на кадр:
  1. полутон, уменьшение до work_width;
  2. плотный поток выбранным бэкендом;
  3. сглаживание поля, дивергенция через оператор Собеля;
  4. усреднение дивергенции по грубой сетке ячеек и экспоненциальное сглаживание во времени;
  5. порог по времени до столкновения: div > 2*dt/ttc_threshold_s;
  6. связные компоненты ячеек -> самая сильная область -> bbox, TTC, сторона обхода.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from common.decision import decide, flow_balance
from .backends import make_flow_backend
from common.timing import StageTimer
from common.types import BaseDetector, Frame, ObstacleResult


@dataclass
class FlowConfig:
    backend: str = "farneback"
    backend_kwargs: dict = field(default_factory=dict)
    work_width: int = 160          # ширина рабочего кадра, px
    pre_blur: float = 0.0          # сигма размытия полутона перед потоком (0 = выкл)
    flow_sigma: float = 3.0        # сигма сглаживания поля потока перед производными, px
    grid_cols: int = 20            # грубая сетка ячеек для дивергенции
    grid_rows: int = 15
    ttc_threshold_s: float = 4.0   # порог времени до столкновения, с
    min_global_mag: float = 0.1    # если средний |поток| меньше, камера стоит, детекции нет (px/кадр)
    min_texture: float = 2.5       # масштаб текстуры (уровней/px): вес достоверности потока в пикселе
                                   # w = (tex / (tex + min_texture))^2, плоские участки почти не голосуют
    min_conf: float = 0.2          # ячейки, где средний вес меньше, не участвуют (сплошная плоская область)
    min_cells: int = 4             # минимальная площадь области в ячейках
    min_persist: int = 1           # область должна держаться столько кадров подряд (1 = сразу);
                                   # 2 даёт +3..4% точности на стене ценой ~0.3 с задержки
    ema_alpha: float = 0.7         # сглаживание дивергенции во времени (1 = без сглаживания)
    center_frac: float = 0.4       # доля кадра для центральной зоны (диагностический TTC)
    wide_frac: float = 0.6         # область шире этой доли кадра -> stop
    default_dt: float = 1.0 / 30   # если временные метки недоступны
    frame_step: int = 1            # поток между кадром k и k-frame_step: при медленном движении
                                   # смещения растут в frame_step раз, а шум потока нет
    max_dt: float = 0.15           # пропуск кадров длиннее этого (с): поток между ними не считается
    corridor_frac: float = 0.6     # доля ширины и высоты кадра по центру, где ищется препятствие на пути;
                                   # 1.0 = весь кадр. Периферия даёт дивергенцию от стен, мимо которых
                                   # проходит камера (периферийный поток), но столкновения там нет
    path_frac: float = 0.15         # область должна задевать центральную полосу этой ширины (доля кадра):
                                   # предмет на курсе столкновения лежит на линии движения (в фокусе
                                   # расширения), предмет сбоку пролетит мимо. 1.0 = не проверять
    gyro_derotate: bool = False    # вычесть из потока поле вращения камеры по ориентации кадров (R_wc):
                                   # точная компенсация поворота до дивергенции, нужна модель объектива
    consistency_tol: float = 0.0   # >0: тревога только если TTC области за последние min_persist кадров
                                   # не растёт больше чем на tol с (препятствие действительно надвигается)
    horizon_mask: bool = False     # пол по горизонту: если в кадре есть вектор тяжести, не голосуют
                                   # ячейки, луч которых смотрит ниже горизонта более чем на horizon_margin_deg
    horizon_margin_deg: float = 0.0
    cam_model: str = "pinhole"     # модель объектива для лучей ячеек: pinhole | equidistant (рыбий глаз)
    cam_f: float = 0.0             # фокусное расстояние, px полного кадра (0 = по полю зрения 60°)
    cam_cx: float = 0.0            # главная точка, px полного кадра (0 = центр)
    cam_cy: float = 0.0
    ground_frac: float = 0.2       # нижняя доля кадра (ниже горизонта) не голосует: земля/пол
                                   # сближается с камерой, но препятствием не является
    row_baseline: float = 0.0      # вычитать долю медианы дивергенции по строке сетки (0 = выкл).
                                   # Пол и тангаж дают вклад, одинаковый по всей ширине строки, а
                                   # препятствие (ствол, стойка, паллета) узкое и выделяется над фоном строки
    derotate: bool = True          # вычитать наклонную плоскость из дивергенции (компенсация поворота)
    derotate_iters: int = 3        # итерации робастной подгонки плоскости


class FlowObstacleDetector(BaseDetector):
    proximity_kind = "ttc_s"

    def __init__(self, cfg: FlowConfig | None = None) -> None:
        self.cfg = cfg or FlowConfig()
        self.backend = make_flow_backend(self.cfg.backend, **self.cfg.backend_kwargs)
        self.name = f"flow:{self.backend.name}"
        self._buf: deque = deque(maxlen=max(1, self.cfg.frame_step))   # (gray, t) последних кадров
        self._div_ema: np.ndarray | None = None
        self._persist = 0

    def reset(self) -> None:
        self._buf.clear()
        self._div_ema = None
        self._persist = 0
        self._ttc_hist = []

    # ------------------------------------------------------------------
    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        ww = self.cfg.work_width
        if w != ww:
            wh = max(8, int(round(h * ww / w)))
            gray = cv2.resize(gray, (ww, wh), interpolation=cv2.INTER_AREA)
        if self.cfg.pre_blur > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.cfg.pre_blur)
        return gray

    def _divergence(self, flow: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Дивергенция поля после сглаживания с весом достоверности (нормированная свёртка).

        В бестекстурных участках поток не определён (проблема апертуры), и обычное
        сглаживание тянет туда артефакты, а дифференцирование превращает их в ложную
        дивергенцию на границах плоских пятен. Взвешенное сглаживание берёт поток
        из текстурных соседей: u_s = G*(w u) / G*w. Возвращает также карту достоверности G*w.
        """
        s = self.cfg.flow_sigma
        u, v = cv2.split(flow)                  # контигуозные каналы для OpenCV
        w = self._confidence(gray)
        if s > 0:
            ws = cv2.GaussianBlur(w, (0, 0), s)
            norm = np.maximum(ws, 1e-3)
            u = cv2.GaussianBlur(u * w, (0, 0), s) / norm
            v = cv2.GaussianBlur(v * w, (0, 0), s) / norm
        else:
            ws = w
        du = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dv = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        div = du + dv
        m = int(2 * s) + 1                      # краевые артефакты сглаживания
        div[:m, :] = 0.0
        div[-m:, :] = 0.0
        div[:, :m] = 0.0
        div[:, -m:] = 0.0
        mag = cv2.magnitude(u, v)
        return div, mag, ws

    def _remove_tilt(self, div_c: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Компенсация поворота камеры.

        Поворот вокруг поперечных осей добавляет к дивергенции член 3(w_x y - w_y x)/f,
        то есть наклонную плоскость b*x + c*y, равную нулю в центре кадра. Сближение
        с фронтальной поверхностью даёт постоянную добавку, препятствие даёт локальный
        бугор. Плоскость a + b*x + c*y подгоняется по ячейкам взвешенным МНК с робастными
        весами Хьюбера (бугор препятствия почти не влияет на наклон), затем наклон
        вычитается, а постоянная часть a остаётся.
        """
        rows, cols = div_c.shape
        ys, xs = np.mgrid[0:rows, 0:cols]
        x = (xs + 0.5) / cols * 2.0 - 1.0          # координаты ячеек в [-1, 1]
        y = (ys + 0.5) / rows * 2.0 - 1.0
        m = ok.ravel()
        if m.sum() < 8:
            return div_c, (0.0, 0.0)
        A = np.stack([np.ones(m.sum()), x.ravel()[m], y.ravel()[m]], axis=1)
        d = div_c.ravel()[m].astype(np.float64)
        w = np.ones_like(d)
        coef = np.zeros(3)
        for _ in range(max(1, self.cfg.derotate_iters)):
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], d * sw, rcond=None)
            r = d - A @ coef
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            k = 1.5 * s
            w = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-9))
        b, c = float(coef[1]), float(coef[2])
        return (div_c - (b * x + c * y)).astype(np.float32), (b, c)

    def _pix_rays(self, u: np.ndarray, v: np.ndarray, W: int, H: int) -> np.ndarray:
        """Лучи через пиксели (u, v) полного кадра, единичные, в осях камеры."""
        cfg = self.cfg
        f = cfg.cam_f or (W / 2) / math.tan(math.radians(30))
        cx, cy = cfg.cam_cx or W / 2, cfg.cam_cy or H / 2
        x, y = (u - cx) / f, (v - cy) / f
        if cfg.cam_model == "equidistant":
            r = np.sqrt(x * x + y * y)
            s = np.where(r > 1e-9, np.sin(r) / np.maximum(r, 1e-9), 1.0)
            d = np.stack([x * s, y * s, np.cos(r)], axis=-1)
        else:
            d = np.stack([x, y, np.ones_like(x)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def _project(self, d: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
        """Проекция направлений в пиксели полного кадра (обратно к _pix_rays)."""
        cfg = self.cfg
        f = cfg.cam_f or (W / 2) / math.tan(math.radians(30))
        cx, cy = cfg.cam_cx or W / 2, cfg.cam_cy or H / 2
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        if cfg.cam_model == "equidistant":
            rho = np.sqrt(x * x + y * y)
            th = np.arctan2(rho, z)
            k = np.where(rho > 1e-9, th / np.maximum(rho, 1e-9), 1.0 / np.maximum(z, 1e-9))
            return cx + f * x * k, cy + f * y * k
        zz = np.maximum(z, 1e-6)
        return cx + f * x / zz, cy + f * y / zz

    def _rotation_flow(self, R_prev: np.ndarray, R_cur: np.ndarray, fw: int, fh: int, W: int, H: int) -> np.ndarray:
        """Поле смещений (в пикселях рабочего кадра fw x fh) от поворота камеры между кадрами.
        Неподвижное направление мира w видно в камере как d = R_wcᵀ w, поэтому
        d_cur = R_cur ᵀ R_prev d_prev; смещение = проекция(d_cur) − проекция(d_prev)."""
        sx, sy = W / fw, H / fh
        key = (fw, fh, W, H)
        if getattr(self, "_rot_key", None) != key:
            uu, vv = np.meshgrid((np.arange(fw) + 0.5) * sx, (np.arange(fh) + 0.5) * sy)
            self._rot_key, self._rot_uv, self._rot_d = key, (uu, vv), self._pix_rays(uu, vv, W, H)
        uu, vv = self._rot_uv
        R_rel = R_cur.T @ R_prev
        d2 = self._rot_d @ R_rel.T
        u2, v2 = self._project(d2, W, H)
        return np.stack([(u2 - uu) / sx, (v2 - vv) / sy], axis=-1).astype(np.float32)

    def _cell_rays(self, W: int, H: int) -> np.ndarray:
        """Единичные направления лучей через центры ячеек сетки, в осях камеры. Кэшируется."""
        key = (W, H)
        if getattr(self, "_rays_key", None) == key:
            return self._rays
        cfg = self.cfg
        f = cfg.cam_f or (W / 2) / math.tan(math.radians(30))
        cx = cfg.cam_cx or W / 2
        cy = cfg.cam_cy or H / 2
        u = (np.arange(cfg.grid_cols) + 0.5) * W / cfg.grid_cols
        v = (np.arange(cfg.grid_rows) + 0.5) * H / cfg.grid_rows
        uu, vv = np.meshgrid(u, v)
        x, y = (uu - cx) / f, (vv - cy) / f
        if cfg.cam_model == "equidistant":
            r = np.sqrt(x * x + y * y)                  # для рыбьего глаза r = theta (угол от оси)
            s = np.where(r > 1e-9, np.sin(r) / np.maximum(r, 1e-9), 1.0)
            d = np.stack([x * s, y * s, np.cos(r)], axis=-1)
        else:
            d = np.stack([x, y, np.ones_like(x)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        self._rays_key, self._rays = key, d
        return d

    def _ground_cells(self, frame: Frame) -> np.ndarray | None:
        """Маска ячеек, смотрящих ниже горизонта более чем на horizon_margin_deg (по вектору тяжести)."""
        if not self.cfg.horizon_mask or frame.gravity is None:
            return None
        W, H = frame.size
        g = np.asarray(frame.gravity, dtype=np.float64)
        g = g / np.linalg.norm(g)
        down = np.degrees(np.arcsin(np.clip(self._cell_rays(W, H) @ g, -1, 1)))   # угол ниже горизонта
        return down > self.cfg.horizon_margin_deg

    def _confidence(self, gray: np.ndarray) -> np.ndarray:
        """Вес достоверности потока в пикселе по градиенту яркости, 0..1."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tex = (np.abs(gx) + np.abs(gy)) * (1.0 / 8.0)
        w = tex / (tex + self.cfg.min_texture)
        return w * w

    # ------------------------------------------------------------------
    def process(self, frame: Frame) -> ObstacleResult:
        cfg = self.cfg
        timer = StageTimer()
        timer.begin()
        W, H = frame.size
        res = ObstacleResult(self.name, frame.index, frame.t, proximity_kind=self.proximity_kind)

        gray = self._prepare(frame.bgr)
        timer.mark("pre")

        if not self._buf:
            self._buf.append((gray, frame.t, frame.R_wc))
            timer.mark("flow")
            timer.mark("post")
            res.timing_ms = timer.end()
            return res

        prev_gray, prev_t, prev_R = self._buf[0]  # самый старый кадр в буфере
        last_t = self._buf[-1][1]
        dt = frame.t - prev_t
        if not (dt > 0):
            dt = cfg.default_dt * len(self._buf)
        if frame.t - last_t > cfg.max_dt:
            # пропуск кадров в записи: смещения слишком велики, начинаем заново с текущего кадра
            self._buf.clear()
            self._buf.append((gray, frame.t, frame.R_wc))
            self._div_ema = None
            self._persist = 0
            timer.mark("flow")
            timer.mark("post")
            res.timing_ms = timer.end()
            res.debug = {"skipped_dt": frame.t - last_t}
            return res

        flow = self.backend.compute(prev_gray, gray)
        if cfg.gyro_derotate and prev_R is not None and frame.R_wc is not None:
            flow = flow - self._rotation_flow(prev_R, frame.R_wc, flow.shape[1], flow.shape[0], W, H)
        timer.mark("flow")

        div, mag, conf = self._divergence(flow, gray)
        gsz = (cfg.grid_cols, cfg.grid_rows)
        div_c = cv2.resize(div, gsz, interpolation=cv2.INTER_AREA) / dt   # дивергенция в 1/с
        conf_c = cv2.resize(conf, gsz, interpolation=cv2.INTER_AREA)
        reliable = conf_c >= cfg.min_conf
        ground = self._ground_cells(frame)
        if ground is not None:
            reliable &= ~ground                           # пол ниже горизонта не голосует
        else:
            g = int(round(cfg.grid_rows * cfg.ground_frac)) if cfg.ground_frac > 0 else 0
            if g > 0:
                reliable[-g:, :] = False                  # пол/земля не голосует и не влияет на наклон
        tilt = (0.0, 0.0)
        if cfg.derotate:
            div_c, tilt = self._remove_tilt(div_c, reliable)
        if cfg.row_baseline > 0:
            div_c = div_c.copy()
            for r in range(div_c.shape[0]):
                ok_r = reliable[r] if reliable[r].sum() >= 4 else np.ones_like(reliable[r])
                div_c[r] -= cfg.row_baseline * float(np.median(div_c[r][ok_r]))
        if cfg.corridor_frac < 1.0:
            cf = cfg.corridor_frac
            c0 = int(round(cfg.grid_cols * (0.5 - cf / 2))); c1 = int(round(cfg.grid_cols * (0.5 + cf / 2)))
            r0 = int(round(cfg.grid_rows * (0.5 - cf / 2))); r1 = int(round(cfg.grid_rows * (0.5 + cf / 2)))
            corridor = np.zeros_like(reliable)
            corridor[r0:r1, c0:c1] = True
            reliable = reliable & corridor                 # после подгонки наклона: он по всему кадру
        if self._div_ema is None:
            self._div_ema = div_c
        else:
            self._div_ema = cfg.ema_alpha * div_c + (1.0 - cfg.ema_alpha) * self._div_ema
        dema = self._div_ema

        div_thr = 2.0 / cfg.ttc_threshold_s               # порог дивергенции, 1/с
        risk = dema / div_thr                             # >= 1 означает срабатывание
        global_mag = float(mag.mean())
        moving = global_mag >= cfg.min_global_mag

        mask = ((dema > div_thr) & reliable & moving).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_strength = -1, 0.0
        if cfg.path_frac < 1.0:
            # полоса пути: столбцы сетки вокруг центра кадра, через которые проходит направление полёта
            p0 = cfg.grid_cols * (0.5 - cfg.path_frac / 2)
            p1 = cfg.grid_cols * (0.5 + cfg.path_frac / 2)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < cfg.min_cells:
                continue
            if cfg.path_frac < 1.0:
                x0c = stats[k, cv2.CC_STAT_LEFT]
                x1c = x0c + stats[k, cv2.CC_STAT_WIDTH]
                if x1c <= p0 or x0c >= p1:
                    continue                      # область в стороне от линии движения: пролетим мимо
            strength = float(dema[labels == k].sum())
            if strength > best_strength:
                best, best_strength = k, strength

        # диагностика: TTC по центральной зоне
        c0 = int(cfg.grid_cols * (0.5 - cfg.center_frac / 2))
        c1 = int(math.ceil(cfg.grid_cols * (0.5 + cfg.center_frac / 2)))
        r0 = int(cfg.grid_rows * (0.5 - cfg.center_frac / 2))
        r1 = int(math.ceil(cfg.grid_rows * (0.5 + cfg.center_frac / 2)))
        center_div = float(dema[r0:r1, c0:c1].mean())
        ttc_center = 2.0 / center_div if center_div > 1e-6 else math.inf

        self._persist = self._persist + 1 if best > 0 else 0
        best_ttc = 2.0 / float(dema[labels == best].mean()) if best > 0 else None
        self._ttc_hist = (getattr(self, "_ttc_hist", []) + [best_ttc])[-max(2, cfg.min_persist):]
        consistent = True
        if cfg.consistency_tol > 0 and best > 0:
            h = self._ttc_hist[-max(2, cfg.min_persist):]
            consistent = all(a is not None for a in h) and all(b - a <= cfg.consistency_tol for a, b in zip(h, h[1:]))
        if best > 0 and self._persist >= cfg.min_persist and consistent:
            x, y, w_, h_ = (stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP],
                            stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT])
            sx, sy = W / cfg.grid_cols, H / cfg.grid_rows
            bbox = (int(x * sx), int(y * sy), int((x + w_) * sx), int((y + h_) * sy))
            mean_div = float(dema[labels == best].mean())
            ttc = 2.0 / mean_div
            res.present = True
            res.bbox = bbox
            res.proximity = ttc
            res.side = decide(bbox, W, cfg.wide_frac)
            res.score = float(np.clip(1.0 - ttc / cfg.ttc_threshold_s, 0.0, 1.0))

        res.debug = {
            "flow": flow,               # плотное поле в рабочем разрешении
            "div": dema,                # сглаженная дивергенция по ячейкам
            "risk": risk,               # дивергенция / порог
            "mask": mask,
            "conf": conf_c,             # достоверность потока по ячейкам, 0..1
            "ttc_center": ttc_center,
            "balance": flow_balance(flow),
            "global_mag": global_mag,
            "dt": dt,
            "tilt": tilt,               # вычтенный наклон дивергенции по x и y, 1/с
        }
        timer.mark("post")
        res.timing_ms = timer.end()
        self._buf.append((gray, frame.t, frame.R_wc))
        return res
