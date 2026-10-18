from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Config:
    grid_size: int = 200
    cell_m: float = 2.0
    seed: int = 7

    ndvi_target: float = 0.30
    ndvi_width: float = 0.15
    w_ndvi: float = 0.5
    w_concavity: float = 0.5
    score_threshold: float = 0.60
    min_site_cells: int = 4
    big_site_cells: int = 25

    payload_orbs: int = 60
    max_sortie_m: float = 3500.0
    cruise_mps: float = 8.0
    base_xy: tuple = (0.0, 0.0)

    pockets_per_rev: int = 12
    auger_rpm: float = 6.0
    hopper_capacity: int = 60


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(float)
    red = red.astype(float)
    return (nir - red) / np.clip(nir + red, 1e-9, None)


def synthetic_field(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.grid_size

    def smooth_noise(sigma):
        return ndimage.gaussian_filter(rng.standard_normal((n, n)), sigma)

    dem = 10 * smooth_noise(14) + 3 * smooth_noise(4)
    dem = (dem - dem.min()) / (np.ptp(dem) + 1e-9) * 20.0

    veg = 0.55 * smooth_noise(6) - 0.35 * (dem - dem.mean()) / dem.std()
    veg = (veg - veg.min()) / (np.ptp(veg) + 1e-9)

    red = 0.25 - 0.15 * veg + 0.01 * rng.standard_normal((n, n))
    nir = 0.30 + 0.30 * veg + 0.01 * rng.standard_normal((n, n))
    return nir, red, dem


def terrain_concavity(dem: np.ndarray) -> np.ndarray:
    smooth = ndimage.gaussian_filter(dem, 2)
    lap = ndimage.laplace(smooth)
    c = np.clip(lap / (np.percentile(np.abs(lap), 95) + 1e-9), 0, 1)
    return c


@dataclass
class Microsite:
    x: float
    y: float
    area_cells: int
    score: float
    orbs: int


def find_microsites(ndvi_map: np.ndarray, dem: np.ndarray, cfg: Config):
    ndvi_term = np.exp(-((ndvi_map - cfg.ndvi_target) / cfg.ndvi_width) ** 2)
    score = cfg.w_ndvi * ndvi_term + cfg.w_concavity * terrain_concavity(dem)
    score = ndimage.gaussian_filter(score, 1)

    mask = score >= cfg.score_threshold
    labels, n = ndimage.label(mask)

    sites = []
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        if len(cells) < cfg.min_site_cells:
            continue
        cy, cx = cells.mean(axis=0)
        sites.append(
            Microsite(
                x=cx * cfg.cell_m,
                y=cy * cfg.cell_m,
                area_cells=len(cells),
                score=float(score[labels == lab].mean()),
                orbs=2 if len(cells) >= cfg.big_site_cells else 1,
            )
        )
    return sites, score


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_length(route, base):
    pts = [base] + route + [base]
    return sum(_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _two_opt(route, base):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                if _route_length(cand, base) + 1e-9 < _route_length(best, base):
                    best, improved = cand, True
    return best


def plan_sorties(sites: list[Microsite], cfg: Config):
    base = cfg.base_xy
    todo = sites[:]
    sorties = []

    while todo:
        route, orbs, pos = [], 0, base
        while todo:
            nxt = min(todo, key=lambda s: _dist(pos, (s.x, s.y)))
            trial = route + [nxt]
            fits_payload = orbs + nxt.orbs <= cfg.payload_orbs
            fits_range = _route_length([(s.x, s.y) for s in trial], base) <= cfg.max_sortie_m
            if not (fits_payload and fits_range):
                break
            route.append(nxt)
            orbs += nxt.orbs
            pos = (nxt.x, nxt.y)
            todo.remove(nxt)
        if not route:
            todo.pop(0)
            continue
        pts = _two_opt([(s.x, s.y) for s in route], base)
        order = {(s.x, s.y): s for s in route}
        sorties.append([order[p] for p in pts])
    return sorties


class SimulatedMotor:
    def run(self, seconds: float):
        pass


class DosingAuger:
    def __init__(self, cfg: Config, motor=None):
        self.cfg = cfg
        self.motor = motor or SimulatedMotor()
        self.orbs_left = cfg.hopper_capacity

    def refill(self):
        self.orbs_left = self.cfg.hopper_capacity

    def dose_time(self, n_orbs: int) -> float:
        deg_per_pocket = 360.0 / self.cfg.pockets_per_rev
        deg_per_sec = self.cfg.auger_rpm * 6.0
        return n_orbs * deg_per_pocket / deg_per_sec

    def dispense(self, n_orbs: int) -> float:
        if n_orbs not in (1, 2):
            raise ValueError("auger releases exactly 1 or 2 orbs per stop")
        if self.orbs_left < n_orbs:
            raise RuntimeError("hopper empty - return to base")
        t = self.dose_time(n_orbs)
        self.motor.run(t)
        self.orbs_left -= n_orbs
        return t


def fly_mission(sorties, cfg: Config, auger: DosingAuger):
    log, clock = [], 0.0
    for s_idx, sortie in enumerate(sorties, start=1):
        auger.refill()
        pos = cfg.base_xy
        for site in sortie:
            clock += _dist(pos, (site.x, site.y)) / cfg.cruise_mps
            dose_s = auger.dispense(site.orbs)
            clock += dose_s
            pos = (site.x, site.y)
            log.append(
                dict(
                    sortie=s_idx, t_s=round(clock, 1), x_m=round(site.x, 1),
                    y_m=round(site.y, 1), orbs=site.orbs,
                    score=round(site.score, 3), dose_s=round(dose_s, 2),
                )
            )
        clock += _dist(pos, cfg.base_xy) / cfg.cruise_mps
    return log, clock


def save_log(log, path="mission_log.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)


def plot(ndvi_map, score, sites, sorties, cfg: Config, path="green_orbs_mission.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [0, cfg.grid_size * cfg.cell_m, cfg.grid_size * cfg.cell_m, 0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))

    im = ax[0].imshow(ndvi_map, cmap="RdYlGn", extent=ext, vmin=-0.1, vmax=0.8)
    ax[0].set_title("EYES: NDVI map")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="NDVI")

    ax[1].imshow(score, cmap="Blues", extent=ext)
    ax[1].set_title("BRAIN: microsites + flight path")
    colors = plt.cm.tab10.colors
    for i, sortie in enumerate(sorties):
        pts = [cfg.base_xy] + [(s.x, s.y) for s in sortie] + [cfg.base_xy]
        xs, ys = zip(*pts)
        ax[1].plot(xs, ys, "-", color=colors[i % 10], lw=1.2, label=f"sortie {i+1}")
    ax[1].scatter([s.x for s in sites if s.orbs == 1], [s.y for s in sites if s.orbs == 1],
                  c="orange", s=22, label="1 orb", zorder=3, edgecolors="k", linewidths=0.4)
    ax[1].scatter([s.x for s in sites if s.orbs == 2], [s.y for s in sites if s.orbs == 2],
                  c="red", s=40, label="2 orbs", zorder=3, edgecolors="k", linewidths=0.4)
    ax[1].scatter(*cfg.base_xy, marker="*", c="gold", s=250, edgecolors="k", zorder=4, label="base")
    ax[1].legend(loc="lower right", fontsize=8)
    for a in ax:
        a.set_xlabel("metres")
        a.set_ylabel("metres")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    cfg = Config()

    nir, red, dem = synthetic_field(cfg)
    ndvi_map = ndvi(nir, red)
    sites, score = find_microsites(ndvi_map, dem, cfg)

    sorties = plan_sorties(sites, cfg)

    auger = DosingAuger(cfg)
    log, total_s = fly_mission(sorties, cfg, auger)

    total_orbs = sum(r["orbs"] for r in log)
    print(f"Microsites found : {len(sites)}")
    print(f"Sorties          : {len(sorties)}")
    print(f"Orbs dispersed   : {total_orbs}")
    print(f"Dose time        : {auger.dose_time(1):.1f}s (1 orb) / {auger.dose_time(2):.1f}s (2 orbs)")
    print(f"Mission time     : {total_s/60:.1f} min (excl. battery swaps)")
    save_log(log)
    plot(ndvi_map, score, sites, sorties, cfg)
    print("Saved: mission_log.csv, green_orbs_mission.png")


if __name__ == "__main__":
    main()
