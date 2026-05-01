#!/usr/bin/env python3
"""
make_epsilon8_lattice.py — Aggressively diversified lattice (M=64, T=12 @ 2 Hz)
- Fully deterministic.
- Coordinates match your pipeline: origin (0,0), +Y is forward.
- Aggressively split long/straight modes to avoid histogram collapse.
- Prunes modes that exceed 4% of observed histogram counts.
Output:
  /data/home/dal667613/NEW_extracted_data/data/lattices/epsilon_8.pkl
  -> numpy float32 of shape (<=64, 12, 2)
"""

import os, pickle, math, json
from typing import List
import numpy as np
from collections import Counter

OUT_PKL  = "/data/home/dal667613/NEW_extracted_data/data/lattices/epsilon_8.pkl"
HIST_JSON = "/data/home/dal667613/NEW_extracted_data/data/histogram.json"
DT       = 0.5   # 2 Hz
T_TARGET = 12    # 6 s
M_TARGET = 64
CAP_FRAC = 0.04  # 4%

def _bezier_cubic(p0, p1, p2, p3, T=12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    u = 1.0 - t
    pts = (u[:,None]**3)*p0 + 3*(u[:,None]**2)*t[:,None]*p1 \
        + 3*u[:,None]*(t[:,None]**2)*p2 + (t[:,None]**3)*p3
    return pts.astype(np.float32)

def _turn_curve(final_yaw_deg: float, progress_y: float) -> np.ndarray:
    th = math.radians(final_yaw_deg)
    yT = float(progress_y)
    xT = yT * math.tan(th/2.0) * 0.7
    xT = max(-12.0, min(12.0, xT))
    p0 = np.array([0.0, 0.0], np.float32)
    p3 = np.array([xT, yT], np.float32)
    d0 = np.array([0.0, 1.0], np.float32)
    d1 = np.array([math.sin(th), math.cos(th)], np.float32)
    h = yT * (0.3 + 0.25 * min(1.0, abs(th)/math.radians(60.0)))
    p1 = p0 + d0 * h
    p2 = p3 - d1 * h
    return _bezier_cubic(p0, p1, p2, p3, T=T_TARGET)

def _straight(progress_y: float) -> np.ndarray:
    p0 = np.array([0.0, 0.0], np.float32)
    p3 = np.array([0.0, float(progress_y)], np.float32)
    d  = float(progress_y) * 0.5
    p1 = p0 + np.array([0.0, d], np.float32)
    p2 = p3 - np.array([0.0, d], np.float32)
    return _bezier_cubic(p0, p1, p2, p3, T=T_TARGET)

def _lane_change(progress_y: float, lateral_x: float) -> np.ndarray:
    p0 = np.array([0.0, 0.0], np.float32)
    p3 = np.array([float(lateral_x), float(progress_y)], np.float32)
    d  = float(progress_y) * 0.55
    p1 = p0 + np.array([0.0, d], np.float32)
    p2 = p3 - np.array([0.0, d], np.float32)
    return _bezier_cubic(p0, p1, p2, p3, T=T_TARGET)

def build_fixed_lattice() -> np.ndarray:
    modes: List[np.ndarray] = []

    # Short/medium turns
    yaw_A = [-45, -30, -15, 15, 30, 45]
    prog_A = [12, 20, 28]
    for p in prog_A:
        for yaw in yaw_A:
            modes.append(_turn_curve(yaw, p))

    # Stronger turns ±60°
    for p in [20, 28, 36]:
        modes.append(_turn_curve(60, p))
        modes.append(_turn_curve(-60, p))

    # Gentle lane changes
    for p in [20, 28, 36]:
        for x in [-3.5, -1.75, 1.75, 3.5]:
            modes.append(_lane_change(p, x))

    # Strong lane changes ±5 m
    for p in [28, 36]:
        modes.append(_lane_change(p, 5.0))
        modes.append(_lane_change(p, -5.0))

    # Crawl & short straights
    for y in [0.5, 3.0, 6.0, 10.0]:
        modes.append(_straight(y))

    # Medium straights + subtle drifts
    for p in [20, 28, 36]:
        modes.append(_straight(p))
    for p in [20, 28]:
        for x in [-0.5, 0.5]:
            modes.append(_lane_change(p, x))
    for p in [28, 36]:
        for x in [-1.0, 1.0]:
            modes.append(_lane_change(p, x))
    modes.append(_lane_change(20, 2.0))

    # Aggressive long straights
    long_y = [30, 40, 50, 55]
    yaw_long = [-15, -10, 0, 10, 15]
    lat_long = [-6.0, -3.0, 0.0, 3.0, 6.0]
    count = 0
    for y in long_y:
        for yaw in yaw_long:
            modes.append(_turn_curve(yaw, y))
            count += 1
            if len(modes) >= M_TARGET:
                break
        if len(modes) >= M_TARGET:
            break
    while len(modes) < M_TARGET:
        modes.append(_lane_change(40, lat_long[count % len(lat_long)]))
        count += 1

    arr = np.stack(modes, axis=0).astype(np.float32)
    assert np.isfinite(arr).all()
    return arr

def prune_lattice(lattice: np.ndarray) -> np.ndarray:
    if not os.path.exists(HIST_JSON):
        print("[WARN] Histogram file not found, skipping pruning")
        return lattice

    with open(HIST_JSON, "r") as f:
        hist = json.load(f)

    total = sum(hist.values())
    cap = int(total * CAP_FRAC)
    print(f"[INFO] Total samples={total}, cap={cap}")

    keep_idx = []
    for idx in range(len(lattice)):
        cnt = hist.get(str(idx), hist.get(idx, 0))  # handle str or int keys
        if cnt > cap:
            print(f"[PRUNE] Mode {idx} count={cnt} exceeds cap={cap}, dropping")
            continue
        keep_idx.append(idx)

    pruned = lattice[keep_idx]
    print(f"[OK] Kept {len(pruned)}/{len(lattice)} modes after pruning")
    return pruned

def main():
    lattice = build_fixed_lattice()
    lattice = prune_lattice(lattice)
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(lattice, f)
    print(f"[OK] Wrote lattice to: {OUT_PKL}")
    print("Shape:", lattice.shape)

if __name__ == "__main__":
    main()
