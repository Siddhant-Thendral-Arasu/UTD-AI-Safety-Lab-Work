#!/usr/bin/env python3
import os, json, math, random, re
from glob import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter, defaultdict

# ====== CONFIG ======
ROOT = "/data/home/dal667613/NEW_extracted_data/data/data"
OUT_DIR = os.path.join(ROOT, "_covernet")
FPS = 10                                     # collection rate (frames per second)
HIST_S = 2.0                                 # seconds of history
FUT_S  = 6.0                                 # seconds of future
HIST_N = int(round(HIST_S * FPS))            # 20
FUT_N  = int(round(FUT_S * FPS))             # 60
SEED   = 1337
SPLIT  = (0.8, 0.1, 0.1)                     # train/val/test
REQUIRE_WINDOW_NPZ = False                   # if True, require NPZ for ALL frames in window

HARD_STEP_M = 6.0
HARD_YAW_JUMP_DEG = 120.0

MAX_DT_IN_WINDOW = 0.25
REQ_FRAC_COUNTS  = 0.60
REQ_FRAC_SPAN    = 0.80

DEFAULT_CHANNELS = [
    "lanes_center", "lane_boundary", "road_edge", "drivable",
    "crosswalk", "stop_line", "traffic_light",
    "drivable_binary", "vehicles", "ego"
]


C0_NAMES = ["lanes_center", "lane_boundary"]    # semantic
C1_NAMES = ["drivable_binary"]                  # occupancy
C2_NAMES = ["vehicles", "ego"]                  # dynamic

# ====== Lattice-free: Fraction Cap per “Mode Bucket” ======
CAP_FRAC = 0.04

SIG_FWD_M    = 2.0
SIG_LAT_M    = 1.0
SIG_TURN_DEG = 10.0

rng = random.Random(SEED)

# ---------------- Helpers: IO ----------------
def list_town_dirs(root: str) -> List[str]:
    cands = sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)])
    keep = []
    for d in cands:
        has_meta = os.path.isdir(os.path.join(d, "metadata"))
        has_npz  = os.path.isdir(os.path.join(d, "covernet_npz_224"))
        has_rgb  = os.path.isdir(os.path.join(d, "rasters_rgb"))
        if has_meta and (has_npz or has_rgb):
            keep.append(d)
        else:
            print(f"[WARN] Skipping {d} (need metadata/ and either covernet_npz_224/ or rasters_rgb/).")
    return keep

def load_frame_meta(meta_dir: str) -> Dict[int, dict]:
    frames = {}
    for p in sorted(glob(os.path.join(meta_dir, "frame_*.json"))):
        m = re.search(r"frame_(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        idx = int(m.group(1))
        with open(p, "r") as f:
            frames[idx] = json.load(f)
    return frames

def load_npz_channels(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    stacked = d["stacked"]
    ch = DEFAULT_CHANNELS
    if "meta" in d:
        try:
            meta = d["meta"].item() if hasattr(d["meta"], "item") else d["meta"]
            if isinstance(meta, dict) and "channels" in meta and isinstance(meta["channels"], (list, tuple)):
                ch = list(meta["channels"])
        except Exception:
            pass
    if len(ch) < stacked.shape[0]:
        extra = [f"ch_{i}" for i in range(len(ch), stacked.shape[0])]
        ch = ch + extra
    return stacked, ch

# ---------------- Helpers: Units / Transforms ----------------
_DT_NOMINAL = 1.0 / FPS

def _heading_to_rad(h):
    return float(h) if abs(h) <= 3.5 else math.radians(float(h))

def world_to_ego_xy(px: np.ndarray, ego_xy: Tuple[float,float], ego_yaw_rad: float, meters_per_unit: float) -> np.ndarray:
    dx = (px[:,0] - ego_xy[0]) * meters_per_unit
    dy = (px[:,1] - ego_xy[1]) * meters_per_unit
    th = -ego_yaw_rad
    ex =  dx*math.cos(th) - dy*math.sin(th)
    ey =  dx*math.sin(th) + dy*math.cos(th)
    return np.stack([ex, ey], axis=1)

def rotate_xy(points_xy: np.ndarray, deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (points_xy @ rot.T)

def _safe_dt(ta: Optional[float], tb: Optional[float]) -> float:
    if ta is None or tb is None:
        return _DT_NOMINAL
    dt = float(tb) - float(ta)
    if not np.isfinite(dt) or dt <= 0.0:
        return _DT_NOMINAL
    return dt

def _dt_scale(dt: float, lo: float = 0.5, hi: float = 3.0) -> float:
    return max(lo, min(hi, dt / _DT_NOMINAL))

def _episode_groups(frames: Dict[int, dict]) -> Dict[Optional[int], List[int]]:
    groups: Dict[Optional[int], List[int]] = {}
    for idx, fm in frames.items():
        ep = (fm.get("metadata") or {}).get("episode_id")
        groups.setdefault(ep, []).append(idx)
    for idxs in groups.values():
        idxs.sort()
    return groups

def _auto_segments(frames: Dict[int, dict]) -> List[List[int]]:
    idxs = sorted(frames.keys())
    if not idxs: return []
    segs = [[idxs[0]]]
    for a, b in zip(idxs, idxs[1:]):
        pa = frames[a]["ego"]["position"][:2]
        pb = frames[b]["ego"]["position"][:2]
        ta = frames[a]["ego"].get("timestamp", None)
        tb = frames[b]["ego"].get("timestamp", None)
        dt = _safe_dt(ta, tb)
        step = math.hypot(pb[0]-pa[0], pb[1]-pa[1])
        if dt > 0.7 or step > HARD_STEP_M * _dt_scale(dt):  # slightly looser
            segs.append([b])
        else:
            segs[-1].append(b)
    return segs

def _find_bad_position_steps(frames: Dict[int, dict], meters_per_unit: float = 1.0) -> set:
    bad = set()
    groups = _episode_groups(frames)
    buckets = list(groups.values()) if set(groups.keys()) != {None} else _auto_segments(frames)
    for idxs in buckets:
        for a, b in zip(idxs, idxs[1:]):
            pa = frames[a]["ego"]["position"][:2]
            pb = frames[b]["ego"]["position"][:2]
            dt = _safe_dt(frames[a]["ego"].get("timestamp"), frames[b]["ego"].get("timestamp"))
            step = math.hypot((pb[0]-pa[0])*meters_per_unit, (pb[1]-pa[1])*meters_per_unit)
            if step > (HARD_STEP_M * _dt_scale(dt)):
                bad.add(a); bad.add(b)
    return bad

def _find_bad_yaw_jumps(frames: Dict[int, dict]) -> set:
    bad = set()
    groups = _episode_groups(frames)
    buckets = list(groups.values()) if set(groups.keys()) != {None} else _auto_segments(frames)
    for idxs in buckets:
        if len(idxs) < 2: continue
        yaws, ts = [], []
        for i in idxs:
            y = frames[i]["ego"].get("heading", 0.0)
            yaws.append(y if abs(y) <= 3.5 else math.radians(y))
            ts.append(frames[i]["ego"].get("timestamp"))
        yaws = np.unwrap(np.array(yaws, np.float64))
        for k in range(len(idxs) - 1):
            dt = _safe_dt(ts[k], ts[k+1])
            ddeg = abs(math.degrees(yaws[k+1] - yaws[k]))
            if ddeg > (HARD_YAW_JUMP_DEG * _dt_scale(dt)):
                bad.add(idxs[k]); bad.add(idxs[k+1])
    return bad

MAX_DT_IN_WINDOW = 1.0
REQ_MIN_HIST_SPAN = 1.5
REQ_MIN_FUT_SPAN  = 5.0
REQ_MIN_HIST_CNT  = 6
REQ_MIN_FUT_CNT   = 12

def _collect_time_window(frames: Dict[int, dict], center_idx: int,
                         past_s: float, fut_s: float) -> Optional[Tuple[List[int], List[int]]]:
    """
    RELAXED: build windows purely by timestamp ranges within the same episode/segment.
    Accept sparse windows as long as we hit minimal span + minimal counts.
    """
    ep = (frames[center_idx].get("metadata") or {}).get("episode_id", None)
    if ep is not None:
        idxs = sorted([i for i,fm in frames.items()
                       if (fm.get("metadata") or {}).get("episode_id", None) == ep])
    else:
        segs = _auto_segments(frames)
        idxs = next((seg for seg in segs if center_idx in seg), None)
        if not idxs: return None

    t = {i: float(frames[i]["ego"].get("timestamp", 0.0)) for i in idxs}
    tc = t.get(center_idx, None)
    if tc is None:
        return None

    hist = [i for i in idxs if (tc - past_s - 1e-6) <= t[i] <= tc + 1e-9]
    fut  = [i for i in idxs if tc - 1e-9 <= t[i] <= (tc + fut_s + 1e-6)]

    if not hist or not fut:
        return None

    hist.sort(); fut.sort()

    if center_idx not in hist: hist.append(center_idx); hist.sort()
    if center_idx not in fut:  fut.append(center_idx);  fut.sort()

    hist_span = t[hist[-1]] - t[hist[0]]
    fut_span  = t[fut[-1]]  - t[fut[0]]

    if hist_span < REQ_MIN_HIST_SPAN or len(hist) < REQ_MIN_HIST_CNT:
        return None
    if fut_span  < REQ_MIN_FUT_SPAN  or len(fut)  < REQ_MIN_FUT_CNT:
        return None

    return hist, fut

    t = {i: float(frames[i]["ego"].get("timestamp", 0.0)) for i in idxs}
    tc = t[center_idx]

    # Past
    hist = [center_idx]
    j = idxs.index(center_idx)
    prev = j - 1
    while prev >= 0:
        i_prev = idxs[prev]
        if tc - t[i_prev] > past_s + 1e-3: break
        if (prev < j - 1):
            dt_local = t[idxs[prev+1]] - t[i_prev]
            if dt_local > MAX_DT_IN_WINDOW: break
        hist.append(i_prev)
        prev -= 1
    hist = sorted(hist)
    if len(hist) < int(REQ_FRAC_COUNTS * (HIST_N + 1)):
        return None
    if (t[hist[-1]] - t[hist[0]]) < past_s * REQ_FRAC_SPAN:
        return None

    # Future
    fut = [center_idx]
    nxt = j + 1
    while nxt < len(idxs):
        i_next = idxs[nxt]
        if t[i_next] - tc > fut_s + 1e-3: break
        if (nxt > j + 1):
            dt_local = t[i_next] - t[idxs[nxt-1]]
            if dt_local > MAX_DT_IN_WINDOW: break
        fut.append(i_next)
        nxt += 1
    fut = sorted(fut)
    if (t[fut[-1]] - t[fut[0]]) < fut_s * REQ_FRAC_SPAN:
        return None
    if len(fut) < int(REQ_FRAC_COUNTS * (FUT_N + 1)):
        return None

    return hist, fut

def _npz_window_present(npz_dir: str, rng: List[int]) -> bool:
    return all(os.path.exists(os.path.join(npz_dir,f"frame_{i:05d}.npz")) for i in rng)

def _resample_future_ego_2hz_by_time(fut_meta_seq: List[dict], meters_per_unit: float) -> np.ndarray:
    if len(fut_meta_seq)<2: raise ValueError("Insufficient future frames.")
    t0 = float(fut_meta_seq[0]["ego"]["timestamp"])
    ego_xy0 = tuple(fut_meta_seq[0]["ego"]["position"][:2])
    ego_yaw0_rad = _heading_to_rad(fut_meta_seq[0]["ego"]["heading"])
    ts, xy = [], []
    for fm in fut_meta_seq[1:]:
        ts.append(float(fm["ego"]["timestamp"])-t0)
        xy.append(fm["ego"]["position"][:2])
    ts = np.array(ts,np.float32); xy = np.array(xy,np.float32)
    if ts[-1] < (FUT_S*0.8): raise ValueError("Future duration too short.")
    fut_ego_x = world_to_ego_xy(xy, ego_xy0, ego_yaw0_rad, meters_per_unit)
    fut_ego_y = rotate_xy(fut_ego_x, +90.0)
    tgt = np.arange(0.5, FUT_S+1e-6, 0.5, dtype=np.float32)
    out = np.empty((tgt.size,2),np.float32)
    for j in range(2):
        out[:,j] = np.interp(tgt, ts, fut_ego_y[:,j])
    return out.astype(np.float32)

# ---------------- Builders ----------------
def derive_future_ego(frame_meta_seq: List[dict], meters_per_unit: float) -> np.ndarray:
    t0 = frame_meta_seq[0]
    ego_xy0 = t0["ego"]["position"][:2]
    ego_yaw0_rad = _heading_to_rad(t0["ego"]["heading"])
    fut_xy_world = np.array([[fm["ego"]["position"][0], fm["ego"]["position"][1]] for fm in frame_meta_seq[1:]], np.float32)
    fut_ego_x = world_to_ego_xy(fut_xy_world, ego_xy0, ego_yaw0_rad, meters_per_unit)
    fut_ego_y = rotate_xy(fut_ego_x, +90.0)
    return fut_ego_y.astype(np.float32)

def derive_history_ego(frame_meta_seq: List[dict], meters_per_unit: float) -> np.ndarray:
    t0 = frame_meta_seq[-1]
    ego_xy0 = t0["ego"]["position"][:2]
    ego_yaw0_rad = _heading_to_rad(t0["ego"]["heading"])
    past_xy = np.array([[fm["ego"]["position"][0], fm["ego"]["position"][1]] for fm in frame_meta_seq[:-1]], np.float32)
    past_ego_x = world_to_ego_xy(past_xy, ego_xy0, ego_yaw0_rad, meters_per_unit)
    past_ego_y = rotate_xy(past_ego_x, +90.0)
    return past_ego_y.astype(np.float32)

# ====== Signature (bucket) without Lattice ======
def _traj_signature_key(fut_2hz: np.ndarray) -> Tuple[int,int,int]:
    endx, endy = float(fut_2hz[-1,0]), float(fut_2hz[-1,1])
    diffs = np.diff(fut_2hz, axis=0)
    hd = np.degrees(np.arctan2(diffs[:,0], diffs[:,1]))
    if hd.size >= 2:
        dhead = np.diff(hd)
        dhead = (dhead + 180.0) % 360.0 - 180.0
        total_turn = float(np.sum(dhead))
    else:
        total_turn = 0.0
    q_fwd  = int(round(endy / SIG_FWD_M))
    q_lat  = int(round(endx / SIG_LAT_M))
    q_turn = int(round(total_turn / SIG_TURN_DEG))
    return (q_fwd, q_lat, q_turn)

def build_index_for_town(town_dir: str) -> List[dict]:
    meta_dir = os.path.join(town_dir,"metadata")
    npz_dir  = os.path.join(town_dir,"covernet_npz_224")
    rgb_dir  = os.path.join(town_dir,"rasters_rgb")
    frames = load_frame_meta(meta_dir)
    if not frames:
        return []

    have_npz = {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                for p in glob(os.path.join(npz_dir,"frame_*.npz"))} if os.path.isdir(npz_dir) else set()
    have_rgb = {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                for p in glob(os.path.join(rgb_dir,"frame_*.png"))} if os.path.isdir(rgb_dir) else set()
    present = have_npz | have_rgb

    m_per_unit = 1.0
    bad_pos = _find_bad_position_steps(frames, meters_per_unit=m_per_unit)
    bad_yaw = _find_bad_yaw_jumps(frames)
    bad_all = bad_pos | bad_yaw
    if bad_all:
        print(f"[SANITY] {os.path.basename(town_dir)}: dropping {len(bad_all)} frames "
              f"(pos>{HARD_STEP_M}m*dt_scale:{len(bad_pos)}, yaw>{HARD_YAW_JUMP_DEG}°*dt_scale:{len(bad_yaw)})")

    idxs_all = sorted(frames.keys())

    rej = defaultdict(int)

    entries = []
    mode_hist = Counter()

    for i in idxs_all:
        if i not in present:
            rej["no_npz_or_rgb"] += 1
            continue
        if i in bad_all:
            rej["center_bad"] += 1
            continue

        win = _collect_time_window(frames, i, HIST_S, FUT_S)
        if win is None:
            rej["no_window"] += 1
            continue
        hist_range, fut_range = win

        if any(j in bad_all for j in (hist_range + fut_range)):
            rej["window_contains_bad"] += 1
            continue
        if REQUIRE_WINDOW_NPZ and (not _npz_window_present(npz_dir, hist_range) or not _npz_window_present(npz_dir, fut_range)):
            rej["window_npz_missing"] += 1
            continue

        hist_meta_seq = [frames[k] for k in hist_range]
        fut_meta_seq  = [frames[k] for k in fut_range]
        try:
            fut_traj = derive_future_ego(fut_meta_seq, m_per_unit)
            hist_traj = derive_history_ego(hist_meta_seq, m_per_unit)
            fut_2hz   = _resample_future_ego_2hz_by_time(fut_meta_seq, m_per_unit)
        except Exception:
            rej["traj_compute_fail"] += 1
            continue

        key = _traj_signature_key(fut_2hz)
        max_allowed = max(1, int(math.ceil(CAP_FRAC * max(1, len(entries)))))
        if mode_hist[key] >= max_allowed:
            rej["cap_bucket"] += 1
            continue
        mode_hist[key] += 1

        npz_path_i = os.path.join(npz_dir, f"frame_{i:05d}.npz")
        rgb_path_i = os.path.join(rgb_dir, f"frame_{i:05d}.png") if os.path.isdir(rgb_dir) else None
        try:
            if os.path.exists(npz_path_i):
                _, ch_names = load_npz_channels(npz_path_i)
            else:
                ch_names = DEFAULT_CHANNELS
        except Exception:
            ch_names = DEFAULT_CHANNELS

        entry = {
            "town": os.path.basename(town_dir),
            "frame_idx": i,
            "npz_path": npz_path_i if os.path.exists(npz_path_i) else None,
            "rgb_path": rgb_path_i if rgb_path_i and os.path.exists(rgb_path_i) else None,
            "channels": ch_names,
            "rgb_compose_names": [C0_NAMES, C1_NAMES, C2_NAMES],
            "fps_model": 2.0,
            "hist_xy_ego_yfwd": hist_traj.tolist(),
            "future_xy_ego_yfwd": fut_traj.tolist(),
            "future_xy_ego_yfwd_2hz": fut_2hz.tolist(),
            "sig_key": key
        }
        entries.append(entry)

    kept = len(entries)
    print(f"[INFO] {os.path.basename(town_dir)} → kept {kept} samples "
          f"(unique buckets: {len(mode_hist)}). Rejections: {dict(rej)}")
    return entries

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    town_dirs = list_town_dirs(ROOT)
    print(f"[INFO] Found {len(town_dirs)} valid towns.")

    all_entries = []
    for td in town_dirs:
        ents = build_index_for_town(td)
        all_entries.extend(ents)
    print(f"[INFO] Total entries collected: {len(all_entries)}")

    rng.shuffle(all_entries)
    n = len(all_entries)
    n_train = int(n*SPLIT[0])
    n_val   = int(n*SPLIT[1])
    train = all_entries[:n_train]
    val   = all_entries[n_train:n_train+n_val]
    test  = all_entries[n_train+n_val:]

    for name, subset in zip(["train","val","test"], [train, val, test]):
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for e in subset:
                f.write(json.dumps(e) + "\n")
        print(f"[INFO] Saved {len(subset)} entries to {out_path}.")

if __name__=="__main__":
    main()