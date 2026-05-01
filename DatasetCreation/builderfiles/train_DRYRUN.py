#!/usr/bin/env python3
import os, json, argparse, pickle, math, random
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEF_TRAIN = "/data/home/dal667613/NEW_extracted_data/data/data/_covernet/train.jsonl"
DEF_VAL   = "/data/home/dal667613/NEW_extracted_data/data/data/_covernet/val.jsonl"
DEF_LAT   = "/data/home/dal667613/NEW_extracted_data/data/lattices/epsilon_8.pkl"
SEED      = 1337

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

FALLBACK_RGB_MAP = [[0,1],[7],[8,9]]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _pick_path(row):
    for k in ("npz_path","npz"):
        if k in row: return row[k]
    raise KeyError("no npz path key (tried 'npz_path','npz')")

def _pick_future(row):
    for k in ("future_xy_ego_yfwd_2hz","future_xy_ego_yfwd","future_xy","future"):
        if k in row and row[k] is not None:
            arr = np.array(row[k], np.float32)
            if arr.ndim==2 and arr.shape[1]==2 and len(arr)>=2:
                return arr
    raise KeyError("no future trajectory (future_* keys)")

def _pick_state(row):
    if "state_vec" in row:
        v = np.array(row["state_vec"], np.float32)
    else:
        cs = row.get("context_scalars", {})
        ego_speed = float(cs.get("ego_speed", 0.0))
        near30    = float(cs.get("nearby_agents_30m", cs.get("nearby_agents_50m", 0.0)))
        tl_min    = float(cs.get("tl_min_dist", 0.0))
        v = np.array([ego_speed, near30, tl_min], np.float32)
    v = np.array(v, dtype=np.float32).reshape(-1)
    if v.shape[0] < 3: v = np.pad(v, (0,3-v.shape[0]))
    elif v.shape[0] > 3: v = v[:3]
    return v

def _compose_rgb_from_stack(stacked, rgb_compose=None):
    """Index-based fallback composition (legacy)."""
    if rgb_compose is None: rgb_compose = FALLBACK_RGB_MAP
    C,H,W = stacked.shape
    out = []
    for grp in rgb_compose:
        acc = np.zeros((H,W), np.uint16)
        for ch in grp:
            if 0 <= ch < C: acc += stacked[ch].astype(np.uint16)
        out.append(np.clip(acc,0,255).astype(np.uint8))
    while len(out)<3: out.append(np.zeros_like(out[0], np.uint8))
    return np.stack(out[:3], axis=0)

def _compose_rgb_from_names(stacked: np.ndarray, channels: List[str], names_groups: List[List[str]]):
    """Name-based composition using groups of channel names."""
    C,H,W = stacked.shape
    ch2idx = {n:i for i,n in enumerate(channels)}
    out = []
    for grp in names_groups:
        acc = np.zeros((H,W), np.float32)
        for name in grp:
            i = ch2idx.get(name, None)
            if i is not None:
                acc += stacked[i].astype(np.float32)
        out.append(np.clip(acc,0,255).astype(np.uint8))
    while len(out)<3: out.append(np.zeros_like(out[0], np.uint8))
    return np.stack(out[:3], axis=0)

def resample_xy(xy: np.ndarray, T: int) -> np.ndarray:
    xy = np.asarray(xy, np.float32)
    if len(xy) == T: return xy
    t_src = np.linspace(0,1,len(xy), dtype=np.float32)
    t_dst = np.linspace(0,1,T, dtype=np.float32)
    out = np.empty((T,2), np.float32)
    for j in range(2):
        out[:, j] = np.interp(t_dst, t_src, xy[:, j])
    return out

def _normalize_origin_up_np(xy: np.ndarray) -> np.ndarray:
    """Translate start to (0,0); rotate so initial motion points to +Y."""
    xy = np.asarray(xy, np.float32).copy()
    xy -= xy[0]
    if xy.shape[0] >= 2:
        v = xy[1] - xy[0]
        spd = float(np.linalg.norm(v))
        if spd > 1e-6:
            th = math.atan2(v[1], v[0])
            rot_a = (math.pi/2) - th
            c, s = math.cos(rot_a), math.sin(rot_a)
            R = np.array([[c, -s], [s, c]], np.float32)
            xy = xy @ R.T
    return xy

def delta_pointwise_L2_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d2 = (a[:,None,:,:] - b[None,:,:,:]).pow(2).sum(dim=-1)
    d2max = d2.max(dim=-1).values
    return d2max.sqrt()

def ade_fde(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = a[:,None,:,:] - b
    l2 = diff.pow(2).sum(-1).sqrt()
    ade = l2.mean(dim=-1)
    fde = l2[..., -1]
    return ade.min(dim=1).values, fde.min(dim=1).values

def _grab_future_np(row: Dict) -> np.ndarray:
    for k in ("future_xy_ego_yfwd_2hz","future_xy_ego_yfwd","future_xy","future"):
        if k in row and row[k] is not None:
            arr = np.asarray(row[k], dtype=np.float32)
            if arr.ndim==2 and arr.shape[1]==2 and len(arr)>=2:
                return arr
    raise KeyError("no future trajectory (future_* keys)")

def _resample_np(xy: np.ndarray, T: int) -> np.ndarray:
    if xy.shape[0] == T: return xy
    t0 = np.linspace(0,1,xy.shape[0],dtype=np.float32)
    t1 = np.linspace(0,1,T,         dtype=np.float32)
    out = np.empty((T,2),np.float32)
    for j in range(2):
        out[:,j] = np.interp(t1,t0,xy[:,j])
    return out

def compute_mode_counts(manifest_path: str, lattice_np: np.ndarray) -> Counter:
    M, T = lattice_np.shape[:2]
    counts = Counter()
    with open(manifest_path, "r") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            row = json.loads(s)
            fut = _grab_future_np(row)
            fut = _normalize_origin_up_np(_resample_np(fut, T))
            diff = lattice_np - fut[None,:,:]
            d = np.sqrt((diff**2).sum(-1)).max(axis=1)
            k = int(d.argmin())
            counts[k] += 1
    return counts

def build_weighted_sampler(train_rows: List[Dict], lattice_np: np.ndarray, alpha: float = 0.5):
    """weight_i = 1 / (count(mode_i)^alpha)  (optional)"""
    M, T = lattice_np.shape[:2]
    futs = []
    for r in train_rows:
        futs.append(_normalize_origin_up_np(_resample_np(_grab_future_np(r), T)))
    futs = np.stack(futs, 0)
    N = len(train_rows)
    mode_ids = []
    bs = 2048
    for s in range(0, N, bs):
        e = min(N, s+bs)
        F = futs[s:e]
        diff = lattice_np[None,:,:,:] - F[:,None,:,:]
        d = np.sqrt((diff**2).sum(-1)).max(axis=2)
        k = d.argmin(axis=1)
        mode_ids.extend(k.tolist())
    cnt = Counter(mode_ids)
    weights = [1.0 / (max(1, cnt[k]) ** alpha) for k in mode_ids]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

class CoverNetDataset(Dataset):
    def __init__(self, manifest: str):
        self.rows: List[Dict] = []
        with open(manifest, "r") as f:
            for line in f:
                s = line.strip()
                if s: self.rows.append(json.loads(s))
        if not self.rows:
            raise RuntimeError(f"No rows in {manifest}")
        print(f"[INFO] Loaded {len(self.rows)} samples from {manifest}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        npz_path = _pick_path(row)
        npz = np.load(npz_path, allow_pickle=True)
        stacked = npz["stacked"]
        fut     = _pick_future(row)
        state   = _pick_state(row)

        rgb = None
        names_groups = row.get("rgb_compose_names", None)
        ch_names = row.get("channels", None)
        if names_groups is not None and ch_names is not None:
            rgb = _compose_rgb_from_names(stacked, ch_names, names_groups)
        else:
            rgb = _compose_rgb_from_stack(stacked, row.get("rgb_compose"))

        return (torch.from_numpy(rgb).to(torch.uint8),
                torch.from_numpy(fut).to(torch.float32),
                torch.from_numpy(state).to(torch.float32),
                row)

def collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    futs = torch.stack([b[1] for b in batch], 0)
    stts = torch.stack([b[2] for b in batch], 0)
    metas= [b[3] for b in batch]
    return imgs, futs, stts, metas

class SimpleCNN(nn.Module):
    def __init__(self, num_modes, state_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,5,2), nn.ReLU(True),
            nn.Conv2d(32,64,5,2), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64+state_dim, num_modes)

    def forward(self, img, state):
        x = self.conv(img).flatten(1)
        z = torch.cat([x, state], dim=1)
        return self.fc(z)

def make_resnet50_backbone():
    from torchvision.models import resnet50
    try:
        from torchvision.models import ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        m = resnet50(pretrained=True)
    m.fc = nn.Identity()
    return m

def _set_bn_eval(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

class NuCoverNetWrapper(nn.Module):
    def __init__(self, num_modes, state_dim=3, freeze_bn=True):
        super().__init__()
        self.num_modes = num_modes
        self.state_dim = state_dim
        self.cover = None
        self.fallback = None
        self.backbone = None

        try:
            from nuscenes.prediction.models.covernet import CoverNet
            use_cover = True
        except Exception as e:
            print(f"[WARN] Could not import CoverNet: {e}. Using SimpleCNN fallback.")
            use_cover = False

        if not use_cover:
            self.fallback = SimpleCNN(num_modes, state_dim)
        else:
            from nuscenes.prediction.models.covernet import CoverNet
            self.backbone = make_resnet50_backbone()
            trials = [
                {"backbone": self.backbone, "num_modes": num_modes},
                {"backbone": self.backbone, "n_modes": num_modes},
                {"backbone": self.backbone, "trajectory_set_size": num_modes},
            ]
            ok = False
            for kw in trials:
                try:
                    self.cover = CoverNet(**kw); ok=True; break
                except Exception:
                    pass
            if not ok:
                print("[WARN] Could not construct CoverNet; using SimpleCNN fallback.")
                self.backbone = None
                self.fallback = SimpleCNN(num_modes, state_dim)

        self.register_buffer("_mean", IMAGENET_MEAN)
        self.register_buffer("_std",  IMAGENET_STD)

        if freeze_bn and self.backbone is not None:
            self.backbone.apply(_set_bn_eval)   # keep BN frozen forever
            print("[INFO] Frozen BatchNorm (eval mode, no affine grads).")

    def freeze_backbone(self, freeze=True):
        if self.backbone is None: return
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, img_u8, state):
        x = img_u8.float()/255.0
        x = (x - self._mean) / self._std
        s = state.float()
        if self.cover is None:
            return self.fallback(x, s)
        if s.size(1) != 3:
            S = s.size(1)
            if S < 3:
                pad = torch.zeros(s.size(0), 3-S, device=s.device, dtype=s.dtype)
                s = torch.cat([s, pad], dim=1)
            else:
                s = s[:, :3]
        return self.cover(x, s)

def make_warmup_cosine(total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step+1)/float(max(1, warmup_steps))
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return lr_lambda

@torch.no_grad()
def assign_labels_delta(fut: torch.Tensor, lattice: torch.Tensor):
    """
    Return hard labels, resampled+normalized futures, and full δ matrix.
    FUTURES are resampled to T and normalized to origin + initial motion -> +Y,
    to match how the lattice was built.
    """
    B, Tf, _ = fut.shape
    M, T, _  = lattice.shape
    fut_np = fut.detach().cpu().numpy()

    fut_rs_norm = []
    for b in range(B):
        arr = resample_xy(fut_np[b], T)
        arr = _normalize_origin_up_np(arr)
        fut_rs_norm.append(arr)
    fut_rs = torch.from_numpy(np.stack(fut_rs_norm, 0)).to(lattice.device).float()

    d = delta_pointwise_L2_torch(fut_rs, lattice)
    return d.argmin(dim=-1), fut_rs, d

def soft_targets_from_deltas(d: torch.Tensor, K: int = 3, tau: float = 6.0):
    """
    d: (B,M) δ distances. Return indices (B,K) and probs (B,K) with softmax over -d/tau.
    """
    K = max(1, min(K, d.size(1)))
    vals, idx = torch.topk(-d, k=K, dim=1)
    logits = vals / max(1e-6, tau)
    probs = torch.softmax(logits, dim=1)
    return idx, probs

def soft_ce_loss(logits: torch.Tensor, idx: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a sparse soft distribution (top-K only)."""
    logp = torch.log_softmax(logits.float(), dim=1)
    gather = torch.gather(logp, 1, idx)
    loss = -(probs * gather).sum(dim=1).mean()
    return loss

def lattice_oracle_cov(manifest_path: str, lattice_np: np.ndarray, eps: float = 8.0):
    """
    Oracle coverage of the lattice: normalize futures same way as lattice,
    then compute δ* = min_m max_t ||f_t - L_m,t||_2 and report cov(δ*<=eps).
    """
    M, T, _ = lattice_np.shape
    ok = 0
    n = 0
    deltas = []
    with open(manifest_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            fut = _grab_future_np(row)
            fut = _normalize_origin_up_np(_resample_np(fut, T))
            diff = lattice_np - fut[None, :, :]
            d = np.sqrt((diff**2).sum(-1)).max(axis=1)
            dmin = float(d.min())
            deltas.append(dmin)
            ok += int(dmin <= eps)
            n += 1
    return (ok / float(max(1, n))), np.array(deltas, dtype=np.float32)

@torch.no_grad()
def evaluate(model, loader, lattice, eps=8.0, topk=(1,5,10), device="cuda"):
    model.eval()
    T = lattice.size(1)
    n = 0
    tot_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    kmax = max(topk)

    top_hits = {k: 0 for k in topk}
    sum_minade = {k: 0.0 for k in (5,10) if k <= kmax}
    sum_minfde = {k: 0.0 for k in (5,10) if k <= kmax}
    cov_eps    = {k: 0 for k in (5,10) if k <= kmax}

    for imgs_u8, fut_xy, state_vec, _ in loader:
        imgs = imgs_u8.to(device)
        fut  = fut_xy.to(device)
        st   = state_vec.to(device)

        labels, fut_rs, _ = assign_labels_delta(fut, lattice)
        with torch.amp.autocast('cuda', enabled=False):
            logits = model(imgs, st).float()
            loss   = ce(logits, labels)

        if not torch.isfinite(loss):
            continue

        tot_loss += loss.item()
        n += imgs.size(0)

        for k in topk:
            _, predk = logits.topk(k, dim=1)
            top_hits[k] += (predk.eq(labels.unsqueeze(1)).any(dim=1).sum().item())

        for k in (5,10):
            if k > kmax: continue
            _, idx = logits.topk(k, dim=1)
            preds = lattice[idx]
            min_ade, min_fde = ade_fde(fut_rs, preds)
            sum_minade[k] += float(min_ade.sum().item())
            sum_minfde[k] += float(min_fde.sum().item())
            d = delta_pointwise_L2_torch(fut_rs, preds)
            cov_eps[k] += int((d.min(dim=1).values <= eps).sum().item())

    out = {"loss": tot_loss / max(1,n), "n": n}
    for k in topk:
        out[f"top{k}"] = top_hits[k] / max(1,n)
    for k in (5,10):
        if k in sum_minade:
            out[f"minADE@{k}"] = sum_minade[k] / max(1,n)
            out[f"minFDE@{k}"] = sum_minfde[k] / max(1,n)
            out[f"cov@{k}_eps"] = cov_eps[k] / max(1,n)
    model.train()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=DEF_TRAIN)
    p.add_argument("--val",   default=DEF_VAL)
    p.add_argument("--lattice", default=DEF_LAT)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr",    type=float, default=3e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=400)
    p.add_argument("--freeze_warmup", type=int, default=600)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="./checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="enable AMP for forward (loss stays fp32)")

    p.add_argument("--weighted_sampler", action="store_true",
                   help="Enable inverse-frequency WeightedRandomSampler on train set.")
    p.add_argument("--invfreq_alpha", type=float, default=0.5,
                   help="Exponent in weights = 1/(count^alpha).")
    p.add_argument("--soft_k", type=int, default=3,
                   help="Use soft targets over top-K nearest modes (K<2 -> hard labels).")
    p.add_argument("--soft_tau", type=float, default=8.0,
                   help="Temperature for soft targets over -δ distances.")
    p.add_argument("--use_constant_loss", action="store_true",
                   help="Use nuscenes ConstantLatticeLoss if available; else CE/soft CE.")

    p.add_argument("--oracle_cov", action="store_true",
                   help="Compute oracle cov@eps on VAL and exit.")
    args = p.parse_args()

    device = torch.device(args.device)

    train_ds = CoverNetDataset(args.train)
    val_ds   = CoverNetDataset(args.val)

    with open(args.lattice, "rb") as f:
        lat_np = pickle.load(f)
    lattice = torch.from_numpy(np.asarray(lat_np, np.float32)).to(device)  # (M,T,2)
    M, T = lattice.shape[0], lattice.shape[1]
    print(f"[INFO] Lattice loaded: {tuple(lattice.shape)} (M={M}, T={T})")

    if args.oracle_cov:
        cov_oracle, delta_list = lattice_oracle_cov(args.val, lat_np, eps=args.eps)
        q = np.percentile(delta_list, [25, 50, 75, 90, 95]).tolist()
        print(f"[ORACLE] cov@ε={cov_oracle:.3f}  median δ*={float(np.median(delta_list)):.2f}  "
              f"mean δ*={float(np.mean(delta_list)):.2f}")
        print(f"[ORACLE] δ* percentiles 25/50/75/90/95: {q}")
        return

    if args.weighted_sampler:
        counts = compute_mode_counts(args.train, lat_np)
        head_preview = dict(list(counts.items())[:8])
        print("[INFO] train mode counts (head preview):", head_preview)
        sampler = build_weighted_sampler(train_ds.rows, lat_np, alpha=args.invfreq_alpha)
        train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)

    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, drop_last=False)

    model = NuCoverNetWrapper(num_modes=M, state_dim=3, freeze_bn=True).to(device)

    if model.cover is not None and model.backbone is not None:
        head_params = [p for n,p in model.cover.named_parameters() if not n.startswith('backbone')]
        params = [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params,                 "lr": args.lr}
        ]
    else:
        params = model.parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    const_loss = None
    if args.use_constant_loss:
        try:
            from nuscenes.prediction.models.covernet import ConstantLatticeLoss
            const_loss = ConstantLatticeLoss(lattice.float())
            print("[INFO] Using ConstantLatticeLoss.")
        except Exception as e:
            print(f"[WARN] ConstantLatticeLoss unavailable ({e}); using CE/soft CE).")
            const_loss = None

    total_steps = args.epochs * max(1, len(train_dl))
    lr_lambda = make_warmup_cosine(total_steps, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    use_amp = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.freeze_backbone(True)

    best_val = float("inf")
    os.makedirs(args.outdir, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running = {"loss":0.0, "top1":0.0, "top5":0.0, "count":0}
        for imgs_u8, fut_xy, state_vec, _ in train_dl:
            imgs = imgs_u8.to(device, non_blocking=True)
            fut  = fut_xy.to(device, non_blocking=True)
            st   = state_vec.to(device, non_blocking=True)

            labels_hard, fut_rs, deltas = assign_labels_delta(fut, lattice)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(imgs, st)

            with torch.amp.autocast('cuda', enabled=False):
                if const_loss is not None:
                    loss = const_loss(logits.float(), fut_rs)
                else:
                    if args.soft_k >= 2:
                        idx_k, probs_k = soft_targets_from_deltas(deltas, K=args.soft_k, tau=args.soft_tau)
                        loss = soft_ce_loss(logits.float(), idx_k, probs_k)
                    else:
                        loss = ce_loss(logits.float(), labels_hard)

            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                print("[WARN] non-finite detected; skipping batch.")
                opt.zero_grad(set_to_none=True)
                scheduler.step(); global_step += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update(); scheduler.step()

            if global_step == args.freeze_warmup:
                model.freeze_backbone(False)
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[INFO] Unfroze backbone. Now-trainable params: {n_trainable}")

            with torch.no_grad():
                B = imgs.size(0)
                running["loss"]  += float(loss.item()) * B
                running["count"] += B
                pred1 = logits.argmax(-1)
                running["top1"]  += int((pred1 == labels_hard).sum().item())
                _, pred5 = logits.topk(5, dim=1)
                running["top5"]  += int((pred5.eq(labels_hard.unsqueeze(1)).any(dim=1).sum().item()))

            if global_step % 20 == 0:
                lr_now = opt.param_groups[0]["lr"]
                avg_loss = running["loss"]/max(1,running["count"])
                avg_t1   = running["top1"]/max(1,running["count"])
                print(f"[Step {global_step:5d}] lr={lr_now:.2e} train_loss={avg_loss:.4f} top1={avg_t1:.3f}")

            global_step += 1

        tr_loss = running["loss"]/max(1,running["count"])
        tr_top1 = running["top1"]/max(1,running["count"])
        tr_top5 = running["top5"]/max(1,running["count"])

        val = evaluate(model, val_dl, lattice, eps=args.eps, topk=(1,5,10), device=device)
        print(
            f"[Epoch {epoch}] "
            f"train: loss={tr_loss:.3f} top1={tr_top1:.3f} top5={tr_top5:.3f} | "
            f"val: loss={val['loss']:.3f} top1={val['top1']:.3f} top5={val['top5']:.3f} top10={val['top10']:.3f} | "
            f"minADE@5={val.get('minADE@5',float('nan')):.2f} "
            f"minFDE@5={val.get('minFDE@5',float('nan')):.2f} "
            f"cov@5(ε)={val.get('cov@5_eps',0):.3f} | "
            f"minADE@10={val.get('minADE@10',float('nan')):.2f} "
            f"minFDE@10={val.get('minFDE@10',float('nan')):.2f} "
            f"cov@10(ε)={val.get('cov@10_eps',0):.3f}"
        )

        key = val.get("minFDE@10", float("inf"))
        if key < best_val:
            best_val = key
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_minFDE10": best_val,
                "lattice_path": args.lattice,
                "lattice_shape": tuple(lattice.shape),
                "soft_k": args.soft_k, "soft_tau": args.soft_tau,
                "weighted_sampler": args.weighted_sampler, "invfreq_alpha": args.invfreq_alpha,
            }
            os.makedirs(args.outdir, exist_ok=True)
            path = os.path.join(args.outdir, "best.ckpt")
            torch.save(ckpt, path)
            print(f"[CKPT] Saved best to {path} (minFDE@10={best_val:.2f})")

if __name__ == "__main__":
    main()
