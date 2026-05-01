import os, json, math, numpy as np
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm

# ================= CONFIG =================
TOWNS_ROOT = "/data/home/dal667613/NEW_extracted_data/data/data"
OUT_SIZE = 224
METERS_PER_PIXEL = 0.2

INCLUDE_STATIC_PROPS = True
PROP_CLASSES = ["traffic_sign", "pole", "barrier", "median", "street_light"]
# =========================================

def load_frames(meta_dir):
    frames = {}
    for p in sorted(glob(os.path.join(meta_dir, "frame_*.json"))):
        tok = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r") as f:
            frames[tok] = json.load(f)
    return frames

def ppm_from_params(params):
    img = params.get("img_size", 512)
    a   = params.get("meters_ahead", 50.0)
    b   = params.get("meters_behind", 20.0)
    s   = params.get("meters_side", 35.0)
    mpp = max((a + b) / img, (2 * s) / img)
    return 1.0 / mpp, img

def world_to_px(xy, ego_xy, ego_yaw_deg, ppm, img_size):
    """xy can be (x,y) tuple or np array like; returns pixel (x,y) in source raster space."""
    x, y = float(xy[0]), float(xy[1])
    dx = x - ego_xy[0]
    dy = y - ego_xy[1]
    th = math.radians(-ego_yaw_deg)
    ex = dx * math.cos(th) - dy * math.sin(th)
    ey = dx * math.sin(th) + dy * math.cos(th)
    ey *= -1.0
    c  = img_size / 2.0
    return ex * ppm + c, ey * ppm + c

def box_corners(center_xy, yaw_deg, ex, ey):
    cx, cy = center_xy
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    local = np.array([[ ex,  ey],
                      [ ex, -ey],
                      [-ex, -ey],
                      [-ex,  ey]], np.float32)
    R = np.array([[c, -s],[s, c]], np.float32)
    return (local @ R.T) + np.array([cx, cy], np.float32)

def draw_vehicle_rounded(draw_rgb, mask, ego_xy, ego_yaw, center_xy, yaw_deg, ex, ey,
                         ppm, img, fill_rgba, label_val):
    """
    Draw a rounded capsule-like vehicle footprint:
    - Start with oriented rectangle (2*ex by 2*ey in meters)
    - Add semicircles on front & rear based on ey
    """
    shorten = min(0.4, ex * 0.35)
    body_ex = max(0.1, ex - shorten)

    body_world = box_corners(center_xy, yaw_deg, body_ex, ey)

    body_px = [world_to_px(p, ego_xy, ego_yaw, ppm, img) for p in body_world]
    body_px = [(float(x), float(y)) for x, y in body_px]

    th = math.radians(yaw_deg)
    fx = center_xy[0] + body_ex * math.cos(th)
    fy = center_xy[1] + body_ex * math.sin(th)
    rx = center_xy[0] - body_ex * math.cos(th)
    ry = center_xy[1] - body_ex * math.sin(th)

    r_px = ey * ppm

    draw_rgb.polygon(body_px, fill=fill_rgba)

    def arc_points(cx, cy, yaw, r, forward=True, steps=16):
        pts = []
        base = yaw if forward else yaw + math.pi
        for i in range(steps + 1):
            a = (i / steps - 0.5) * math.pi  # -90..+90
            ang = base + a
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            pts.append((px, py))
        return pts

    front_arc_w = arc_points(fx, fy, th, ey, forward=True, steps=16)
    rear_arc_w  = arc_points(rx, ry, th, ey, forward=False, steps=16)
    front_arc_p = [world_to_px(p, ego_xy, ego_yaw, ppm, img) for p in front_arc_w]
    rear_arc_p  = [world_to_px(p, ego_xy, ego_yaw, ppm, img) for p in rear_arc_w]

    draw_rgb.polygon([(float(x), float(y)) for x, y in front_arc_p], fill=fill_rgba)
    draw_rgb.polygon([(float(x), float(y)) for x, y in rear_arc_p],  fill=fill_rgba)

    draw_rgb.line(body_px + [body_px[0]], width=1, fill=(255,255,255,120))
    draw_rgb.line([(float(x), float(y)) for x, y in front_arc_p], width=1, fill=(255,255,255,120))
    draw_rgb.line([(float(x), float(y)) for x, y in rear_arc_p],  width=1, fill=(255,255,255,120))

    poly_mask = Image.new("L", (img, img), 0)
    dmask = ImageDraw.Draw(poly_mask)
    dmask.polygon(body_px, fill=label_val)
    dmask.polygon([(float(x), float(y)) for x, y in front_arc_p], fill=label_val)
    dmask.polygon([(float(x), float(y)) for x, y in rear_arc_p],  fill=label_val)
    ma = np.array(mask, np.uint8)
    np.maximum(ma, np.array(poly_mask, np.uint8), out=ma)
    return Image.fromarray(ma, "L")

def edge_from_mask_u8(mask_u8):
    m  = (mask_u8 > 0).astype(np.uint8)
    up = np.pad(m[1:, :], ((0, 1), (0, 0)))
    dn = np.pad(m[:-1, :], ((1, 0), (0, 0)))
    lf = np.pad(m[:, 1:], ((0, 0), (0, 1)))
    rg = np.pad(m[:, :-1], ((0, 0), (1, 0)))
    return ((m != up) | (m != dn) | (m != lf) | (m != rg)).astype(np.uint8) * 255

def nn_resize(arr):
    return np.array(Image.fromarray(arr).resize((OUT_SIZE, OUT_SIZE), Image.NEAREST), np.uint8)

def colored_lane_palette(idx):
    base = [(255,77,77),(255,153,51),(255,221,51),(77,255,77),(51,204,255),
            (102,102,255),(204,102,255),(255,102,204),(102,255,204),(255,178,102)]
    return base[idx % len(base)]

def main():
    towns = []
    if os.path.isdir(TOWNS_ROOT) and any(d.startswith("Town") for d in os.listdir(TOWNS_ROOT)):
        for d in sorted(os.listdir(TOWNS_ROOT)):
            p = os.path.join(TOWNS_ROOT, d)
            if os.path.isdir(p):
                towns.append(p)
    else:
        towns.append(TOWNS_ROOT)

    for town_dir in towns:
        meta_dir = os.path.join(town_dir, "metadata")
        lbl_dir  = os.path.join(town_dir, "rasters_lbl")
        rgb_dir  = os.path.join(town_dir, "rasters_rgb")
        if not (os.path.isdir(meta_dir) and os.path.isdir(lbl_dir)):
            print(f"[WARN] Skipping {town_dir} (missing metadata/ or rasters_lbl/).")
            continue

        out_vis = os.path.join(town_dir, "covernet_vis_224")
        out_npz = os.path.join(town_dir, "covernet_npz_224")
        os.makedirs(out_vis, exist_ok=True)
        os.makedirs(out_npz, exist_ok=True)

        frames = load_frames(meta_dir)
        if not frames:
            print(f"[WARN] No frames in {meta_dir}, skipping.")
            continue

        first_meta = next(iter(frames.values()))["metadata"]
        PPM_SRC, SRC_SZ = ppm_from_params(first_meta["raster_params"])
        CROP_METERS = OUT_SIZE * METERS_PER_PIXEL
        crop_px = int(round(CROP_METERS * PPM_SRC))
        cx = cy = SRC_SZ // 2
        half = crop_px // 2
        crop_box = (cx - half, cy - half, cx + half, cy + half)
        L, T, R, B = crop_box

        def crop_arr(a):
            """Safely crop a numpy array using Pillow, always returning np.uint8."""
            if isinstance(a, np.ndarray):
                im = Image.fromarray(a.astype(np.uint8))
            elif isinstance(a, Image.Image):
                im = a.convert("L")
            else:
                raise TypeError(f"Unsupported type for crop_arr: {type(a)}")
            return np.array(im.crop(crop_box), np.uint8)

        print(f"[INFO] Processing {town_dir}")
        for token, frame in tqdm(frames.items(), desc=f"CoverNet {os.path.basename(town_dir)}"):
            idx = token.split("_")[-1]
            ego = frame["ego"]
            agents = frame.get("agents", [])
            meta = frame["metadata"]

            ego_xy  = (ego["position"][0], ego["position"][1])
            ego_yaw = ego["heading"]
            ego_ex  = float(ego.get("extent", [2.0, 1.0])[0])
            ego_ey  = float(ego.get("extent", [2.0, 1.0])[1])

            lbl_path = os.path.join(lbl_dir, f"frame_{idx}.png")
            rgb_path = os.path.join(rgb_dir, f"frame_{idx}.png")
            if not os.path.exists(lbl_path):
                continue

            lbl_src = np.array(Image.open(lbl_path).convert("L"), np.uint8)
            rgb_src = Image.open(rgb_path).convert("RGBA") if os.path.exists(rgb_path) \
                      else Image.new("RGBA", (SRC_SZ, SRC_SZ), (0,0,0,255))

            lanes_center_src  = (lbl_src == 3).astype(np.uint8) * 255
            lane_boundary_src = (lbl_src == 2).astype(np.uint8) * 255
            drivable_src      = (lbl_src == 1).astype(np.uint8) * 255
            road_edge_src     = edge_from_mask_u8(drivable_src)

            crosswalk_src = Image.new("L", (SRC_SZ, SRC_SZ), 0)
            d_cw = ImageDraw.Draw(crosswalk_src)
            for poly in meta.get("crosswalk_polys", []):
                pts = [world_to_px(p, ego_xy, ego_yaw, PPM_SRC, SRC_SZ) for p in poly]
                d_cw.polygon(pts, fill=255)
            crosswalk_src = np.array(crosswalk_src, np.uint8)

            stopline_src = Image.new("L", (SRC_SZ, SRC_SZ), 0)
            d_sl = ImageDraw.Draw(stopline_src)
            for seg in meta.get("stop_lines", []):
                if len(seg) != 2:
                    continue
                p1 = world_to_px(seg[0], ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                p2 = world_to_px(seg[1], ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                d_sl.line([p1, p2], fill=255, width=2)
            stopline_src = np.array(stopline_src, np.uint8)

            tlight_src = Image.new("L", (SRC_SZ, SRC_SZ), 0)
            d_tl = ImageDraw.Draw(tlight_src)
            for tl in meta.get("traffic_lights_near_ego", []):
                loc = tl.get("location", [None, None, None])
                if loc[0] is None:
                    continue
                px, py = world_to_px((loc[0], loc[1]), ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                r = 3
                d_tl.ellipse([px - r, py - r, px + r, py + r], fill=255)
            tlight_src = np.array(tlight_src, np.uint8)

            overlay = Image.new("RGBA", (SRC_SZ, SRC_SZ), (0,0,0,255))
            overlay.alpha_composite(rgb_src)
            d_vis = ImageDraw.Draw(overlay, "RGBA")
            centers = meta.get("lane_centers_world", [])
            for k, entry in enumerate(centers):
                pts = [world_to_px(p, ego_xy, ego_yaw, PPM_SRC, SRC_SZ) for p in entry.get("poly", [])]
                if len(pts) >= 2:
                    d_vis.line(pts, width=2, fill=(*colored_lane_palette(k), 255))

            vehicle_mask_src = np.zeros((SRC_SZ, SRC_SZ), np.uint8)
            ego_mask_src     = np.zeros((SRC_SZ, SRC_SZ), np.uint8)

            ego_mask_src = draw_vehicle_rounded(
                d_vis, ego_mask_src, ego_xy, ego_yaw, ego_xy, ego_yaw,
                ego_ex, ego_ey, PPM_SRC, SRC_SZ, (0,120,255,140), 255
            )

            for a in agents:
                ax, ay, _ = a["position"]
                ex = float(a.get("extent", [2.0, 1.0])[0])
                ey = float(a.get("extent", [2.0, 1.0])[1])
                vehicle_mask_src = draw_vehicle_rounded(
                    d_vis, vehicle_mask_src, ego_xy, ego_yaw,
                    (ax, ay), a["heading"], ex, ey, PPM_SRC, SRC_SZ,
                    (0,255,0,110), 255
                )

            static_prop_srcs = {}
            if INCLUDE_STATIC_PROPS:
                props = frame["metadata"].get("static_props_near_ego", [])
                for cname in PROP_CLASSES:
                    canvas = Image.new("L", (SRC_SZ, SRC_SZ), 0)
                    draw = ImageDraw.Draw(canvas)
                    for sp in props:
                        if sp.get("class") != cname:
                            continue
                        x,y,_ = sp.get("location", [None,None,None])
                        if x is None:
                            continue
                        ex, ey = sp.get("extent", [0.5, 0.5])
                        yaw = math.radians(sp.get("yaw", 0.0))
                        cx1 = x + ex * math.cos(yaw);  cy1 = y + ex * math.sin(yaw)
                        cx2 = x - ex * math.cos(yaw);  cy2 = y - ex * math.sin(yaw)
                        p1 = world_to_px((cx1, cy1), ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                        p2 = world_to_px((cx2, cy2), ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                        draw.line([p1, p2], fill=255, width=2)
                        r = max(1, int(ey * PPM_SRC))
                        px, py = world_to_px((x, y), ego_xy, ego_yaw, PPM_SRC, SRC_SZ)
                        draw.ellipse([px-r, py-r, px+r, py+r], fill=255)
                    static_prop_srcs[cname] = np.array(canvas, np.uint8)

            overlay_c = overlay.crop((L, T, R, B)).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)

            lanes_center  = nn_resize(crop_arr(lanes_center_src))
            lane_boundary = nn_resize(crop_arr(lane_boundary_src))
            road_edge     = nn_resize(crop_arr(road_edge_src))
            drivable      = nn_resize(crop_arr(drivable_src))
            crosswalk     = nn_resize(crop_arr(crosswalk_src))
            stop_line     = nn_resize(crop_arr(stopline_src))
            traffic_light = nn_resize(crop_arr(tlight_src))
            vehicles      = nn_resize(crop_arr(vehicle_mask_src))
            ego_mask      = nn_resize(crop_arr(ego_mask_src))
            drivable_bin  = (drivable > 0).astype(np.uint8) * 255

            stacked_list = [
                ("lanes_center", lanes_center),
                ("lane_boundary", lane_boundary),
                ("road_edge", road_edge),
                ("drivable", drivable),
                ("crosswalk", crosswalk),
                ("stop_line", stop_line),
                ("traffic_light", traffic_light),
                ("drivable_binary", drivable_bin),
                ("vehicles", vehicles),
                ("ego", ego_mask),
            ]

            if INCLUDE_STATIC_PROPS:
                for cname in PROP_CLASSES:
                    arr_src = static_prop_srcs.get(cname, np.zeros((SRC_SZ,SRC_SZ), np.uint8))
                    arr = nn_resize(crop_arr(arr_src))
                    stacked_list.append((f"prop_{cname}", arr))

            channel_names = [n for n,_ in stacked_list]
            stacked = np.stack([a for _,a in stacked_list], axis=0).astype(np.uint8)

            token_png = f"{token}.png"
            token_npz = f"{token}.npz"
            overlay_c.convert("RGB").save(os.path.join(out_vis, token_png))

            np.savez_compressed(
                os.path.join(out_npz, token_npz),
                stacked=stacked,
                meta=dict(
                    out_size=int(OUT_SIZE),
                    meters_per_pixel=float(METERS_PER_PIXEL),
                    channels=channel_names
                )
            )

        print(f"[OK] Wrote: {out_vis} (PNGs) and {out_npz} (*.npz)")

if __name__ == "__main__":
    main()