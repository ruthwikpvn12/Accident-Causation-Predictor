import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort
from itertools import combinations
from collections import deque, defaultdict

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
VIDEO_PATH   = "data/raw_video.mp4"
MODEL_PATH   = "yolov8n.pt"
FRAME_RATE   = 30

# ── Trim output video ─────────────────────────
TRIM_START_SEC = 0    # start at 0 seconds
TRIM_END_SEC   = 11   # stop at 11 seconds

# ── Speed vs accuracy ─────────────────────────
PROCESS_EVERY_N = 3   # process every 3rd frame
YOLO_IMGSZ      = 320 # lower = faster

# TTC thresholds (seconds) — higher = earlier warning
TTC_WARN      = 10.0  # warn 10 seconds before predicted impact
TTC_DANGER    =  6.0  # escalate to full alert

# Trajectory lookahead
EXTRAPOLATE_FRAMES = 40  # look further ahead

# Velocity smoothing
VEL_HISTORY   = 5

# Risk accumulation — trigger fast, decay slow
RISK_INCREMENT    = 45
RISK_DECREMENT    = 1
RISK_MAX          = 100
RISK_WARN_THRESH  = 15   # fire warning very early
RISK_ALERT_THRESH = 35   # fire alert early

# Stale pair cleanup
STALE_LIMIT   = 20

# Vehicle class IDs in COCO: car=2, motorbike=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# ─────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────
model   = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

# Per-track state
vel_history   = defaultdict(lambda: deque(maxlen=VEL_HISTORY))
prev_centers  = {}
prev_boxes    = {}

# Per-pair state  {(id_a, id_b): {"risk": float, "stale": int, "ttc": float}}
pair_state    = defaultdict(lambda: {"risk": 0.0, "stale": 0, "ttc": 9999})

# Alert latch & smoothing — prevents blinking
LATCH_FRAMES     = 90    # hold alert for 3 s at 30 fps
SMOOTH_RISE_RATE = 0.30  # fast fade-in
SMOOTH_FALL_RATE = 0.01  # very slow fade-out — reduces text flicker

latch_warn_level = 0     # latched level (0/1/2)
latch_countdown  = 0     # frames remaining in latch
smooth_alpha     = 0.0   # smoothed opacity of overlay

# Cache last known track boxes & trajectory dots so skipped frames
# draw the same annotations — eliminates box blinking entirely
cached_tracks    = {}    # {tid: (x1,y1,x2,y2)}
cached_traj_dots = []    # [(pt, color), ...]
cached_risk_bars = []    # [(mx,my,filled,bar_color,ttc_label), ...]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


def box_size(box):
    """Diagonal of the bounding box — used as dynamic proximity scale."""
    x1, y1, x2, y2 = box
    return np.hypot(x2 - x1, y2 - y1)


def smooth_velocity(track_id):
    """Average velocity over stored history for a track."""
    hist = vel_history[track_id]
    if not hist:
        return np.zeros(2)
    return np.mean(hist, axis=0)


def acceleration(track_id):
    """Estimate acceleration as delta of last two smoothed velocity samples."""
    hist = vel_history[track_id]
    if len(hist) < 2:
        return np.zeros(2)
    return np.array(hist[-1]) - np.array(hist[-2])


def ttc_with_accel(rel_pos, rel_vel, rel_acc):
    """
    Solve  |rel_pos + rel_vel*t + 0.5*rel_acc*t²| = 0
    for the smallest positive t (returns 9999 if no convergence).
    Uses the component along the approach direction for speed.
    """
    dist  = np.linalg.norm(rel_pos)
    if dist < 1e-6:
        return 0.0

    approach_dir   = -rel_pos / dist                 # unit vector toward collision
    closing_speed  = np.dot(rel_vel, approach_dir)   # positive = approaching
    closing_accel  = np.dot(rel_acc, approach_dir)   # positive = accelerating toward

    if closing_speed <= 0:
        return 9999.0   # moving apart

    # Quadratic: 0.5*a*t² + v*t - d = 0  (solve for t)
    if abs(closing_accel) < 0.001:
        return dist / (closing_speed * FRAME_RATE)

    a = 0.5 * closing_accel * FRAME_RATE ** 2
    b = closing_speed * FRAME_RATE
    c = -dist
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return dist / (closing_speed * FRAME_RATE)   # fallback linear

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b + sqrt_d) / (2 * a)
    t2 = (-b - sqrt_d) / (2 * a)
    positives = [t for t in (t1, t2) if t > 0]
    return min(positives) if positives else 9999.0


def trajectory_will_intersect(c1, v1, c2, v2, n_frames, prox_scale):
    """
    Extrapolate both tracks n_frames into the future and check if any
    future position pair comes within prox_scale pixels.
    Returns (True, min_distance) or (False, min_distance).
    """
    min_dist = float("inf")
    for t in range(1, n_frames + 1):
        fp1 = c1 + v1 * t
        fp2 = c2 + v2 * t
        d   = np.linalg.norm(fp1 - fp2)
        if d < min_dist:
            min_dist = d
    return min_dist < prox_scale, min_dist


def draw_annotations(frame, tracks_dict, traj_dots, risk_bars,
                     s_alpha, latch_level, frame_idx):
    """Draw all persistent annotations onto frame — used for both
    processed and skipped frames so boxes never blink."""

    # Bounding boxes + IDs
    for tid, (x1, y1, x2, y2) in tracks_dict.items():
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 200, 80), 2)
        cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)

    # Trajectory dots
    for (pt, color) in traj_dots:
        cv2.circle(frame, pt, 3, color, -1)

    # Risk bars
    bar_h = 8
    for (mx, my, filled, bar_color, ttc_label) in risk_bars:
        bar_w = 60
        cv2.rectangle(frame, (mx - bar_w//2, my - bar_h//2),
                      (mx + bar_w//2, my + bar_h//2), (40, 40, 40), -1)
        cv2.rectangle(frame, (mx - bar_w//2, my - bar_h//2),
                      (mx - bar_w//2 + filled, my + bar_h//2), bar_color, -1)
        cv2.putText(frame, ttc_label, (mx - bar_w//2, my - bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

    # Alert overlay
    if s_alpha > 0.01:
        text      = "⚠  ACCIDENT PREDICTED!" if latch_level >= 2 else "⚠  COLLISION RISK!"
        bar_color = (0, 0, 200)             if latch_level >= 2 else (0, 120, 255)
        font      = cv2.FONT_HERSHEY_DUPLEX
        scale, thickness = 1.3, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx  = int((frame.shape[1] - tw) / 2)
        ty  = 80
        pad = 16
        x1b, y1b = tx - pad, ty - th - pad
        x2b, y2b = tx + tw + pad, ty + pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1b, y1b), (x2b, y2b), bar_color, -1)
        cv2.addWeighted(overlay, s_alpha * 0.75, frame,
                        1 - s_alpha * 0.75, 0, frame)

        if s_alpha > 0.3:
            cv2.putText(frame, text, (tx, ty), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.line(frame, (x1b, y2b + 2), (x2b, y2b + 2), bar_color, 2)

    # Frame counter
    cv2.putText(frame, f"Frame {frame_idx}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
raw_fps      = cap.get(cv2.CAP_PROP_FPS)
fps          = raw_fps if (raw_fps and 5 < raw_fps < 240) else FRAME_RATE

# ── Trim: only process frames in [start, end] window ──
trim_start_frame = int(TRIM_START_SEC * fps)
trim_end_frame   = int(TRIM_END_SEC   * fps)
total_frames     = trim_end_frame - trim_start_frame

# Seek to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start_frame)

print(f"Input  → {width}x{height}  |  FPS: {fps}")
print(f"Trimmed: {TRIM_START_SEC}s → {TRIM_END_SEC}s  ({total_frames} frames)")

out = cv2.VideoWriter(
    "output_predicted.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,           # ← output uses EXACT same fps as input
    (width, height)
)

frame_idx = 0
processed = 0
start_time = time.time()
print(f"Processing every {PROCESS_EVERY_N} frames  |  YOLO imgsz={YOLO_IMGSZ}")
print(f"~{total_frames // PROCESS_EVERY_N} inference calls  |  Est. duration: {total_frames / fps:.1f}s")

# Cache last rendered overlay so skipped frames still get the annotation

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Stop at trim end
    if frame_idx > total_frames:
        break

    should_process = (frame_idx % PROCESS_EVERY_N == 0) or frame_idx == 1

    if should_process:
        processed += 1

        # ── DETECT ──────────────────────────────
        results    = model(frame, imgsz=YOLO_IMGSZ, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, float(box.conf)])

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks  = tracker.update(dets_np)

        # ── TRACK STATE ─────────────────────────
        current_centers = {}
        current_boxes   = {}
        seen_ids        = set()
        cached_tracks   = {}   # rebuild each processed frame

        for t in tracks:
            x1, y1, x2, y2, tid = t
            tid = int(tid)
            seen_ids.add(tid)
            bx = (x1, y1, x2, y2)
            c  = box_center(bx)
            current_centers[tid] = c
            current_boxes[tid]   = bx
            cached_tracks[tid]   = bx
            if tid in prev_centers:
                vel_history[tid].append(c - prev_centers[tid])

        # ── PAIR RISK EVALUATION ─────────────────
        active_pairs       = set()
        warn_level         = 0
        cached_traj_dots   = []   # rebuild each processed frame
        cached_risk_bars   = []

        for id1, id2 in combinations(current_centers.keys(), 2):
            if id1 not in prev_centers or id2 not in prev_centers:
                continue

            pair = tuple(sorted((id1, id2)))
            active_pairs.add(pair)
            ps   = pair_state[pair]
            ps["stale"] = 0

            c1  = current_centers[id1]
            c2  = current_centers[id2]
            sv1 = smooth_velocity(id1)
            sv2 = smooth_velocity(id2)
            ac1 = acceleration(id1)
            ac2 = acceleration(id2)

            rel_pos = c1 - c2
            rel_vel = sv1 - sv2
            rel_acc = ac1 - ac2

            prox_scale  = (box_size(current_boxes[id1]) + box_size(current_boxes[id2])) / 2.0
            approaching = np.dot(rel_vel, rel_pos) < 0

            if approaching:
                ttc = ttc_with_accel(rel_pos, rel_vel, rel_acc)
                ps["ttc"] = ttc
                will_hit, future_min_dist = trajectory_will_intersect(
                    c1, sv1, c2, sv2, EXTRAPOLATE_FRAMES, prox_scale)
                dangerous = ttc < TTC_WARN or will_hit
                if dangerous:
                    boost = RISK_INCREMENT
                    if ttc < TTC_DANGER:             boost += 15
                    if will_hit and future_min_dist < prox_scale * 0.5: boost += 10
                    ps["risk"] = min(RISK_MAX, ps["risk"] + boost)
                    # Cache trajectory dots
                    for steps in range(1, EXTRAPOLATE_FRAMES + 1, 3):
                        pt1   = (int(c1[0] + sv1[0]*steps), int(c1[1] + sv1[1]*steps))
                        pt2   = (int(c2[0] + sv2[0]*steps), int(c2[1] + sv2[1]*steps))
                        alpha = 1 - steps / EXTRAPOLATE_FRAMES
                        col   = (0, int(165*alpha), int(255*alpha))
                        cached_traj_dots.append((pt1, col))
                        cached_traj_dots.append((pt2, col))
                else:
                    ps["risk"] = max(0, ps["risk"] - RISK_DECREMENT)
                    ps["ttc"]  = 9999
            else:
                ps["risk"] = max(0, ps["risk"] - RISK_DECREMENT)
                ps["ttc"]  = 9999

            risk = ps["risk"]
            if risk > RISK_WARN_THRESH:
                mx        = int((c1[0] + c2[0]) / 2)
                my        = int((c1[1] + c2[1]) / 2)
                bar_w     = 60
                filled    = int(bar_w * risk / RISK_MAX)
                bar_color = (0, 60, 255) if risk >= RISK_ALERT_THRESH else (0, 165, 255)
                ttc_label = f"{ps['ttc']:.1f}s" if ps["ttc"] < 9999 else ""
                cached_risk_bars.append((mx, my, filled, bar_color, ttc_label))
                if risk >= RISK_ALERT_THRESH:   warn_level = max(warn_level, 2)
                elif risk >= RISK_WARN_THRESH:  warn_level = max(warn_level, 1)

        # ── STALE PAIR CLEANUP ───────────────────
        for pair in list(pair_state.keys()):
            if pair not in active_pairs:
                pair_state[pair]["stale"] += 1
                if pair_state[pair]["stale"] > STALE_LIMIT:
                    del pair_state[pair]

        prev_centers = current_centers.copy()
        prev_boxes   = current_boxes.copy()

        # ── UPDATE LATCH & SMOOTH ALPHA ──────────
        if warn_level > latch_warn_level:
            latch_warn_level = warn_level
            latch_countdown  = LATCH_FRAMES
        elif warn_level == latch_warn_level:
            latch_countdown  = LATCH_FRAMES
        else:
            if latch_countdown > 0: latch_countdown -= 1
            else:                   latch_warn_level = 0

        target_alpha = latch_warn_level / 2.0
        if smooth_alpha < target_alpha:
            smooth_alpha = min(target_alpha, smooth_alpha + SMOOTH_RISE_RATE)
        else:
            smooth_alpha = max(target_alpha, smooth_alpha - SMOOTH_FALL_RATE)

    # ── DRAW ALL ANNOTATIONS (processed & skipped frames) ────
    # Using cached values means boxes/dots/bars appear on EVERY frame — no blinking
    draw_annotations(frame, cached_tracks, cached_traj_dots, cached_risk_bars,
                     smooth_alpha, latch_warn_level, frame_idx)

    out.write(frame)

    if frame_idx % 200 == 0:
        pct     = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        elapsed = time.time() - start_time
        eta     = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
        print(f"  {pct:5.1f}%  |  Frame {frame_idx}/{total_frames}"
              f"  |  Elapsed: {elapsed:.0f}s  |  ETA: {eta:.0f}s")

cap.release()
out.release()
total_time = time.time() - start_time
print(f"Done! Saved to output_predicted.mp4  |  Total time: {total_time:.1f}s")