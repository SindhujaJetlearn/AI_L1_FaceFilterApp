import cv2
import numpy as np
import time
import random


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _overlay_alpha(bg, fg, x, y):
    """Overlay BGRA fg image onto BGR bg at position (x, y)."""
    h, w = fg.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = _clamp(x, 0, bw), _clamp(y, 0, bh)
    x2, y2 = _clamp(x + w, 0, bw), _clamp(y + h, 0, bh)
    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return bg
    roi = bg[y1:y2, x1:x2]
    fgr = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    if fgr.shape[2] == 4:
        alpha = fgr[:, :, 3:4] / 255.0
        roi[:] = (fgr[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        roi[:] = fgr
    return bg


# ─────────────────────────────────────────────
# 1. Dog Filter
# ─────────────────────────────────────────────

def apply_dog(frame, faces):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw // 2

        # --- ears (floppy dog ears) ---
        ear_w, ear_h = fw // 3, fh // 3
        # left ear
        pts_l = np.array([
            [fx - ear_w // 4, fy],
            [fx + ear_w,       fy - ear_h],
            [fx + ear_w // 2,  fy + ear_h // 2]
        ], np.int32)
        cv2.fillPoly(out, [pts_l], (40, 20, 100))
        cv2.polylines(out, [pts_l], True, (20, 10, 60), 2)
        # right ear
        pts_r = np.array([
            [fx + fw + ear_w // 4, fy],
            [fx + fw - ear_w,       fy - ear_h],
            [fx + fw - ear_w // 2,  fy + ear_h // 2]
        ], np.int32)
        cv2.fillPoly(out, [pts_r], (40, 20, 100))
        cv2.polylines(out, [pts_r], True, (20, 10, 60), 2)

        # --- nose (big brown oval on nose tip) ---
        nose_cx = cx
        nose_cy = fy + int(fh * 0.65)
        nose_rx, nose_ry = fw // 8, fh // 14
        cv2.ellipse(out, (nose_cx, nose_cy), (nose_rx, nose_ry), 0, 0, 360, (40, 60, 140), -1)
        cv2.ellipse(out, (nose_cx, nose_cy), (nose_rx, nose_ry), 0, 0, 360, (20, 30, 80), 2)
        # nostril highlights
        cv2.circle(out, (nose_cx - nose_rx // 3, nose_cy - nose_ry // 3), 2, (80, 100, 200), -1)

        # --- tongue ---
        t_cx = cx
        t_top = fy + int(fh * 0.78)
        t_w, t_h = fw // 5, fh // 6
        pts_t = np.array([
            [t_cx - t_w, t_top],
            [t_cx + t_w, t_top],
            [t_cx + t_w, t_top + t_h - t_h // 4],
            [t_cx,       t_top + t_h],
            [t_cx - t_w, t_top + t_h - t_h // 4]
        ], np.int32)
        cv2.fillPoly(out, [pts_t], (80, 80, 220))
        cv2.polylines(out, [pts_t], True, (50, 50, 160), 2)
        # tongue crease
        cv2.line(out, (t_cx, t_top + 4), (t_cx, t_top + t_h - 6), (60, 60, 180), 2)

        # --- whiskers ---
        whisker_y = fy + int(fh * 0.62)
        whisker_color = (200, 200, 200)
        wt = 1
        # left whiskers
        for i in range(3):
            y_off = (i - 1) * fh // 14
            cv2.line(out, (fx - fw // 6, whisker_y + y_off),
                     (fx + fw // 4, whisker_y + y_off + (i - 1) * 3), whisker_color, wt)
        # right whiskers
        for i in range(3):
            y_off = (i - 1) * fh // 14
            cv2.line(out, (fx + fw + fw // 6, whisker_y + y_off),
                     (fx + fw - fw // 4, whisker_y + y_off + (i - 1) * 3), whisker_color, wt)

        # --- detection rect ---
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 200, 100), 1)
    return out


# ─────────────────────────────────────────────
# 2. Sunglasses Filter
# ─────────────────────────────────────────────

def apply_sunglasses(frame, faces):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        eye_y = fy + int(fh * 0.38)
        lens_h = int(fh * 0.18)
        lens_w = int(fw * 0.38)
        gap = int(fw * 0.05)
        cx = fx + fw // 2

        lx = cx - gap // 2 - lens_w
        rx = cx + gap // 2

        overlay = out.copy()
        # lenses
        cv2.rectangle(overlay, (lx, eye_y), (lx + lens_w, eye_y + lens_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (rx, eye_y), (rx + lens_w, eye_y + lens_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # lens frames
        cv2.rectangle(out, (lx, eye_y), (lx + lens_w, eye_y + lens_h), (60, 60, 60), 2)
        cv2.rectangle(out, (rx, eye_y), (rx + lens_w, eye_y + lens_h), (60, 60, 60), 2)

        # bridge
        cv2.line(out, (lx + lens_w, eye_y + lens_h // 2),
                 (rx, eye_y + lens_h // 2), (80, 80, 80), 2)

        # arms
        cv2.line(out, (lx, eye_y + lens_h // 2), (fx - 5, eye_y + lens_h // 3), (80, 80, 80), 2)
        cv2.line(out, (rx + lens_w, eye_y + lens_h // 2),
                 (fx + fw + 5, eye_y + lens_h // 3), (80, 80, 80), 2)

        # glare
        cv2.line(out, (lx + 4, eye_y + 3), (lx + lens_w // 3, eye_y + 3), (200, 200, 200), 1)
        cv2.line(out, (rx + 4, eye_y + 3), (rx + lens_w // 3, eye_y + 3), (200, 200, 200), 1)

        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 200, 255), 1)
    return out


# ─────────────────────────────────────────────
# 3. Crown Filter
# ─────────────────────────────────────────────

def apply_crown(frame, faces):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        crown_h = int(fh * 0.35)
        crown_y = fy - crown_h + int(fh * 0.05)
        crown_x = fx
        gold  = (0, 215, 255)
        gold2 = (0, 165, 205)
        dgold = (0, 100, 140)

        # base band
        pts = np.array([
            [crown_x,        crown_y + crown_h],
            [crown_x + fw,   crown_y + crown_h],
            [crown_x + fw,   crown_y + crown_h - crown_h // 4],
            [crown_x,        crown_y + crown_h - crown_h // 4],
        ], np.int32)
        cv2.fillPoly(out, [pts], gold2)

        # crown spikes (5 points)
        spike_pts = np.array([
            [crown_x,                  crown_y + crown_h - crown_h // 4],
            [crown_x,                  crown_y + crown_h // 2],
            [crown_x + fw // 4,        crown_y + crown_h - crown_h // 4],
            [crown_x + fw * 2 // 8,    crown_y],
            [crown_x + fw * 3 // 8,    crown_y + crown_h - crown_h // 4],
            [crown_x + fw // 2,        crown_y - crown_h // 6],
            [crown_x + fw * 5 // 8,    crown_y + crown_h - crown_h // 4],
            [crown_x + fw * 6 // 8,    crown_y],
            [crown_x + fw * 7 // 8,    crown_y + crown_h - crown_h // 4],
            [crown_x + fw,             crown_y + crown_h // 2],
            [crown_x + fw,             crown_y + crown_h - crown_h // 4],
        ], np.int32)
        cv2.fillPoly(out, [spike_pts], gold)
        cv2.polylines(out, [spike_pts], True, dgold, 2)

        # jewels
        jewel_positions = [fw // 8, fw // 2, fw * 7 // 8]
        jewel_colors = [(0, 0, 220), (0, 200, 50), (180, 0, 0)]
        for jx, jc in zip(jewel_positions, jewel_colors):
            jy = crown_y + crown_h - crown_h // 8
            cv2.circle(out, (crown_x + jx, jy), crown_h // 12, jc, -1)
            cv2.circle(out, (crown_x + jx, jy), crown_h // 12, (255, 255, 255), 1)

        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 215, 255), 1)
    return out


# ─────────────────────────────────────────────
# 4. Rainbow Filter
# ─────────────────────────────────────────────

def apply_rainbow(frame, faces):
    out = frame.copy()
    t = time.time()
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw // 2
        arc_y = fy - int(fh * 0.05)
        radius_outer = int(fw * 0.8)
        # animated shift
        phase = int((t % 2) * 20)
        colors = [
            (0,   0,   255),   # red
            (0,   69,  255),   # orange
            (0,   255, 255),   # yellow
            (0,   255, 0),     # green
            (255, 0,   0),     # blue
            (255, 0,   128),   # indigo
            (200, 0,   200),   # violet
        ]
        for i, color in enumerate(reversed(colors)):
            r = radius_outer - i * (fw // 12) + phase % (fw // 12)
            if r < 5:
                continue
            cv2.ellipse(out, (cx, arc_y), (r, r), 0, 180 + 10, 360 - 10, color, fw // 16)

        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (255, 100, 200), 1)
    return out


# ─────────────────────────────────────────────
# 5. Neon Filter
# ─────────────────────────────────────────────

def apply_neon(frame, faces):
    # Glowing edge detection on full frame, tinted cyan/purple
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # colour tint: cyan on bright edges
    tinted = edges_col.copy()
    tinted[:, :, 0] = edges       # B
    tinted[:, :, 1] = edges // 2  # G
    tinted[:, :, 2] = 0           # R

    # blur for glow
    glow = cv2.GaussianBlur(tinted, (9, 9), 0)
    out = cv2.addWeighted(frame, 0.4, glow, 0.6, 0)

    # purple tint on face regions
    for (fx, fy, fw, fh) in faces:
        roi = out[fy:fy+fh, fx:fx+fw]
        purple = np.zeros_like(roi)
        purple[:, :, 0] = 180
        purple[:, :, 2] = 80
        out[fy:fy+fh, fx:fx+fw] = cv2.addWeighted(roi, 0.7, purple, 0.3, 0)
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (255, 0, 200), 2)
    return out


# ─────────────────────────────────────────────
# 6. Vintage Filter
# ─────────────────────────────────────────────

def apply_vintage(frame, faces):
    out = frame.copy().astype(np.float32)

    # sepia
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ], dtype=np.float32)
    out = cv2.transform(out, sepia_kernel)
    out = np.clip(out, 0, 255)

    # film grain
    grain = np.random.randn(*frame.shape[:2]).astype(np.float32) * 18
    out[:, :, 0] += grain
    out[:, :, 1] += grain
    out[:, :, 2] += grain
    out = np.clip(out, 0, 255).astype(np.uint8)

    # vignette
    rows, cols = frame.shape[:2]
    sigma = 0.5
    X = cv2.getGaussianKernel(cols, int(cols * sigma))
    Y = cv2.getGaussianKernel(rows, int(rows * sigma))
    kernel = Y * X.T
    mask = kernel / kernel.max()
    out = (out * mask[:, :, np.newaxis]).astype(np.uint8)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (80, 100, 150), 1)
    return out


# ─────────────────────────────────────────────
# 7. Cyberpunk Filter
# ─────────────────────────────────────────────

def apply_cyberpunk(frame, faces):
    out = frame.copy()
    h, w = out.shape[:2]
    t = time.time()

    # scanlines
    for y in range(0, h, 4):
        cv2.line(out, (0, y), (w, y), (0, 0, 0), 1)

    # cyan tint overlay
    tint = np.zeros_like(out)
    tint[:, :, 0] = 20
    tint[:, :, 1] = 40
    out = cv2.addWeighted(out, 0.85, tint, 0.15, 0)

    # HUD corner brackets
    blen = 30
    bcolor = (0, 255, 200)
    bw = 2
    # top-left
    cv2.line(out, (10, 10), (10 + blen, 10), bcolor, bw)
    cv2.line(out, (10, 10), (10, 10 + blen), bcolor, bw)
    # top-right
    cv2.line(out, (w - 10, 10), (w - 10 - blen, 10), bcolor, bw)
    cv2.line(out, (w - 10, 10), (w - 10, 10 + blen), bcolor, bw)
    # bottom-left
    cv2.line(out, (10, h - 10), (10 + blen, h - 10), bcolor, bw)
    cv2.line(out, (10, h - 10), (10, h - 10 - blen), bcolor, bw)
    # bottom-right
    cv2.line(out, (w - 10, h - 10), (w - 10 - blen, h - 10), bcolor, bw)
    cv2.line(out, (w - 10, h - 10), (w - 10, h - 10 - blen), bcolor, bw)

    # animated HUD text
    ts = f"SYS {int(t * 10) % 1000:03d}"
    cv2.putText(out, ts, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bcolor, 1)
    cv2.putText(out, "FACE_ID: ACTIVE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bcolor, 1)

    for (fx, fy, fw, fh) in faces:
        # face corner brackets
        bl = fw // 5
        fc = (0, 240, 255)
        cv2.line(out, (fx, fy), (fx + bl, fy), fc, 2)
        cv2.line(out, (fx, fy), (fx, fy + bl), fc, 2)
        cv2.line(out, (fx + fw, fy), (fx + fw - bl, fy), fc, 2)
        cv2.line(out, (fx + fw, fy), (fx + fw, fy + bl), fc, 2)
        cv2.line(out, (fx, fy + fh), (fx + bl, fy + fh), fc, 2)
        cv2.line(out, (fx, fy + fh), (fx, fy + fh - bl), fc, 2)
        cv2.line(out, (fx + fw, fy + fh), (fx + fw - bl, fy + fh), fc, 2)
        cv2.line(out, (fx + fw, fy + fh), (fx + fw, fy + fh - bl), fc, 2)
        # data readout
        cv2.putText(out, f"W:{fw} H:{fh}", (fx, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 150), 1)
    return out


# ─────────────────────────────────────────────
# 8. Sketch Filter
# ─────────────────────────────────────────────

def apply_sketch(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
    out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (80, 80, 80), 1)
    return out


# ─────────────────────────────────────────────
# 9. Pixelate Filter
# ─────────────────────────────────────────────

def apply_pixelate(frame, faces):
    out = frame.copy()
    block = 12
    for (fx, fy, fw, fh) in faces:
        roi = out[fy:fy+fh, fx:fx+fw]
        small = cv2.resize(roi, (fw // block, fh // block), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        out[fy:fy+fh, fx:fx+fw] = pixelated
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 200, 255), 1)
    return out


# ─────────────────────────────────────────────
# 10. Mirror Filter
# ─────────────────────────────────────────────

def apply_mirror(frame, faces):
    out = frame.copy()
    h, w = out.shape[:2]
    left_half  = out[:, :w // 2]
    right_half = cv2.flip(left_half, 1)

    # pad/trim to fit
    rw = w - w // 2
    if right_half.shape[1] < rw:
        pad = np.zeros((h, rw - right_half.shape[1], 3), dtype=np.uint8)
        right_half = np.hstack([right_half, pad])
    elif right_half.shape[1] > rw:
        right_half = right_half[:, :rw]

    out[:, w // 2:] = right_half

    # mirror line
    cv2.line(out, (w // 2, 0), (w // 2, h), (80, 200, 255), 1)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (80, 200, 255), 1)
    return out


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

FILTER_MAP = {
    "none":       lambda f, faces: f,
    "dog":        apply_dog,
    "sunglasses": apply_sunglasses,
    "crown":      apply_crown,
    "rainbow":    apply_rainbow,
    "neon":       apply_neon,
    "vintage":    apply_vintage,
    "cyberpunk":  apply_cyberpunk,
    "sketch":     apply_sketch,
    "pixelate":   apply_pixelate,
    "mirror":     apply_mirror,
}


def apply_filter(frame, filter_name, faces):
    fn = FILTER_MAP.get(filter_name, FILTER_MAP["none"])
    return fn(frame, faces)
