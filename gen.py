"""Procedural: a colossal molten plasma ribbon being drawn into orbit around a
small dark world.

1024x1024, rendered at 2x then downsampled. numpy + Pillow only, all seeded.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FINAL = 1024
S = 2
WS = FINAL * S


# ---------------------------------------------------------------- geometry
def catmull(pts, n):
    pts = np.asarray(pts, np.float32)
    p = np.vstack([pts[0], pts, pts[-1]])
    seg = len(p) - 3
    per = max(2, n // seg)
    out = []
    for i in range(seg):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t = np.linspace(0, 1, per, endpoint=(i == seg - 1))[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def frames(path):
    d = np.gradient(path, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    return d, nrm


def band_mask(path, widths):
    d, nrm = frames(path)
    left = path + nrm * widths[:, None]
    right = path - nrm * widths[:, None]
    poly = np.vstack([left, right[::-1]])
    im = Image.new("L", (WS, WS), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=255)
    return im


def blur_f(im, r):
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), np.float32) / 255.0


def glow_sprite(r, falloff=2.2):
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / max(r, 1)
    return np.clip(1 - d, 0, 1) ** falloff


def stamp(img, cx, cy, sprite_np, color, gain):
    r = sprite_np.shape[0] // 2
    x0, y0 = int(cx) - r, int(cy) - r
    x1, y1 = x0 + sprite_np.shape[0], y0 + sprite_np.shape[0]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(WS, x1), min(WS, y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = sprite_np[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    img[y0:y1, x0:x1] += sub[..., None] * np.asarray(color, np.float32) * gain


def noise_field(shape, cell, rng, smooth=Image.BICUBIC):
    """Smooth value noise: random grid upsampled."""
    h, w = shape
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    g = rng.uniform(0, 1, (gh, gw)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), smooth)
    return np.asarray(im, np.float32) / 255.0


def fbm_aniso(xx, yy, rng, theta, along, across, octaves=4):
    """fbm in a rotated, anisotropic frame -> streaks aligned with `theta`."""
    ct, st = np.cos(theta), np.sin(theta)
    u = xx * ct + yy * st
    v = -xx * st + yy * ct
    out = np.zeros(xx.shape, np.float32)
    amp, cell = 1.0, 1.0
    for o in range(octaves):
        # one noise field, sampled through the rotated coords via a pre-rendered tile
        h, w = xx.shape
        # build a field directly as a function of u,v by using a coarse random grid
        gh = max(2, int(h / (64 * cell)) + 2)
        gw = max(2, int(w / (64 * cell)) + 2)
        g = rng.uniform(0, 1, (gh, gw)).astype(np.float32)
        gu = (u / (along * cell)) % 1.0 * (gw - 1)
        gv = (v / (across * cell)) % 1.0 * (gh - 1)
        i0 = gv.astype(np.int32); j0 = gu.astype(np.int32)
        i1 = np.clip(i0 + 1, 0, gh - 1); j1 = np.clip(j0 + 1, 0, gw - 1)
        fu = gu - j0; fv = gv - i0
        fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
        out += amp * (g[i0, j0] * (1 - fu) * (1 - fv) + g[i0, j1] * fu * (1 - fv)
                      + g[i1, j0] * (1 - fu) * fv + g[i1, j1] * fu * fv)
        amp *= 0.55
        cell *= 2.1
    return out


# ---------------------------------------------------------------- scene
def render(variant=0, seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((WS, WS, 3), np.float32)
    yy, xx = np.mgrid[0:WS, 0:WS].astype(np.float32)

    # ---- world placement (declared early: everything reacts to it)
    ang_map = [(790, 300, 96), (300, 250, 100), (800, 292, 84)]
    pcx, pcy, pr1024 = ang_map[variant % 3]
    pc = np.array([pcx, pcy], np.float32) * S
    pr = pr1024 * S

    # ---- sky
    top = np.array([6, 2, 7], np.float32)
    bot = np.array([26, 5, 15], np.float32)
    t = (yy / WS)[..., None]
    img += top * (1 - t) + bot * t

    for _ in range(7):
        cx, cy = rng.uniform(0, WS), rng.uniform(0, WS)
        rad = rng.uniform(0.18, 0.5) * WS
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad * rad)))
        col = np.array([rng.uniform(40, 80), rng.uniform(4, 12), rng.uniform(20, 50)], np.float32)
        img += g[..., None] * col * 0.20
    g = np.exp(-(((xx - WS * 0.85) ** 2 + (yy - WS * 0.18) ** 2) / (2 * (0.4 * WS) ** 2)))
    img += g[..., None] * np.array([8, 16, 40], np.float32) * 0.35

    # ---- stars
    spr_small = glow_sprite(2, 2.0)
    spr_big = glow_sprite(7, 2.6)
    for _ in range(1400):
        x, y = rng.uniform(0, WS), rng.uniform(0, WS)
        b = rng.uniform(0.15, 1.0) ** 2.2
        tint = np.array([1.0, rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0)], np.float32)
        if b > 0.8:
            stamp(img, x, y, spr_big, tint * 110, b * 0.8)
        stamp(img, x, y, spr_small, tint * 200, b)
    for _ in range(3):
        x, y = rng.uniform(0.06, 0.92) * WS, rng.uniform(0.02, 0.35) * WS
        stamp(img, x, y, spr_big, np.array([150, 165, 255], np.float32), 2.0)
        for a in range(6):
            ang = a * np.pi / 6 + rng.uniform(0, .3)
            for k in range(2, 24):
                px, py = int(x + np.cos(ang) * k), int(y + np.sin(ang) * k)
                if 0 <= px < WS and 0 <= py < WS:
                    img[py, px] += np.array([60, 70, 130], np.float32) * (1 - k / 24) * 0.5

    # ---- main ribbon: colossal at lower-left, thinning toward the world
    ctrl = [
        (-220, 1220), (-70, 1055), (90, 915), (235, 790),
        (360, 668), (455, 566), (530, 486), (592, 428),
        (650, 388), (706, 362), (758, 348), (818, 340),
    ]
    if variant == 1:
        ctrl = [(a, b * 1.05 - 26) for a, b in ctrl]
    if variant == 2:
        ctrl = [(a * 0.95 + 24, b * 1.03 - 14) for a, b in ctrl]
    path = catmull(np.array(ctrl, np.float32) * S, 1600)
    n = len(path)
    s = np.linspace(0, 1, n)
    w_main = np.interp(s, [0.0, 0.12, 0.3, 0.5, 0.66, 0.78, 0.9, 1.0],
                       [52, 82, 92, 74, 50, 34, 20, 9]) * S
    d, nrm = frames(path)

    # ---- siphon: the ribbon's tip spirals into orbit around the world
    tip = path[-1]
    a0 = np.arctan2(tip[1] - pc[1], tip[0] - pc[0])
    turns = 1.55
    ts = np.linspace(0, 1, 260)
    radii = pr * (1.22 - 0.62 * ts) ** 1.0
    a = a0 + turns * 2 * np.pi * ts ** 1.35
    # ease the spiral out of the tip so there's no visible seam
    spiral = np.stack([pc[0] + np.cos(a) * radii, pc[1] + np.sin(a) * radii], 1).astype(np.float32)
    blend = np.clip(ts / 0.14, 0, 1)[:, None]
    spiral = tip * (1 - blend) + spiral * blend
    spath = np.vstack([path[-14:], spiral[6:]])
    sn_ = len(spath)
    ss = np.linspace(0, 1, sn_)
    w_spiral = np.interp(ss, [0, 0.1, 0.3, 1.0], [9, 8, 5.5, 1.6]) * S

    # ---- heat fields (main band + siphon), combined
    m_main = blur_f(band_mask(path, w_main * 1.28), 2.0 * S)
    m_sip = blur_f(band_mask(spath, w_spiral * 1.5), 1.6 * S)
    m_hot = np.clip(m_main / (m_main.max() + 1e-9) + 0.85 * m_sip / (m_sip.max() + 1e-9), 0, 1)
    m_hot = m_hot ** 0.85

    # flow-aligned turbulence: streaks run along the ribbon, not across it
    theta = np.arctan2(*(path[-1] - path[len(path) // 3])[::-1])
    turb = fbm_aniso(xx, yy, rng, theta, along=150.0, across=26.0, octaves=4)
    turb = (turb - turb.min()) / (np.ptp(turb) + 1e-9)
    fine = fbm_aniso(xx, yy, rng, theta, along=46.0, across=11.0, octaves=3)
    fine = (fine - fine.min()) / (np.ptp(fine) + 1e-9)
    tex = 0.7 * turb + 0.3 * fine
    heat = np.clip(m_hot * (0.55 + 0.85 * tex), 0, 1.0)

    def molten(h):
        return np.stack([
            np.clip(h * 1.35, 0, 1) * 255,
            np.clip(h * 1.5 - 0.42, 0, 1) * 240,
            np.clip(h * 2.0 - 1.05, 0, 1) * 235,
        ], -1)

    # wide corona + band glow
    m_wide = band_mask(path, w_main * 2.6)
    m_wide_s = band_mask(spath, w_spiral * 5.0)
    glow_wide = np.clip(blur_f(m_wide, 26 * S) + 0.8 * blur_f(m_wide_s, 8 * S), 0, 1)
    img += glow_wide[..., None] * np.array([255, 24, 132], np.float32) * 0.30
    img += blur_f(band_mask(path, w_main * 1.5), 7 * S)[..., None] * np.array([190, 24, 84], np.float32) * 0.55
    img += molten(heat) * 1.05

    # ---- flow filaments: thin strands advected along the band
    fil_im = Image.new("RGB", (WS, WS), (0, 0, 0))
    dr = ImageDraw.Draw(fil_im)
    for _ in range(360):
        i0 = rng.integers(0, n - 40)
        ln = int(rng.uniform(40, 240))
        off = rng.normal(0, 0.7)
        ph = rng.uniform(0, 6.28)
        amp = rng.uniform(0.02, 0.08)
        pts = []
        for k in range(0, ln, 6):
            j = min(i0 + k, n - 1)
            o = off + amp * np.sin(ph + s[j] * 22)
            pts.append(tuple(path[j] + nrm[j] * (o * w_main[j])))
        if len(pts) > 1:
            hot = rng.uniform(0.4, 1.0)
            dr.line(pts, fill=(int(255 * hot), int(230 * hot), int(190 * hot)),
                    width=int(rng.integers(1, 3)), joint="curve")
    fl = np.asarray(fil_im.filter(ImageFilter.GaussianBlur(0.8 * S)), np.float32) / 255.0
    img += fl * np.clip(m_hot * 1.4, 0, 1)[..., None] * 1.6

    # ---- embers shed around the flow and the siphon
    spr_e = glow_sprite(6, 2.4)
    for _ in range(520):
        if rng.random() < 0.62:
            i = rng.integers(0, n)
            p = path[i] + nrm[i] * rng.normal(0, 1) * w_main[i] * rng.uniform(0.9, 3.2)
            p = p + d[i] * rng.normal(0, 26) * S
        else:
            i = rng.integers(0, sn_)
            _, nn = frames(spath)
            p = spath[i] + nn[i] * rng.normal(0, 1) * w_spiral[i] * rng.uniform(1.2, 4.0)
        if not (0 <= p[0] < WS and 0 <= p[1] < WS):
            continue
        he = rng.uniform(0.25, 1.0) ** 2
        stamp(img, p[0], p[1], spr_e, np.array([255, 120 + 110 * he, 50 + 90 * he], np.float32), he * 1.9)

    # ---- the world: dark body, blue rim, ribbon light spilling on it
    dist = np.sqrt((xx - pc[0]) ** 2 + (yy - pc[1]) ** 2)

    halo = np.exp(-((dist - pr) / (pr * 0.7)) ** 2)
    img += halo[..., None] * np.array([70, 160, 255], np.float32) * 0.22
    halo2 = np.exp(-((dist - pr) / (pr * 2.0)) ** 2)
    img += halo2[..., None] * np.array([34, 70, 200], np.float32) * 0.09

    ldir = np.array([-0.82, 0.57], np.float32)
    nd = np.clip(((xx - pc[0]) * ldir[0] + (yy - pc[1]) * ldir[1]) / pr, -1, 1)
    body = np.array([15, 6, 13], np.float32) + np.clip(nd, 0, 1)[..., None] * np.array([46, 18, 26], np.float32)
    disk = dist < pr
    img[disk] = body[disk]

    ang = np.arctan2(yy - pc[1], xx - pc[0])
    toward = np.arctan2(path[-1][1] - pc[1], path[-1][0] - pc[0])
    facing = np.clip(np.cos(ang - toward), -1, 1)
    rim = np.exp(-((dist - pr) / (1.8 * S)) ** 2) * (0.22 + 0.78 * np.clip(facing, 0, 1) ** 1.3)
    img += rim[..., None] * np.array([80, 200, 255], np.float32) * 1.9

    # impact kiss on the limb nearest the ribbon
    kx, ky = pc[0] - pr * 0.66, pc[1] + pr * 0.60
    kiss = np.exp(-(((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * (8 * S) ** 2)))
    img += kiss[..., None] * np.array([255, 236, 214], np.float32) * 1.4

    # ---- grade
    img = np.tanh(img / 170.0) * 1.02
    r = np.sqrt((xx - WS * 0.5) ** 2 + (yy - WS * 0.46) ** 2) / (WS * 0.72)
    img *= np.clip(1.14 - 0.5 * r ** 2.2, 0.42, 1.0)[..., None]

    out = np.clip(img, 0, 1) * 255
    return Image.fromarray(out.astype(np.uint8)).resize((FINAL, FINAL), Image.LANCZOS)


if __name__ == "__main__":
    for v, sd in [(0, 11), (1, 23), (2, 37)]:
        im = render(v, sd)
        im.save(f"variant_{v}.png", optimize=True)
        print(v, im.size)
