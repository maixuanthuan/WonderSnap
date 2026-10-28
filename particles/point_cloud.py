import numpy as np


OBJECTS = ["bigben", "pyramids", "turtle_tower", "liberty"]


def _rng(seed):
    return np.random.default_rng(seed)


def _normalize(points):
    points = points.astype(np.float32)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points -= center
    extent = np.max(np.ptp(points, axis=0))
    if extent > 0:
        points /= extent
    return points


# ------------------------------------------------------------- primitives ---
# Sizes are HALF sizes, z is up. Every primitive returns exactly k points (k, 3).

def _split(k, m):
    counts = np.full(m, k // m)
    counts[: k % m] += 1
    return counts


def _accept(k, gen):
    """Call gen(m) (which may reject points) until k points are collected."""
    if k <= 0:
        return np.zeros((0, 3))
    out, got = [], 0
    while got < k:
        p = gen(max(2 * (k - got), 64))
        out.append(p)
        got += len(p)
    return np.concatenate(out)[:k]


def _compose(n, parts):
    """parts = [(weight, fn(k) -> (k, 3))]; splits n points by weight."""
    w = np.array([p[0] for p in parts], dtype=np.float64)
    counts = np.floor(n * w / w.sum()).astype(int)
    counts[np.argmax(w)] += n - counts.sum()
    return np.concatenate([fn(c) for (_, fn), c in zip(parts, counts) if c > 0])


def _four(k, fn):
    """Repeat a part at the 4 corners: fn(count, sx, sy) with sx, sy = +-1."""
    signs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return np.concatenate([fn(c, sx, sy) for c, (sx, sy) in zip(_split(k, 4), signs)])


def _box(k, rng, x, y, z, hx, hy, h, top=True):
    """Surface of an axis-aligned box (4 walls + optional roof), area-weighted."""
    area = np.array([hy * h, hy * h, hx * h, hx * h, 2 * hx * hy if top else 0.0])
    f = rng.choice(5, k, p=area / area.sum())
    u, w, v = rng.uniform(-1, 1, k), rng.uniform(-1, 1, k), rng.uniform(0, 1, k)
    px = np.select([f == 0, f == 1], [-hx, hx], u * hx)
    py = np.select([f == 2, f == 3, f == 4], [-hy, hy, w * hy], u * hy)
    pz = np.where(f == 4, h, v * h)
    return np.column_stack([px + x, py + y, pz + z])


def _frustum(k, rng, x, y, z, h, hx0, hy0, hx1, hy1, curve=1.0):
    """Walls of a rectangular frustum: half size (hx0, hy0) at the bottom -> (hx1, hy1) at the top.
    curve > 1 makes the walls concave (Asian roof / spire)."""
    wmax = max(hx0 + hy0, hx1 + hy1)

    def gen(m):
        t = rng.uniform(0, 1, m)
        s = (1 - t) ** curve
        hx = hx1 + (hx0 - hx1) * s
        hy = hy1 + (hy0 - hy1) * s
        keep = rng.uniform(0, wmax, m) < hx + hy      # more points where the walls are wider
        t, hx, hy = t[keep], hx[keep], hy[keep]
        c = len(t)
        on_x = rng.uniform(0, 1, c) * (hx + hy) < hy
        sign = rng.choice([-1.0, 1.0], c)
        u = rng.uniform(-1, 1, c)
        px = np.where(on_x, sign * hx, u * hx)
        py = np.where(on_x, u * hy, sign * hy)
        return np.column_stack([px + x, py + y, t * h + z])

    return _accept(k, gen)


def _cone(k, rng, x, y, z, h, rx0, ry0, rx1, ry1, folds=0.0):
    """Side of an elliptic cone/cylinder; folds adds vertical drapery ridges."""
    rmax = max(rx0 + ry0, rx1 + ry1)

    def gen(m):
        t = rng.uniform(0, 1, m)
        rx = rx0 + (rx1 - rx0) * t
        ry = ry0 + (ry1 - ry0) * t
        keep = rng.uniform(0, rmax, m) < rx + ry
        t, rx, ry = t[keep], rx[keep], ry[keep]
        a = rng.uniform(0, 2 * np.pi, len(t))
        f = 1.0 + folds * np.sin(9 * a + 3 * t)
        return np.column_stack([x + rx * f * np.cos(a), y + ry * f * np.sin(a), z + t * h])

    return _accept(k, gen)


def _ball(k, rng, c, r, upper=False):
    """Ellipsoid surface; upper=True keeps only the top half (mound)."""
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    if upper:
        d[:, 2] = np.abs(d[:, 2])
    return d * np.asarray(r, dtype=np.float64) + np.asarray(c, dtype=np.float64)


def _tube(k, rng, a, b, r):
    """Thin cylinder from point a to point b (arms, rays, pilasters)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    ref = [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 * np.linalg.norm(d) else [1.0, 0.0, 0.0]
    n1 = np.cross(d, ref)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(d, n1)
    n2 /= np.linalg.norm(n2)
    t = rng.uniform(0, 1, k)[:, None]
    ang = rng.uniform(0, 2 * np.pi, k)[:, None]
    return a + t * d + r * (np.cos(ang) * n1 + np.sin(ang) * n2)


def _disc(k, rng, x, y, z, r0, r1, sy=1.0):
    """Flat ring at height z (water ripples)."""
    r = np.sqrt(rng.uniform(r0 * r0, r1 * r1, k))
    a = rng.uniform(0, 2 * np.pi, k)
    return np.column_stack([x + r * np.cos(a), y + r * sy * np.sin(a), np.full(k, z)])


def _bands(p, rng, z0, z1, n, frac):
    """Snap a fraction of the points to n horizontal lines -> stone courses / floor lines."""
    sel = (p[:, 2] > z0) & (p[:, 2] < z1) & (rng.random(len(p)) < frac)
    step = (z1 - z0) / n
    p[sel, 2] = z0 + np.round((p[sel, 2] - z0) / step) * step
    return p


def _arched(k, rng, x, y, z, hx, hy, h, long_arches=(), short_arches=()):
    """Box walls (no roof) with arched openings.
    arch = (center along the wall, half width, spring height, sill height), heights relative to z.
    long_arches -> walls at y = +-hy, short_arches -> walls at x = +-hx."""
    def gen(m):
        p = _box(m, rng, 0, 0, 0, hx, hy, h, top=False)
        on_x = np.isclose(np.abs(p[:, 0]), hx)
        a = np.where(on_x, p[:, 1], p[:, 0])
        b = p[:, 2]
        hole = np.zeros(m, dtype=bool)
        for arches, face in ((short_arches, on_x), (long_arches, ~on_x)):
            for c, w, s, sill in arches:
                inside = (np.abs(a - c) < w) & (b > sill) & ((b < s) | ((a - c) ** 2 + (b - s) ** 2 < w * w))
                hole |= face & inside
        return p[~hole] + (x, y, z)

    return _accept(k, gen)


def _eaves(k, rng, z, hx, hy, lift):
    """Up-turned roof corners (Vietnamese style)."""
    return _four(k, lambda c, sx, sy: _tube(
        c, rng, (sx * hx, sy * hy, z), (sx * hx * 1.18, sy * hy * 1.18, z + lift), 0.004))


# -------------------------------------------------------------- landmarks ---
def _clock_faces(k, rng, half, zc, rc):
    """4 clock dials (ring + hour marks + two hands) on the sides of a square tower."""
    face = rng.integers(0, 4, k)
    kind = rng.random(k)
    ang = rng.uniform(0, 2 * np.pi, k)
    rad = rc * rng.uniform(0.86, 1.0, k)                              # dial ring
    marks = kind > 0.62
    ang[marks] = rng.integers(0, 12, marks.sum()) * (np.pi / 6)
    rad[marks] = rc * rng.uniform(0.66, 0.80, marks.sum())           # hour marks
    hands = kind > 0.85
    long_hand = rng.random(k) < 0.5
    ang[hands] = np.where(long_hand[hands], np.pi * 0.30, np.pi * 0.85)
    rad[hands] = rc * rng.uniform(0, 1, hands.sum()) * np.where(long_hand[hands], 0.75, 0.5)
    a, b = rad * np.cos(ang), rad * np.sin(ang)
    px = np.select([face == 0, face == 1], [-half, half], a)
    py = np.select([face == 2, face == 3], [-half, half], a)
    return np.column_stack([px, py, zc + b])


def _bigben(n, rng):
    # Elizabeth Tower: square shaft, clock stage with 4 dials, belfry, tall spire with pinnacles.
    p = _compose(n, [
        (28, lambda k: _box(k, rng, 0, 0, 0.00, 0.055, 0.055, 0.60, top=False)),
        (4, lambda k: _four(k, lambda c, sx, sy: _tube(c, rng, (sx * 0.057, sy * 0.057, 0.0), (sx * 0.057, sy * 0.057, 0.60), 0.004))),
        (10, lambda k: _box(k, rng, 0, 0, 0.60, 0.070, 0.070, 0.13)),
        (18, lambda k: _clock_faces(k, rng, 0.071, 0.665, 0.050)),
        (9, lambda k: _box(k, rng, 0, 0, 0.73, 0.060, 0.060, 0.07)),
        (20, lambda k: _frustum(k, rng, 0, 0, 0.80, 0.19, 0.060, 0.060, 0.004, 0.004, curve=1.3)),
        (6, lambda k: _four(k, lambda c, sx, sy: _frustum(c, rng, sx * 0.06, sy * 0.06, 0.80, 0.06, 0.009, 0.009, 0.0, 0.0))),
        (2, lambda k: _tube(k, rng, (0, 0, 0.99), (0, 0, 1.03), 0.003)),
    ])
    return _bands(p, rng, 0.0, 0.60, 16, 0.25)


def _pyramids(n, rng):
    # Giza: Khufu in front, Khafre and Menkaure behind, stone courses on the faces.
    def pyr(k, x, y, half, h):
        p = _frustum(k, rng, x, y, 0.0, h, half, half, 0.0, 0.0)
        return _bands(p, rng, 0.0, h, max(int(h / 0.028), 4), 0.35)

    return _compose(n, [
        (55, lambda k: pyr(k, 0.00, 0.00, 0.40, 0.51)),
        (30, lambda k: pyr(k, 0.68, 0.48, 0.24, 0.31)),
        (15, lambda k: pyr(k, 1.05, 0.82, 0.12, 0.15)),
    ])


def _turtle_tower(n, rng):
    # Thap Rua (Ho Guom): 3 storeys with arched openings, curved roofs, small island in the lake.
    return _compose(n, [
        (10, lambda k: _ball(k, rng, (0, 0, 0), (0.42, 0.32, 0.05), upper=True)),      # island
        (5, lambda k: _disc(k, rng, 0, 0, 0.0, 0.46, 0.58, sy=0.78)),                  # ripples
        (22, lambda k: _arched(k, rng, 0, 0, 0.05, 0.22, 0.15, 0.20,
                               long_arches=[(-0.13, 0.04, 0.10, 0.0), (0.0, 0.05, 0.11, 0.0), (0.13, 0.04, 0.10, 0.0)],
                               short_arches=[(0.0, 0.05, 0.11, 0.0)])),
        (9, lambda k: _frustum(k, rng, 0, 0, 0.25, 0.04, 0.26, 0.19, 0.15, 0.11, curve=1.8)),
        (3, lambda k: _eaves(k, rng, 0.25, 0.26, 0.19, 0.03)),
        (13, lambda k: _arched(k, rng, 0, 0, 0.29, 0.14, 0.10, 0.13,
                               long_arches=[(0.0, 0.045, 0.07, 0.02)],
                               short_arches=[(0.0, 0.035, 0.07, 0.02)])),
        (6, lambda k: _frustum(k, rng, 0, 0, 0.42, 0.035, 0.18, 0.14, 0.10, 0.08, curve=1.8)),
        (2, lambda k: _eaves(k, rng, 0.42, 0.18, 0.14, 0.025)),
        (8, lambda k: _arched(k, rng, 0, 0, 0.455, 0.09, 0.07, 0.09,
                              long_arches=[(0.0, 0.03, 0.045, 0.015)],
                              short_arches=[(0.0, 0.025, 0.045, 0.015)])),
        (9, lambda k: _frustum(k, rng, 0, 0, 0.545, 0.10, 0.13, 0.11, 0.012, 0.012, curve=2.0)),
        (2, lambda k: _eaves(k, rng, 0.545, 0.13, 0.11, 0.03)),
        (2, lambda k: _ball(k, rng, (0, 0, 0.665), (0.014, 0.014, 0.022))),             # finial
    ])


def _crown(k, rng, head):
    """7 rays of the crown, fanned around the front of the head (front = -y)."""
    head = np.asarray(head, dtype=np.float64)
    parts = []
    for c, phi in zip(_split(k, 7), np.linspace(-1.9, 1.9, 7)):
        d = np.array([np.sin(phi) * np.cos(0.45), -np.cos(phi) * np.cos(0.45), np.sin(0.45)])
        a = head + d * 0.022 + (0.0, 0.0, 0.01)
        parts.append(_tube(c, rng, a, a + d * 0.045, 0.0035))
    return np.concatenate(parts)


def _liberty(n, rng):
    # Statue of Liberty: stepped pedestal, draped robe, crowned head, raised torch, tablet.
    head = (0.0, 0.0, 0.835)
    return _compose(n, [
        (7, lambda k: _frustum(k, rng, 0, 0, 0.00, 0.06, 0.19, 0.19, 0.16, 0.16)),
        (5, lambda k: _box(k, rng, 0, 0, 0.06, 0.13, 0.13, 0.05)),
        (12, lambda k: _bands(_frustum(k, rng, 0, 0, 0.11, 0.24, 0.105, 0.105, 0.09, 0.09), rng, 0.11, 0.35, 8, 0.25)),
        (4, lambda k: _box(k, rng, 0, 0, 0.35, 0.11, 0.11, 0.025)),
        (4, lambda k: _frustum(k, rng, 0, 0, 0.375, 0.045, 0.085, 0.085, 0.075, 0.075)),
        (22, lambda k: _cone(k, rng, 0, 0, 0.42, 0.36, 0.07, 0.055, 0.045, 0.032, folds=0.08)),   # robe
        (5, lambda k: _ball(k, rng, (0, 0, 0.785), (0.05, 0.035, 0.022))),                       # shoulders
        (5, lambda k: _ball(k, rng, head, (0.022, 0.022, 0.027))),
        (6, lambda k: _crown(k, rng, head)),
        (8, lambda k: _tube(k, rng, (0.040, 0.0, 0.79), (0.055, 0.01, 0.93), 0.012)),            # raised arm
        (2, lambda k: _tube(k, rng, (0.055, 0.01, 0.93), (0.058, 0.012, 0.955), 0.009)),         # torch handle
        (2, lambda k: _cone(k, rng, 0.058, 0.012, 0.955, 0.012, 0.010, 0.010, 0.018, 0.018)),    # torch cup
        (4, lambda k: _ball(k, rng, (0.058, 0.012, 0.985), (0.013, 0.013, 0.022))),              # flame
        (4, lambda k: _tube(k, rng, (-0.045, 0.0, 0.78), (-0.06, -0.02, 0.70), 0.011)),          # left arm
        (5, lambda k: _box(k, rng, -0.065, -0.03, 0.64, 0.012, 0.030, 0.09)),                    # tablet
    ])


_BUILDERS = {
    "bigben": _bigben,
    "pyramids": _pyramids,
    "turtle_tower": _turtle_tower,
    "liberty": _liberty,
}


def build_point_cloud(name, n, seed=17):
    rng = _rng(seed)
    pts = _normalize(_BUILDERS.get(name.lower(), _bigben)(n, rng))

    # Deterministic micro variation prevents a perfectly synthetic look.
    pts += rng.normal(0.0, 0.0025, pts.shape).astype(np.float32)

    # Order by height layer, then by angle: particle i sits at a similar place in every
    # wonder, so morphs flow layer-to-layer instead of crossing chaotically.
    order = np.lexsort((np.arctan2(pts[:, 1], pts[:, 0]), np.round((pts[:, 2] + 0.5) * 60)))
    return np.ascontiguousarray(pts[order], dtype=np.float32)