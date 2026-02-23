# invert_frames_np.py
# NumPy-only inversion (180° about Z) + boost pad symmetry + team-tag swap.
# Designed to accept a batch from replaydfframesxyz: typically (x, y, z) or just x.

import math
import numpy as np

# ----------------------------
# Feature layout (from the uploaded *featurescaling*.txt files)
# ----------------------------

FEAT_DIMS = {
    # because who_scores_next is listed at 144/208/272 (i.e., after the feature vector)
    "duel": 144,     # who_scores_next=144, frames_until_stoppage=145
    "doubles": 208,  # who_scores_next=208, frames_until_stoppage=209
    "standard": 272, # who_scores_next=272, frames_until_stoppage=273
}

XYZ_BLOCKS = {
    # each tuple is (start, end) for a block stored as [x...][y...][z...]
    "duel": {
        "ang_vel": (0, 9),
        "pos":     (9, 33),
        "rot":     (33, 42),
        "vel":     (42, 66),
    },
    "doubles": {
        "ang_vel": (0, 15),
        "pos":     (15, 45),
        "rot":     (45, 60),
        "vel":     (60, 90),
    },
    "standard": {
        "ang_vel": (0, 21),
        "pos":     (21, 57),
        "rot":     (57, 78),
        "vel":     (78, 114),
    },
}

# boost pad timer columns (34 cols, in your feature-order)
BOOST_TIMER_SLICE = {
    "duel":    (68, 102),   # boost_0_timer=68 ... boost_9_timer=101
    "doubles": (94, 128),   # boost_0_timer=94 ... boost_9_timer=127
    "standard":(120, 154),  # boost_0_timer=120 ... boost_9_timer=153
}

# This permutation is self-inverse (applying it twice returns identity)
BOOST_TIMER_PERM = [
    26, 15, 14, 13, 12, 11, 10, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 33, 31, 30,
    29, 28, 27, 32,  2,  1,  0,  7,  6,  5,
     4,  3,  9,  8
]

BOOSTCOLLECT_TAG_COLS = {
    "duel":    (106, 108),   # 106,107
    "doubles": (136, 140),   # 136..139
    "standard":(166, 172),   # 166..171
}

TEAM_TAG_COLS = {
    "duel":    (137, 139),   # 137,138
    "doubles": (195, 199),   # 195..198
    "standard":(253, 259),   # 253..258
}

# ----------------------------
# Helpers
# ----------------------------

_TWO_PI = 2.0 * math.pi

def _wrap_to_pi_np(theta: np.ndarray) -> np.ndarray:
    # maps to (-pi, pi]
    return np.remainder(theta + math.pi, _TWO_PI) - math.pi

def _view_2d_np(x: np.ndarray):
    # supports x shaped (N,F) or (B,T,F) etc
    if x.ndim == 2:
        return x, None
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape

def _restore_shape_np(x2d: np.ndarray, shape):
    if shape is None:
        return x2d
    return x2d.reshape(*shape)

def _flip_xy_in_xyz_block_np(x2d: np.ndarray, start: int, end: int):
    # block layout: [x...][y...][z...]
    L = end - start
    n = L // 3
    if n <= 0 or (3 * n) != L:
        raise ValueError(f"XYZ block ({start},{end}) length {L} is not divisible by 3.")
    x2d[:, start:start+n] *= -1
    x2d[:, start+n:start+2*n] *= -1
    # z stays

def _add_pi_to_yaw_in_rot_block_np(x2d: np.ndarray, start: int, end: int):
    # rot block layout: [rot_x...][rot_y...][rot_z...]
    L = end - start
    n = L // 3
    if n <= 0 or (3 * n) != L:
        raise ValueError(f"ROT block ({start},{end}) length {L} is not divisible by 3.")
    yaw_sl = slice(start + n, start + 2*n)  # rot_y_*
    x2d[:, yaw_sl] = _wrap_to_pi_np(x2d[:, yaw_sl] + math.pi)

def _permute_boost_timers_np(x2d: np.ndarray, start: int, end: int):
    perm = np.asarray(BOOST_TIMER_PERM, dtype=np.int64)
    tmp = x2d[:, start:end].copy()
    x2d[:, start:end] = tmp[:, perm]

def _invert_boostcollect_tags_np(x2d: np.ndarray, start: int, end: int):
    # values are encoded as k/34 with k in [0..34]
    t = x2d[:, start:end]
    k = np.rint(t * 34.0).astype(np.int64)
    k = np.clip(k, 0, 34)
    # 0 means "no boost collected" -> stays 0; else k -> 35-k
    k_inv = np.where(k == 0, k, 35 - k)
    x2d[:, start:end] = (k_inv.astype(t.dtype) / np.asarray(34.0, dtype=t.dtype))

def _invert_team_tags_np(x2d: np.ndarray, start: int, end: int):
    # 0 <-> 1
    x2d[:, start:end] = np.asarray(1.0, dtype=x2d.dtype) - x2d[:, start:end]

def _invert_who_scores_next_np(y: np.ndarray) -> np.ndarray:
    # works for y in {0,1} and keeps y=0.5 fixed
    y = np.asarray(y)
    return np.asarray(1.0, dtype=y.dtype) - y

# ----------------------------
# Public API
# ----------------------------

def invert_x(x: np.ndarray, gamemode: str) -> np.ndarray:
    """
    Invert features x under 180° rotation about Z + boost-pad symmetry + team swap tag.
    x can be (N,F) or (...,F). Returns same shape as x.
    """
    if gamemode not in FEAT_DIMS:
        raise ValueError(f"Unknown gamemode '{gamemode}'. Expected one of {list(FEAT_DIMS.keys())}.")

    x = np.asarray(x)
    if x.shape[-1] != FEAT_DIMS[gamemode]:
        raise ValueError(f"x has F={x.shape[-1]} but expected F={FEAT_DIMS[gamemode]} for {gamemode}.")

    # preserve dtype; ensure float math works
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)

    out = x.copy()
    x2d, shape = _view_2d_np(out)

    # kinematics: flip x,y for ang_vel/pos/vel
    blocks = XYZ_BLOCKS[gamemode]
    _flip_xy_in_xyz_block_np(x2d, *blocks["ang_vel"])
    _flip_xy_in_xyz_block_np(x2d, *blocks["pos"])
    _flip_xy_in_xyz_block_np(x2d, *blocks["vel"])

    # orientation: add pi to yaw (rot_y_*)
    _add_pi_to_yaw_in_rot_block_np(x2d, *blocks["rot"])

    # boost pad timers: permute
    bt0, bt1 = BOOST_TIMER_SLICE[gamemode]
    _permute_boost_timers_np(x2d, bt0, bt1)

    # boostcollect tags: invert k -> 35-k (except k=0)
    tg0, tg1 = BOOSTCOLLECT_TAG_COLS[gamemode]
    _invert_boostcollect_tags_np(x2d, tg0, tg1)

    # team tags: 0 <-> 1
    tm0, tm1 = TEAM_TAG_COLS[gamemode]
    _invert_team_tags_np(x2d, tm0, tm1)

    return _restore_shape_np(x2d, shape)

def invert_batch_np(batch, gamemode: str, invert_labels: bool = True):
    """
    Accepts either:
      - x
      - (x, y)
      - (x, y, z)
      - (x, y, z, *extras)

    Returns same structure with x inverted; y inverted iff invert_labels=True.
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty batch tuple/list.")
        x = batch[0]
        y = batch[1] if len(batch) >= 2 else None
        z = batch[2] if len(batch) >= 3 else None
        extras = tuple(batch[3:]) if len(batch) > 3 else ()

        x_inv = invert_x(x, gamemode)

        if y is not None and invert_labels:
            y_inv = _invert_who_scores_next_np(y)
        else:
            y_inv = y

        if len(batch) == 1:
            return (x_inv,)
        if len(batch) == 2:
            return (x_inv, y_inv)
        if len(batch) == 3:
            return (x_inv, y_inv, z)
        return (x_inv, y_inv, z, *extras)

    # batch is just x
    return invert_x(batch, gamemode)

# ----------------------------
# Minimal sanity test (involution: T(T(x)) == x)
# ----------------------------

def _test_involution_np(seed: int = 0):
    rng = np.random.default_rng(seed)

    for gm, F in FEAT_DIMS.items():
        x = rng.standard_normal((4096, F), dtype=np.float32)

        # make team tags binary
        tm0, tm1 = TEAM_TAG_COLS[gm]
        x[:, tm0:tm1] = rng.integers(0, 2, size=(x.shape[0], tm1 - tm0), dtype=np.int64).astype(np.float32)

        # make boostcollect tags valid k/34
        tg0, tg1 = BOOSTCOLLECT_TAG_COLS[gm]
        k = rng.integers(0, 35, size=(x.shape[0], tg1 - tg0), dtype=np.int64)
        x[:, tg0:tg1] = (k.astype(np.float32) / 34.0)

        # make yaw in [-pi, pi]
        r0, r1 = XYZ_BLOCKS[gm]["rot"]
        n = (r1 - r0) // 3
        yaw_sl = slice(r0 + n, r0 + 2*n)
        x[:, yaw_sl] = (rng.random(size=x[:, yaw_sl].shape, dtype=np.float32) * _TWO_PI) - math.pi

        y = np.tile(np.asarray([0.0, 0.5, 1.0], dtype=np.float32), x.shape[0] // 3 + 1)[: x.shape[0]]

        x2 = invert_x(invert_x(x, gm), gm)
        y2 = _invert_who_scores_next_np(_invert_who_scores_next_np(y))

        max_dx = np.max(np.abs(x2 - x))
        max_dy = np.max(np.abs(y2 - y))
        print(f"[{gm}] max|T(T(x))-x|={max_dx:.3e}  max|T(T(y))-y|={max_dy:.3e}")

if __name__ == "__main__":
    _test_involution_np()
