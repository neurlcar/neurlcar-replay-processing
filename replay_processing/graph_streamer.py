import numpy as np
from typing import List, Tuple, Dict, Any

NODE_INDEX_MAP = {
    "duel": {
        "ball": [
            # ang_vel (ball)
            0, 3, 6,
            # pos (ball)
            9, 17, 25,
            # rot (ball)
            33, 36, 39,
            # vel (ball)
            42, 50, 58,
            # pos_x_futureball_1..5
            11, 12, 13, 14, 15,
            # pos_y_futureball_1..5
            19, 20, 21, 22, 23,
            # pos_z_futureball_1..5
            27, 28, 29, 30, 31,
            # vel_x_futureball_1..5
            44, 45, 46, 47, 48,
            # vel_y_futureball_1..5
            52, 53, 54, 55, 56,
            # vel_z_futureball_1..5
            60, 61, 62, 63, 64,
        ],
        "blue_0": [
            # ang_vel
            1, 4, 7,
            # pos
            10, 18, 26,
            # rot
            34, 37, 40,
            # vel
            43, 51, 59,
            # ball cam
            66,
            # boost-related (per-player)
            102, 103, 106, 108,
            # physical state
            111, 113, 115, 117,   # demo, grounded, handbrake, hitbox
            # jump state
            120, 122, 123, 125, 128,
            # orientation & control
            130, 132, 135, 137, 139, 142,  # pitch, roll, steer, team, throttle, yaw
        ],
        "orange_0": [
            # ang_vel
            2, 5, 8,
            # pos
            16, 24, 32,
            # rot
            35, 38, 41,
            # vel
            49, 57, 65,
            # ball cam
            67,
            # boost-related
            105, 104, 107, 109,   # boost, boost_collected, boostcollect_tag, boosting
            # physical state
            112, 114, 116, 118,   # demo, grounded, handbrake, hitbox
            # jump state
            121, 127, 124, 126, 129,
            # orientation & control
            131, 133, 136, 138, 140, 143,  # pitch, roll, steer, team, throttle, yaw
        ],
        "boost": list(range(68, 102)),  # 68..101
        "global": [
            110,  # delta
            119,  # is_overtime
            134,  # seconds_remaining
            141,  # time
        ],
    },

    "doubles": {
        "ball": [
            # ang_vel
            0, 5, 10,
            # pos (ball + futureball_1..5)
            15, 18, 19, 20, 21, 22,
            25, 28, 29, 30, 31, 32,
            35, 38, 39, 40, 41, 42,
            # rot
            45, 50, 55,
            # vel (ball + futureball_1..5)
            60, 63, 64, 65, 66, 67,
            70, 73, 74, 75, 76, 77,
            80, 83, 84, 85, 86, 87,
        ],
        "blue_0": [
            # ang_vel
            1, 6, 11,
            # pos
            16, 26, 36,
            # rot
            46, 51, 56,
            # vel
            61, 71, 81,
            # ball cam
            90,
            # boost-related
            128, 130, 136, 140,
            # physical state
            145, 149, 153, 157,
            # jump state
            162, 166, 168, 172, 178,
            # orientation & control
            182, 186, 191, 195, 199, 204,
        ],
        "blue_1": [
            # ang_vel
            2, 7, 12,
            # pos
            17, 27, 37,
            # rot
            47, 52, 57,
            # vel
            62, 72, 82,
            # ball cam
            91,
            # boost-related
            129, 131, 137, 141,
            # physical state
            146, 150, 154, 158,
            # jump state
            163, 167, 169, 173, 179,
            # orientation & control
            183, 187, 192, 196, 200, 205,
        ],
        "orange_0": [
            # ang_vel
            3, 8, 13,
            # pos
            23, 33, 43,
            # rot
            48, 53, 58,
            # vel
            68, 78, 88,
            # ball cam
            92,
            # boost-related
            134, 132, 138, 142,
            # physical state
            147, 151, 155, 159,
            # jump state
            164, 176, 170, 174, 180,
            # orientation & control
            184, 188, 193, 197, 201, 206,
        ],
        "orange_1": [
            # ang_vel
            4, 9, 14,
            # pos
            24, 34, 44,
            # rot
            49, 54, 59,
            # vel
            69, 79, 89,
            # ball cam
            93,
            # boost-related
            135, 133, 139, 143,
            # physical state
            148, 152, 156, 160,
            # jump state
            165, 177, 171, 175, 181,
            # orientation & control
            185, 189, 194, 198, 202, 207,
        ],
        "boost": list(range(94, 128)),  # 94..127
        "global": [
            144,  # delta
            161,  # is_overtime
            190,  # seconds_remaining
            203,  # time
        ],
    },

    "standard": {
        "ball": [
            # ang_vel
            0, 7, 14,
            # pos (ball + futureball_1..5)
            21, 25, 26, 27, 28, 29,
            33, 37, 38, 39, 40, 41,
            45, 49, 50, 51, 52, 53,
            # rot
            57, 64, 71,
            # vel (ball + futureball_1..5)
            78, 82, 83, 84, 85, 86,
            90, 94, 95, 96, 97, 98,
            102, 106, 107, 108, 109, 110,
        ],
        "blue_0": [
            1, 8, 15,      # ang_vel
            22, 34, 46,    # pos
            58, 65, 72,    # rot
            79, 91, 103,   # vel
            114,           # ball cam
            154, 157, 166, 172,  # boost-related
            179, 185, 191, 197,  # demo, grounded, handbrake, hitbox
            204, 210, 213, 219, 228,  # jump state
            234, 240, 247, 253, 259, 266,  # pitch, roll, steer, team, throttle, yaw
        ],
        "blue_1": [
            2, 9, 16,
            23, 35, 47,
            59, 66, 73,
            80, 92, 104,
            115,
            155, 158, 167, 173,
            180, 186, 192, 198,
            205, 211, 214, 220, 229,
            235, 241, 248, 254, 260, 267,
        ],
        "blue_2": [
            3, 10, 17,
            24, 36, 48,
            60, 67, 74,
            81, 93, 105,
            116,
            156, 159, 168, 174,
            181, 187, 193, 199,
            206, 212, 215, 221, 230,
            236, 242, 249, 255, 261, 268,
        ],
        "orange_0": [
            4, 11, 18,
            30, 42, 54,
            61, 68, 75,
            87, 99, 111,
            117,
            163, 160, 169, 175,
            182, 188, 194, 200,
            207, 225, 216, 222, 231,
            237, 243, 250, 256, 262, 269,
        ],
        "orange_1": [
            5, 12, 19,
            31, 43, 55,
            62, 69, 76,
            88, 100, 112,
            118,
            164, 161, 170, 176,
            183, 189, 195, 201,
            208, 226, 217, 223, 232,
            238, 244, 251, 257, 263, 270,
        ],
        "orange_2": [
            6, 13, 20,
            32, 44, 56,
            63, 70, 77,
            89, 101, 113,
            119,
            165, 162, 171, 177,
            184, 190, 196, 202,
            209, 227, 218, 224, 233,
            239, 245, 252, 258, 264, 271,
        ],
        "boost": list(range(120, 154)),  # 120..153
        "global": [
            178,  # delta
            203,  # is_overtime
            246,  # seconds_remaining
            265,  # time
        ],
    },
}

PLAYER_SLOTS = [
    "blue_0", "blue_1", "blue_2",
    "orange_0", "orange_1", "orange_2",
]

STATIC_POSITIONS = {
    "field_center_floor": (0.0, 0.0, 0.0),
    "field_center_ceiling": (0.0, 0.0, 2044.0),
    "field_left_wall_center": (4096.0, 0.0, 1022.0),
    "field_right_wall_center": (-4096.0, 0.0, 0.0),
    "field_blue_back_wall_center": (0.0, -5120.0, 1022.0),
    "field_orange_back_wall_center": (0.0, 5120.0, 1022.0),
    "goal_blue_bottom_center": (0.0, -5120.0, 0.0),
    "goal_blue_crossbar_center": (0.0, -5120.0, 642.775),
    "goal_blue_left_post_center": (892.775, -5120.0, 321.3875),
    "goal_blue_right_post_center": (-892.775, -5120.0, 321.3875),
    "goal_orange_bottom_center": (0.0, 5120.0, 0.0),
    "goal_orange_crossbar_center": (0.0, 5120.0, 642.775),
    "goal_orange_left_post_center": (892.775, 5120.0, 321.3875),
    "goal_orange_right_post_center": (-892.775, 5120.0, 321.3875),
}

def get_static_positions_np(position_scale: float = 8192.0, dtype=np.float32) -> np.ndarray:
    """
    Returns [14, 3] static positions scaled by position_scale (8192).
    """
    coords_uu = np.array(list(STATIC_POSITIONS.values()), dtype=dtype)  # [14,3]
    return coords_uu / dtype(position_scale)


MASKS_NP = {
    "duel": np.array([
        # 0–8: nodes
        False,  # 0 node_blue_player_0      (real)
        False,  # 1 node_orange_player_0    (real)
        True,   # 2 node_blue_player_1      (dummy)
        True,   # 3 node_orange_player_1    (dummy)
        True,   # 4 node_blue_player_2      (dummy)
        True,   # 5 node_orange_player_2    (dummy)
        False,  # 6 node_ball               (real)
        False,  # 7 node_boost_pads         (real)
        False,  # 8 node_global             (real)

        # 9–29: dyn↔dyn edges
        False,  #  9 edge_dd_0_1            (blue0–orange0, real)
        True,   # 10 edge_dd_0_2            (dummy)
        True,   # 11 edge_dd_0_3            (dummy)
        True,   # 12 edge_dd_0_4            (dummy)
        True,   # 13 edge_dd_0_5            (dummy)
        False,  # 14 edge_dd_0_6            (blue0–ball, real)
        True,   # 15 edge_dd_1_2            (dummy)
        True,   # 16 edge_dd_1_3            (dummy)
        True,   # 17 edge_dd_1_4            (dummy)
        True,   # 18 edge_dd_1_5            (dummy)
        False,  # 19 edge_dd_1_6            (orange0–ball, real)
        True,   # 20 edge_dd_2_3            (dummy)
        True,   # 21 edge_dd_2_4            (dummy)
        True,   # 22 edge_dd_2_5            (dummy)
        True,   # 23 edge_dd_2_6            (dummy)
        True,   # 24 edge_dd_3_4            (dummy)
        True,   # 25 edge_dd_3_5            (dummy)
        True,   # 26 edge_dd_3_6            (dummy)
        True,   # 27 edge_dd_4_5            (dummy)
        True,   # 28 edge_dd_4_6            (dummy)
        True,   # 29 edge_dd_5_6            (dummy)  # <- this was the missing one

        # 30–36: dyn↔static edges
        False,  # 30 edge_ds_0_static       (blue0, real)
        False,  # 31 edge_ds_1_static       (orange0, real)
        True,   # 32 edge_ds_2_static       (dummy)
        True,   # 33 edge_ds_3_static       (dummy)
        True,   # 34 edge_ds_4_static       (dummy)
        True,   # 35 edge_ds_5_static       (dummy)
        False,  # 36 edge_ds_6_static       (ball, real)
    ], dtype=bool),

    "doubles": np.array([
        # 0–8: nodes
        False,  # 0 node_blue_player_0      (real)
        False,  # 1 node_orange_player_0    (real)
        False,  # 2 node_blue_player_1      (real)
        False,  # 3 node_orange_player_1    (real)
        True,   # 4 node_blue_player_2      (dummy)
        True,   # 5 node_orange_player_2    (dummy)
        False,  # 6 node_ball               (real)
        False,  # 7 node_boost_pads         (real)
        False,  # 8 node_global             (real)

        # 9–29: dyn↔dyn edges
        False,  #  9 edge_dd_0_1            (real)
        False,  # 10 edge_dd_0_2            (real)
        False,  # 11 edge_dd_0_3            (real)
        True,   # 12 edge_dd_0_4            (dummy)
        True,   # 13 edge_dd_0_5            (dummy)
        False,  # 14 edge_dd_0_6            (real)
        False,  # 15 edge_dd_1_2            (real)
        False,  # 16 edge_dd_1_3            (real)
        True,   # 17 edge_dd_1_4            (dummy)
        True,   # 18 edge_dd_1_5            (dummy)
        False,  # 19 edge_dd_1_6            (real)
        False,  # 20 edge_dd_2_3            (real)
        True,   # 21 edge_dd_2_4            (dummy)
        True,   # 22 edge_dd_2_5            (dummy)
        False,  # 23 edge_dd_2_6            (real)
        True,   # 24 edge_dd_3_4            (dummy)
        True,   # 25 edge_dd_3_5            (dummy)
        False,  # 26 edge_dd_3_6            (real)
        True,   # 27 edge_dd_4_5            (dummy)
        True,   # 28 edge_dd_4_6            (dummy)
        True,   # 29 edge_dd_5_6            (dummy)

        # 30–36: dyn↔static edges
        False,  # 30 edge_ds_0_static       (blue0, real)
        False,  # 31 edge_ds_1_static       (orange0, real)
        False,  # 32 edge_ds_2_static       (blue1, real)
        False,  # 33 edge_ds_3_static       (orange1, real)
        True,   # 34 edge_ds_4_static       (blue2, dummy)
        True,   # 35 edge_ds_5_static       (orange2, dummy)
        False,  # 36 edge_ds_6_static       (ball, real)
    ], dtype=bool),

    "standard": np.zeros(37, dtype=bool),
}

# ---------------------------
# NumPy edge computations
# ---------------------------

def _wrap_delta_np(delta: np.ndarray, half_range: float) -> np.ndarray:
    period = 2.0 * half_range
    return (delta + half_range) % period - half_range


def _orientation_delta_scaled_np(rot_i: np.ndarray, rot_j: np.ndarray) -> np.ndarray:
    """
    rot_i, rot_j: [B,3] scaled rotations:
      rot_x in [-0.5, 0.5], rot_y, rot_z in [-1, 1]
    returns [B,3] wrapped deltas in same scaled units
    """
    d = rot_j - rot_i
    d_rx = _wrap_delta_np(d[:, 0], half_range=0.5)
    d_ry = _wrap_delta_np(d[:, 1], half_range=1.0)
    d_rz = _wrap_delta_np(d[:, 2], half_range=1.0)
    return np.stack([d_rx, d_ry, d_rz], axis=1)


def _compute_dyn_dyn_edge_np(node_i: np.ndarray, node_j: np.ndarray) -> np.ndarray:
    """
    node_i, node_j: [B, D_node] where first 12 dims are:
      0:3   ang_vel
      3:6   pos
      6:9   rot
      9:12  vel

    returns [B,18]
    """
    eps = 1e-6

    ang_i = node_i[:, 0:3]
    pos_i = node_i[:, 3:6]
    rot_i = node_i[:, 6:9]
    vel_i = node_i[:, 9:12]

    ang_j = node_j[:, 0:3]
    pos_j = node_j[:, 3:6]
    rot_j = node_j[:, 6:9]
    vel_j = node_j[:, 9:12]

    # positional delta + dist
    dpos = pos_j - pos_i                         # [B,3]
    dpos_dist = np.linalg.norm(dpos, axis=1, keepdims=True)  # [B,1]

    dx, dy, dz = dpos[:, 0], dpos[:, 1], dpos[:, 2]

    # angles
    theta_xy = np.arctan2(dy, dx + eps)
    theta_xz = np.arctan2(dz, dx + eps)
    theta_yz = np.arctan2(dz, dy + eps)
    pos_angles = np.stack([theta_xy, theta_xz, theta_yz], axis=1)  # [B,3]

    # velocity delta + dist
    dvel = vel_j - vel_i
    dvel_dist = np.linalg.norm(dvel, axis=1, keepdims=True)  # [B,1]

    # velocity dot
    dot_v = np.sum(vel_i * vel_j, axis=1, keepdims=True)  # [B,1]

    # angular velocity delta
    dang = ang_j - ang_i

    # orientation delta wrapped
    drot = _orientation_delta_scaled_np(rot_i, rot_j)

    edge = np.concatenate(
        [dpos, dpos_dist, pos_angles, dvel, dvel_dist, dot_v, dang, drot],
        axis=1
    ).astype(np.float32, copy=False)

    return edge


def _compute_dyn_static_edge_np(pos_dyn: np.ndarray, pos_static: np.ndarray) -> np.ndarray:
    """
    pos_dyn: [B,3]
    pos_static: [14,3]
    returns: [B,56] = 14*(dx,dy,dz,dist)
    """
    # [B,1,3] and [1,14,3]
    dpos = pos_static[None, :, :] - pos_dyn[:, None, :]  # [B,14,3]
    dist = np.linalg.norm(dpos, axis=2, keepdims=True)   # [B,14,1]
    edges = np.concatenate([dpos, dist], axis=2)         # [B,14,4]
    return edges.reshape(edges.shape[0], -1).astype(np.float32, copy=False)


# ---------------------------
# NumPy node/edge creation
# ---------------------------

def get_node_indices(gamemode: str, node: str, node_index_map: Dict[str, Dict[str, List[int]]]) -> List[int]:
    try:
        return node_index_map[gamemode][node]
    except KeyError:
        return None


def create_nodes_np(x: np.ndarray, gamemode: str, node_index_map: Dict[str, Dict[str, List[int]]]) -> List[np.ndarray]:
    """
    Returns nodes_list (length 9): players (padded), ball, boost, global
    """
    idx = node_index_map[gamemode]
    ball_node = x[:, idx["ball"]]
    boost_node = x[:, idx["boost"]]
    global_node = x[:, idx["global"]]

    if gamemode == "duel":
        blue0 = x[:, get_node_indices(gamemode, "blue_0", node_index_map)]
        orange0 = x[:, get_node_indices(gamemode, "orange_0", node_index_map)]

        dummy = np.zeros_like(blue0, dtype=np.float32)

        blue1 = dummy.copy()
        blue2 = dummy.copy()
        orange1 = dummy.copy()
        orange2 = dummy.copy()

    elif gamemode == "doubles":
        blue0 = x[:, get_node_indices(gamemode, "blue_0", node_index_map)]
        blue1 = x[:, get_node_indices(gamemode, "blue_1", node_index_map)]
        orange0 = x[:, get_node_indices(gamemode, "orange_0", node_index_map)]
        orange1 = x[:, get_node_indices(gamemode, "orange_1", node_index_map)]

        dummy = np.zeros_like(blue0, dtype=np.float32)
        blue2 = dummy.copy()
        orange2 = dummy.copy()

    elif gamemode == "standard":
        blue0 = x[:, get_node_indices(gamemode, "blue_0", node_index_map)]
        blue1 = x[:, get_node_indices(gamemode, "blue_1", node_index_map)]
        blue2 = x[:, get_node_indices(gamemode, "blue_2", node_index_map)]
        orange0 = x[:, get_node_indices(gamemode, "orange_0", node_index_map)]
        orange1 = x[:, get_node_indices(gamemode, "orange_1", node_index_map)]
        orange2 = x[:, get_node_indices(gamemode, "orange_2", node_index_map)]
    else:
        raise ValueError(f"Unknown gamemode: {gamemode}")

    nodes_list = [
        blue0,      # 0
        orange0,    # 1
        blue1,      # 2
        orange1,    # 3
        blue2,      # 4
        orange2,    # 5
        ball_node,  # 6
        boost_node, # 7
        global_node # 8
    ]

    return nodes_list


def create_edges_np(nodes_list: List[np.ndarray], static_positions: np.ndarray) -> List[np.ndarray]:
    """
    Returns edges_list length 28: 21 dyn-dyn + 7 dyn-static
    """
    dynamic_nodes = nodes_list[:7]  # players+ball
    dyn_dyn_edges: List[np.ndarray] = []

    # unordered pairs among 7 dynamic nodes
    for i in range(7):
        for j in range(i + 1, 7):
            dyn_dyn_edges.append(_compute_dyn_dyn_edge_np(dynamic_nodes[i], dynamic_nodes[j]))

    dyn_static_edges: List[np.ndarray] = []
    for dyn_node in dynamic_nodes:
        pos_dyn = dyn_node[:, 3:6]  # per your node layout
        dyn_static_edges.append(_compute_dyn_static_edge_np(pos_dyn, static_positions))

    return dyn_dyn_edges + dyn_static_edges


def xyz_to_graph(
    x: np.ndarray,
    gamemode: str,
    node_index_map=None,
    position_scale: float = 8192.0
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Pure NumPy version of xyz_to_graph.

    Args:
      x: [B, x_dim] float32/float64. Will be converted to float32.
      gamemode: "duel" | "doubles" | "standard"
      node_index_map: your NODE_INDEX_MAP dict
      position_scale: used only for static positions

    Returns:
      tokens: list length 37 of np arrays
      token_mask: [37] bool array
    """
    if node_index_map is None:
        node_index_map = NODE_INDEX_MAP
    if gamemode not in node_index_map:
        raise ValueError(f"Unknown gamemode: {gamemode}")

    x = np.asarray(x, dtype=np.float32)
    static_positions = get_static_positions_np(position_scale=position_scale, dtype=np.float32)

    nodes = create_nodes_np(x, gamemode, node_index_map)
    edges = create_edges_np(nodes, static_positions)

    tokens = nodes + edges
    if len(tokens) != 37:
        raise RuntimeError(f"Expected 37 tokens, got {len(tokens)}")

    token_mask = MASKS_NP[gamemode].copy()
    return tokens, token_mask
