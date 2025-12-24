from carball.json_parser.game import Game
import fcl
import numpy as np
import pandas as pd
import os
from replay_processing.get_base_dir import get_base_dir


def test_wheel_collision(wheelindex, frame, t, wheels, wheeltris, arena_mesh, otherhitboxes, num_others, hitboxtris,
                         ball_c, balldata, request, result):
    wheelverts = np.array(np.split(wheels[wheelindex][frame], 17))
    wheel = fcl.BVHModel()
    wheel.beginModel(len(wheelverts), len(wheeltris))
    wheel.addSubModel(wheelverts, wheeltris)
    wheel.endModel()
    wheel_c = fcl.CollisionObject(wheel, t)
    wheel_arena_c = fcl.collide(wheel_c, arena_mesh, request, result)
    if wheel_arena_c == 1:
        del wheel_arena_c, wheel, wheelverts
        return 1

    ballt = fcl.Transform(balldata[frame, [0, 1, 2]])
    ball_c.setTransform(ballt)
    wheel_ball_c = fcl.collide(wheel_c, ball_c, request, result)
    if wheel_ball_c == 1:
        del wheel_arena_c, wheel, wheelverts
        del wheel_ball_c, ballt
        return 1

    hitboxverts = np.array(np.split(otherhitboxes[frame], 8 * num_others))
    hitboxes = fcl.BVHModel()
    hitboxes.beginModel(len(hitboxverts), len(hitboxtris))
    hitboxes.addSubModel(hitboxverts, hitboxtris)
    hitboxes.endModel()
    hitboxes_c = fcl.CollisionObject(hitboxes, t)
    wheel_hitboxes_c = fcl.collide(wheel_c, hitboxes_c, request, result)
    if wheel_hitboxes_c == 1:
        del wheel_arena_c, wheel, wheelverts
        del wheel_ball_c, ballt
        del hitboxverts, hitboxes, hitboxes_c, wheel_hitboxes_c
        return 1
    else:
        del wheel_arena_c, wheel, wheelverts
        del wheel_ball_c, ballt
        del hitboxverts, hitboxes, hitboxes_c, wheel_hitboxes_c
        return 0


def detect_flip_resets(game: Game):
    print(os.getcwd())
    vertspath = os.path.join(get_base_dir(), "replay_processing", "fieldverts.pk1")
    fieldverts = np.array(pd.read_pickle(vertspath))
    trispath = os.path.join(get_base_dir(), "replay_processing", "fieldtris.pk1")
    fieldtris = np.array(pd.read_pickle(trispath))
    fieldmesh = fcl.BVHModel()
    fieldmesh.beginModel(len(fieldverts), len(fieldtris))
    fieldmesh.addSubModel(fieldverts, fieldtris)
    fieldmesh.endModel()
    t = fcl.Transform()
    arena_c = fcl.CollisionObject(fieldmesh, t)

    wheeltris = np.array(
        [[1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0], [5, 6, 0], [6, 7, 0], [7, 8, 0], [8, 9, 0], [9, 10, 0],
         [10, 11, 0], [11, 12, 0], [12, 13, 0], [13, 14, 0], [14, 15, 0], [15, 16, 0], [16, 1, 0]])
    hitboxtris = np.array(
        [[0, 1, 4], [1, 5, 4], [2, 6, 7], [2, 3, 7], [0, 1, 3], [2, 3, 1], [5, 6, 4], [6, 7, 4], [0, 3, 4],
         [7, 3, 4], [6, 5, 1], [1, 2, 6]])

    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()

    # stacking the hitboxes of the others into one CollisionObject, so I need to create triangles for that
    for _ in range(len(game.players) - 2):
        hitboxtris = np.vstack((hitboxtris, (hitboxtris + 8)))

    balldata = game.ball.values
    balldata = np.nan_to_num(balldata)
    ball = fcl.Sphere(92.75)
    ball_c = fcl.CollisionObject(ball, t)
    num_others = len(game.players) - 1

    i = 0
    for player in game.players:
        wheels = [np.concatenate(wheelverts, axis=1) for wheelverts in player.wheelvertices]

        otherhitboxes = [game.players[n].vertexcoords for n in range(len(game.players))]
        del otherhitboxes[i]  # don't consider the player's own hitbox for wheel collision detection
        otherhitboxes = [np.concatenate(hitboxverts, axis=1) for hitboxverts in otherhitboxes]
        otherhitboxes = np.concatenate(otherhitboxes, axis=1)  # this is all hitboxes of others combined into one array

        flip_resets = []
        # FLwheel_c = []
        # FRwheel_c = []
        # BRwheel_c = []
        # BLwheel_c = []
        for frame in range(len(player.data)):
            flip_reset = 1
            # FL = 0
            # FR = 0
            # BR = 0
            # BL = 0
            for wheelindex in [0, 1, 2, 3]:
                wheel_collision = test_wheel_collision(wheelindex, frame, t, wheels, wheeltris, arena_c, otherhitboxes,
                                                       num_others,
                                                       hitboxtris, ball_c, balldata, request, result)
                if wheel_collision == 0:
                    flip_reset = 0
                    break
                # if wheelindex == 0:
                #     FL = 1
                # if wheelindex == 1:
                #     FR = 1
                # if wheelindex == 2:
                #     BR = 1
                # if wheelindex == 3:
                #     BL = 1

            flip_resets.append(flip_reset)
            # FLwheel_c.append(FL)
            # FRwheel_c.append(FR)
            # BRwheel_c.append(BR)
            # BLwheel_c.append(BL)

        df = pd.DataFrame()
        player.jumps = df
        player.jumps['grounded'] = flip_resets
        # game.players[i].jumps['FL'] = FLwheel_c
        # game.players[i].jumps['FR'] = FRwheel_c
        # game.players[i].jumps['BR'] = BRwheel_c
        # game.players[i].jumps['BL'] = BLwheel_c
        game.players[i].jumps.index = game.frames.index
        i += 1

    del t, arena_c, request, result, fieldmesh, ball, ball_c
    return game
