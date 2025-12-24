from carball.json_parser.game import Game
from rlutilities.simulation import Ball
from rlutilities.simulation import Game as Gamerlu
from rlutilities.linear_algebra import vec3

import numpy as np
import pandas as pd


def vec3_to_list(v3: vec3):
    l1 = v3[0]
    l2 = v3[1]
    l3 = v3[2]
    v3list = [l1, l2, l3]
    return v3list


def create_futureball(game: Game):
    Gamerlu.set_mode("soccar")
    ball = Ball()
    game.ball = game.ball.fillna(0)

    futureball = []
    for row in game.ball.values:
        p = vec3(row[0], row[1], row[2])
        v = vec3(row[3], row[4], row[5])
        a = vec3(row[6], row[7], row[8])

        ball.position = p
        ball.velocity = v
        ball.angular_velocity = a

        for _ in range(12):  # simulate one tenth second into the future. predict 0.1, 0.5, 1, 3, 5 seconds from t0
            ball.step(1/120)
        v1 = vec3_to_list(ball.velocity)
        p1 = vec3_to_list(ball.position)
        for _ in range(48):
            ball.step(1/120)
        v2 = vec3_to_list(ball.velocity)
        p2 = vec3_to_list(ball.position)
        for _ in range(60):
            ball.step(1/120)
        v3 = vec3_to_list(ball.velocity)
        p3 = vec3_to_list(ball.position)
        for _ in range(240):
            ball.step(1/120)
        v4 = vec3_to_list(ball.velocity)
        p4 = vec3_to_list(ball.position)
        for _ in range(240):
            ball.step(1/120)
        v5 = vec3_to_list(ball.velocity)
        p5 = vec3_to_list(ball.position)

        futureball_row = v1 + p1 + v2 + p2 + v3 + p3 + v4 + p4 + v5 + p5
        futureball.append(futureball_row)

    futureball = pd.DataFrame(np.array(futureball), index=game.ball.index)
    futureball.columns = ['vel_x_futureball_1', 'vel_y_futureball_1', 'vel_z_futureball_1',
                          'pos_x_futureball_1', 'pos_y_futureball_1', 'pos_z_futureball_1',
                          'vel_x_futureball_2', 'vel_y_futureball_2', 'vel_z_futureball_2',
                          'pos_x_futureball_2', 'pos_y_futureball_2', 'pos_z_futureball_2',
                          'vel_x_futureball_3', 'vel_y_futureball_3', 'vel_z_futureball_3',
                          'pos_x_futureball_3', 'pos_y_futureball_3', 'pos_z_futureball_3',
                          'vel_x_futureball_4', 'vel_y_futureball_4', 'vel_z_futureball_4',
                          'pos_x_futureball_4', 'pos_y_futureball_4', 'pos_z_futureball_4',
                          'vel_x_futureball_5', 'vel_y_futureball_5', 'vel_z_futureball_5',
                          'pos_x_futureball_5', 'pos_y_futureball_5', 'pos_z_futureball_5',
                          ]
    game.futureball = futureball

    return game