from scipy.spatial.transform import Rotation
from carball.json_parser.game import Game
import numpy as np
import pandas as pd
from .hitbox import *
import trimesh




def get_hitbox_vertices(game:Game):
    # in order, front left top, front right top, front right bottom, front left bottom, back left top, back right top,
    # back right bottom, back left bottom
    for player in game.players:
        df = player.data
        car = player.loadout[0]['car']
        hitbox = get_hitbox(car)
        vertex_displacements = get_vertex_displacements(hitbox)

        posxyz = df[['pos_x', 'pos_y', 'pos_z']]

        # in player.data, rot_x is pitch, rot_y is yaw, and rot_z is roll
        # yxz with ZYX from_euler appears to work...!
        rot = df[['rot_y', 'rot_x', 'rot_z']]
        rot = Rotation.from_euler('ZYX', rot)

        vertexcoords = []
        for vertex in vertex_displacements:
            rot_v_displacement_coords = rot.apply(vertex)
            v_coords = posxyz + rot_v_displacement_coords
            v_coords = np.nan_to_num(v_coords, nan=-1000)
            vertexcoords.append(v_coords)


        player.vertexcoords = vertexcoords

    return game

def circle_verts(radius, n):
    theta = np.linspace(0, 2 * np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

def get_wheel_vertices(game: Game):
    # in order, front left, front right, back right, back left, coords xyz relative to rj
    for player in game.players:
        df = player.data
        car = player.loadout[0]['car']
        hitbox = get_hitbox(car)
        wheel_displacements = get_wheel_displacements(hitbox)
        radii = get_wheel_radii(hitbox)
        front_wheel_radius = radii[0]
        back_wheel_radius = radii[1]

        frontx, frontz = circle_verts(front_wheel_radius, 16) # the wheels are along the xz plane
        backx, backz = circle_verts(back_wheel_radius, 16)

        frontxyz = np.array([frontx, np.zeros_like(frontx), frontz]).T
        backxyz = np.array([backx, np.zeros_like(backx), backz]).T

        frontxyzL = frontxyz + wheel_displacements[0]
        frontxyzR = frontxyz + wheel_displacements[1]
        backxyzR = backxyz + wheel_displacements[2]
        backxyzL = backxyz + wheel_displacements[3]

        wheelverts = [frontxyzL, frontxyzR, backxyzR, backxyzL]

        posxyz = df[['pos_x', 'pos_y', 'pos_z']]

        # in player.data, rot_x is pitch, rot_y is yaw, and rot_z is roll
        # yxz with ZYX from_euler appears to work...!
        rot = df[['rot_y', 'rot_x', 'rot_z']]
        rot = Rotation.from_euler('ZYX', rot)

        wheel_displacements_rotated = [rot.apply(displacement) for displacement in wheel_displacements]
        wheelcenters = [posxyz + displacement for displacement in wheel_displacements_rotated]

        player.wheelvertices = [[],[],[],[]]
        for i in range(4):
            wheel = wheelverts[i] # loop over wheels
            center = wheelcenters[i]
            center = np.nan_to_num(center, nan=-1000)
            player.wheelvertices[i].append(center)
            for vert in wheel: # loop over each vert array in wheel
                wheel_vert_rotated = rot.apply(vert) # apply the rotation, relative to rj
                wheel_vert_xyz = wheel_vert_rotated + posxyz # add rj's translastion i.e. posxyz
                wheel_vert_xyz = np.nan_to_num(wheel_vert_xyz, nan=-1000)
                player.wheelvertices[i].append(wheel_vert_xyz)

    return game

def get_ball_vertices(game: Game):
    sphere = trimesh.creation.icosphere(radius=92.75)
    baseverts = sphere.vertices

    df = game.ball
    posxyz = df[['pos_x', 'pos_y', 'pos_z']]
    vertlist = []
    for row in posxyz:
        verts = baseverts + row
        vertlist.append(verts)

    game.ballverts = vertlist



