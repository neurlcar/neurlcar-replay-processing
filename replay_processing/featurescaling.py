from math import pi


def scale_features(xy, gamemode):
    # the goal of this scaling is to bring all values to between -1 and 1
    if gamemode == 'duel':
        # see ./duelfeaturescaling.txt for more feature scaling notes
        # see https://github.com/RLBot/RLBot/wiki/Useful-Game-Values for the value limits

        xy[:, 0:9] = xy[:, 0:9] / 6000  # angular velocities
        xy[:, 11:45] = xy[:, 11:45] / 10  # boost timers
        xy[:, 47:49] = xy[:, 47:49] / 255  # boost amounts
        xy[:, 49:51] = xy[:, 49:51] / 34  # boostcollect tags
        xy[:, 54:56] = xy[:, 54:56] / 4  # demo timers (timers go a bit past the 3s respawn time)
        xy[:, 60:62] = xy[:, 60:62] / 6  # hitbox numbers (0 through 5)
        xy[:, 63:65] = xy[:, 63:65] / 10  # jump 2 expiration timers
        xy[:, 65:67] = xy[:, 65:67] / 2  # jump classifications
        xy[:, 75:78] = xy[:, 75:78] / 8192  # pos_x players and ball
        xy[:, 78:81] = xy[:, 78:81] / 6000  # pos_y players and ball
        xy[:, 81:84] = xy[:, 81:84] / 2000  # pos_z players and ball
        xy[:, 86:95] = xy[:, 86:95] / pi  # rotations x, y, and z of players and ball
        xy[:, 95] = xy[:, 95] / 300  # seconds_remaining
        xy[:, 102] = xy[:, 102] / 1000  # time (time elapsed since replay start)
        xy[:, [103, 106, 109]] = xy[:, [103, 106, 109]] / 60000  # velocities x, y, and z of ball
        # velocities x, y, and z of players
        xy[:, [104, 105, 107, 108, 110, 111]] = xy[:, [104, 105, 107, 108, 110, 111]] / 23000

    else:
        raise(Exception("Enter a valid game mode. 'duel', 'doubles', or 'standard'"))

    return xy

def scale_features_xyz_old(xyz, gamemode):  #  without ball prediction
    if gamemode == 'duel':
        xyz[:, 0:9] = xyz[:, 0:9] / 6000  # angular velocities
        xyz[:, 9:12] = xyz[:, 9:12] / 8192  # pos_x players and ball
        xyz[:, 12:15] = xyz[:, 12:15] / 6000  # pos_y players and ball
        xyz[:, 15:18] = xyz[:, 15:18] / 2100  # pos_z players and ball
        xyz[:, 18:27] = xyz[:, 18:27] / pi  # rotations x, y, and z of players and ball
        # velocities x, y, and z of players
        xyz[:, [28, 29, 31, 32, 34, 35]] = xyz[:, [28, 29, 31, 32, 34, 35]] / 23000
        xyz[:, [27, 30, 33]] = xyz[:, [27, 30, 33]] / 60000  # velocities x, y, and z of ball
        xyz[:, 38:72] = xyz[:, 38:72] / 10  # boost timers
        xyz[:, [72, 75]] = xyz[:, [72, 75]] / 255  # boost amounts
        xyz[:, 76:78] = xyz[:, 76:78] / 34  # boostcollect tags
        xyz[:, 81:83] = xyz[:, 81:83] / 4  # demo timers (timers go a bit past the 3s respawn time)
        xyz[:, 87:89] = xyz[:, 87:89] / 6  # hitbox numbers (0 through 5)
        xyz[:, 90:92] = xyz[:, 90:92] / 10  # jump 2 expiration timers
        xyz[:, 93:95] = xyz[:, 93:95] / 2  # jump classifications
        xyz[:, 104] = xyz[:, 104] / 300  # seconds_remaining
        xyz[:, 111] = xyz[:, 111] / 1000  # time (time elapsed since replay start)
    else:
        raise(Exception("Enter a valid game mode. 'duel', 'doubles', or 'standard'"))
    return xyz


def scale_features_xyz(xyz, gamemode):
    if gamemode == 'standard':
        xyz[:, 0:21] = xyz[:, 0:21] / 6000  # angular velocities
        xyz[:, 21:57] = xyz[:, 21:57] / 8192  # positions. scale by max field dim
        xyz[:, 57:78] = xyz[:, 57:78] / pi  # rotations x, y, and z of players and ball
        xyz[:, 78:114] = xyz[:, 78:114] / 60000  # velocities players and ball. scale by ball speed limit
        xyz[:, 120:154] = xyz[:, 120:154] / 10  # boost timers
        xyz[:, [154, 155, 156, 163, 164, 165]] = xyz[:, [154, 155, 156, 163, 164, 165]] / 255  # boost amounts
        xyz[:, 166:172] = xyz[:, 166:172] / 34  # boostcollect tags
        xyz[:, 179:185] = xyz[:, 179:185] / 4  # demo timers (timers go a bit past the 3s respawn time)
        xyz[:, 197:203] = xyz[:, 197:203] / 6  # hitbox numbers (0 through 5)
        xyz[:, 204:210] = xyz[:, 204:210] / 10  # jump 2 expiration timers
        xyz[:, 213:219] = xyz[:, 213:219] / 2  # jump classifications
        xyz[:, 246] = xyz[:, 246] / 300  # seconds_remaining
        xyz[:, 265] = xyz[:, 265] / 1000  # time (time elapsed since replay start)

    elif gamemode == 'doubles':         
        xyz[:, 0:15] = xyz[:, 0:15] / 6000  # angular velocities
        xyz[:, 15:45] = xyz[:, 15:45] / 8192  # positions. scale by max field dim
        xyz[:, 45:60] = xyz[:, 45:60] / pi  # rotations x, y, and z of players and ball
        xyz[:, 60:90] = xyz[:, 60:90] / 60000  # velocities players and ball. scale by ball speed limit
        xyz[:, 94:128] = xyz[:, 94:128] / 10  # boost timers
        xyz[:, [128, 129, 134, 135]] = xyz[:, [128, 129, 134, 135]] / 255  # boost amounts
        xyz[:, 136:140] = xyz[:, 136:140] / 34  # boostcollect tags
        xyz[:, 145:149] = xyz[:, 145:149] / 4  # demo timers (timers go a bit past the 3s respawn time)
        xyz[:, 157:161] = xyz[:, 157:161] / 6  # hitbox numbers (0 through 5)
        xyz[:, 162:166] = xyz[:, 162:166] / 10  # jump 2 expiration timers
        xyz[:, 168:172] = xyz[:, 168:172] / 2  # jump classifications
        xyz[:, 190] = xyz[:, 190] / 300  # seconds_remaining
        xyz[:, 203] = xyz[:, 203] / 1000  # time (time elapsed since replay start)

    elif gamemode == 'duel':
        xyz[:, 0:9] = xyz[:, 0:9] / 6000  # angular velocities
        xyz[:, 9:33] = xyz[:, 9:33] / 8192  # positions. scale by max field dim
        xyz[:, 33:42] = xyz[:, 33:42] / pi  # rotations x, y, and z of players and ball
        xyz[:, 42:66] = xyz[:, 42:66] / 60000  # velocities players and ball. scale by ball speed limit
        xyz[:, 68:102] = xyz[:, 68:102] / 10  # boost timers
        xyz[:, [102, 105]] = xyz[:, [102, 105]] / 255  # boost amounts
        xyz[:, 106:108] = xyz[:, 106:108] / 34  # boostcollect tags
        xyz[:, 111:113] = xyz[:, 111:113] / 4  # demo timers (timers go a bit past the 3s respawn time)
        xyz[:, 117:119] = xyz[:, 117:119] / 6  # hitbox numbers (0 through 5)
        xyz[:, 120:122] = xyz[:, 120:122] / 10  # jump 2 expiration timers
        xyz[:, 123:125] = xyz[:, 123:125] / 2  # jump classifications
        xyz[:, 134] = xyz[:, 134] / 300  # seconds_remaining
        xyz[:, 141] = xyz[:, 141] / 1000  # time (time elapsed since replay start)
    else:
        raise(Exception("Enter a valid game mode. 'duel', 'doubles', or 'standard'"))

    return xyz







