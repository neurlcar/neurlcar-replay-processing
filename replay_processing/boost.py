import time
import numpy as np
import pandas as pd
import scipy.spatial.distance
from carball.json_parser.game import Game
from .goalfunctions import get_goal_frames




def get_time(starttime):
    currenttime = time.perf_counter()
    elapsed = currenttime - starttime
    return elapsed


def get_boost_collect_tags(game: Game):
    # coordinates of all the boosts (x, y, label)
    allboosts = np.array([
        (0.0, -4240.0, 1),
        (-1792.0, -4184.0, 2),
        (1792.0, -4184.0, 3),
        (-3072, -4096, 4),
        (3072, -4096, 5),
        (-940.0, -3308.0, 6),
        (940.0, -3308.0, 7),
        (0.0, -2816.0, 8),
        (-3584.0, -2484.0, 9),
        (3584.0, -2484.0, 10),
        (-1788.0, -2300.0, 11),
        (1788.0, -2300.0, 12),
        (-2048.0, -1036.0, 13),
        (0.0, -1024.0, 14),
        (2048.0, -1036.0, 15),
        (-3584, 0, 16),
        (-1024.0, 0.0, 17),
        (1024.0, 0.0, 18),
        (3584, 0, 19),
        (-2048.0, 1036.0, 20),
        (0.0, 1024.0, 21),
        (2048.0, 1036.0, 22),
        (-1788.0, 2300.0, 23),
        (1788.0, 2300.0, 24),
        (-3584.0, 2484.0, 25),
        (3584.0, 2484.0, 26),
        (0.0, 2816.0, 27),
        (-940.0, 3310.0, 28),
        (940.0, 3308.0, 29),
        (3072, 4096, 30),
        (-3072, 4096, 31),
        (-1792.0, 4184.0, 32),
        (1792.0, 4184.0, 33),
        (0.0, 4240.0, 34)])
    # bigboosts = np.array([
    #    (3072, -4096, 5),
    #    (-3072, -4096, 4),
    #    (3584, 0, 19),
    #    (-3584, 0, 16),
    #    (3072, 4096, 30),
    #    (-3072, 4096, 31)])
    #bigboosts_labels = (5, 4, 16, 19, 30, 31)

    for player in game.players:
        # get boost deltas
        player.data["boost_delta"] = player.data.boost.diff().shift(-1)
        ''' 
        add boost_collected boolean and add it as another column  there's a bug where the player boost amount gets
        corrected upwards. small boosts add ~30 boosts, but sometimes they add numbers in the low 20s so any value above
        15 is a "real" boost pickup. Any value that's positive that pushes the player to 255 can be considered a "real"
        boost pickup as well.
        '''
        player.data["boost_collected"] = player.data["boost_collected"] = \
            (player.data.boost_delta >= 15) | ((player.data['boost_delta'] > 0) & (player.data['boost'] == 255))

        # get indices where boost_collected is True
        boostcollectindices = np.array(np.where(player.data.boost_collected))  # index numbers
        boostcollectindices = np.array(player.data.index[boostcollectindices])  # index labels
        boostcollectindices = np.array(np.reshape(boostcollectindices, -1))  # reduce dimensionality from 2 to 1
        # get pos_x and pos_y of boostcollectframeindices
        boostcollectcoordinates = player.data.loc[boostcollectindices, ("pos_x", "pos_y")]
        # get a 34 x len(boostcollectindices) array of distances to boosts upon boost_collect
        distances = scipy.spatial.distance.cdist(boostcollectcoordinates, allboosts[:, 0:2])
        tagofclosestboost = allboosts[distances.argmin(axis=1), 2]
        player.data.loc[boostcollectindices, "boostcollect_tag"] = tagofclosestboost.astype(int)

    return game


def append_boost_timer_columns(game: Game):

    allboosts = np.array([
        (0.0, -4240.0, 1),
        (-1792.0, -4184.0, 2),
        (1792.0, -4184.0, 3),
        (-3072, -4096, 4),
        (3072, -4096, 5),
        (-940.0, -3308.0, 6),
        (940.0, -3308.0, 7),
        (0.0, -2816.0, 8),
        (-3584.0, -2484.0, 9),
        (3584.0, -2484.0, 10),
        (-1788.0, -2300.0, 11),
        (1788.0, -2300.0, 12),
        (-2048.0, -1036.0, 13),
        (0.0, -1024.0, 14),
        (2048.0, -1036.0, 15),
        (-3584, 0, 16),
        (-1024.0, 0.0, 17),
        (1024.0, 0.0, 18),
        (3584, 0, 19),
        (-2048.0, 1036.0, 20),
        (0.0, 1024.0, 21),
        (2048.0, 1036.0, 22),
        (-1788.0, 2300.0, 23),
        (1788.0, 2300.0, 24),
        (-3584.0, 2484.0, 25),
        (3584.0, 2484.0, 26),
        (0.0, 2816.0, 27),
        (-940.0, 3310.0, 28),
        (940.0, 3308.0, 29),
        (3072, 4096, 30),
        (-3072, 4096, 31),
        (-1792.0, 4184.0, 32),
        (1792.0, 4184.0, 33),
        (0.0, 4240.0, 34)])

    frames = game.frames
    bigboosts_labels = (4, 3, 15, 18, 29, 30)  # shifted down one from the array

    allboostcollecttags = []
    allboostcollectindices = []
    for player in game.players:
        boostcollectindices = np.array(np.where(player.data.boost_collected))  # index numbers
        boostcollectindices = np.array(player.data.index[boostcollectindices])  # index labels
        boostcollectindices = np.array(np.reshape(boostcollectindices, -1))  # reduce dimensionality from 2 to 1
        boostcollecttags = player.data.loc[boostcollectindices]["boostcollect_tag"]
        # combine tags into a single array and indices into a single array
        allboostcollectindices = np.append(allboostcollectindices, boostcollectindices)
        allboostcollecttags = np.append(allboostcollecttags, boostcollecttags)

    allboostcollectcoords = np.vstack((allboostcollectindices, allboostcollecttags)).T.astype(int)

    goalframes = get_goal_frames(game)
    goalframes.append(frames.index[0])
    if any(game.frames.is_overtime):
        isot = game.frames.is_overtime.fillna(False)
        otframe = np.argmax(isot)  # finds the integer index number of the first True (e.g. 11302)
        otframe = game.frames.index[otframe]  # find the labelled index number of the first True (e.g. 11313)
        goalframes.append(otframe)

    stoppageframes = goalframes


    boosttimerdf = np.full((len(frames), len(allboosts)), None)  # create empty df width num of boosts, len(frames)
    boosttimerdf = pd.DataFrame(boosttimerdf)
    boosttimerdf.index = frames.index  # reindex with labelled indices from frames
    boosttimerdf = boosttimerdf.fillna(0)

    boosttimerdfempty = pd.DataFrame(np.zeros_like(boosttimerdf))  # will need an empty version of this df to work with
    boosttimerdfempty.index = frames.index  # reindex with labelled indices from frames

    a = pd.DataFrame(allboostcollectcoords)

    a[1] = a[1] - 1  # tag = tag - 1 so that the tag number matches the column number in boosttimerdf

    #  create a boolean boostpickupmask with indexes and tags on the 34-row by len(game.frames) df
    boostpickupmask = pd.DataFrame(np.zeros_like(boosttimerdf, dtype=bool))
    boostpickupmask.index = frames.index  # reindex with labelled indices from frames
    for index, tag in np.array(a[[0, 1]]):
        if index in stoppageframes:
            continue
        boostpickupmask.at[index, tag] = True

    for i in range(34):  # make every column of boosttimerdf frames.delta
        boosttimerdf[i] = frames.delta

    #  add a virtual time of 10s to the boost timers every time there is a stoppage
    for frame in stoppageframes:
        boosttimerdf.loc[frame] = 10.0
    boosttimerdfmasked = boosttimerdf.mask(boostpickupmask, None)

    btdfcumsum = boosttimerdfmasked.cumsum(axis=0).fillna(method='pad')

    resetter = -btdfcumsum[boosttimerdfmasked.isnull()]
    # get diff between the values of the negative cumsum mask so that only cumsum interval since pickup is counted
    for i in range(34):
        resetdiffs = resetter[i][resetter[i].notnull()].diff().fillna(resetter[i])
        resetter[i].loc[boostpickupmask[i]] = resetdiffs
    finalbtdf = boosttimerdfmasked.where(boosttimerdfmasked.notnull(), resetter).cumsum(axis=0)

    finalbtdf.loc[:, finalbtdf.columns.isin(bigboosts_labels)] = -(
            (finalbtdf.loc[:, finalbtdf.columns.isin(bigboosts_labels)]).clip(upper=10.0) - 10.0)

    finalbtdf.loc[:, ~finalbtdf.columns.isin(bigboosts_labels)] = -(
                (finalbtdf.loc[:, ~finalbtdf.columns.isin(bigboosts_labels)]).clip(upper=4.0) - 4.0)

    finalbtdf = finalbtdf + 0.0  # remove floating point negative zeros i.e. change -0.0 to 0.0

    # name columns
    btdfcolnames = []
    for i in range(len(finalbtdf.columns)):
        colname = "boost_" + str(i) +"_timer"
        btdfcolnames = np.append(btdfcolnames, colname)
    finalbtdf.columns = btdfcolnames

    game.frames = pd.concat([game.frames, finalbtdf], axis=1)
    return game




