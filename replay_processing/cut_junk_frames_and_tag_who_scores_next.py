import pandas as pd
import numpy as np
from .gettimerframes import get_starting_gun_frames
from .goalfunctions import get_goal_frames, get_goal_tag_sequence
from carball.json_parser.game import Game
from scipy.ndimage.interpolation import shift
import warnings


def cut_junk_frames_and_tag_who_scores_next(game: Game):

    '''


    :param game: a carball game class

    modifies game.frames by cutting out frames between play stoppages (goals and ball-hitting-the-floor timeout)
    creates two new columns in game.frames
    who_scores next places a label for each frame: 0 (blue), 1 (orange), or 0.5 (noone scored, timeout/ff)
    frames_until_stoppage marks how many frames there are until the next play stoppage

    '''
    frames = game.frames

    startingframes = np.array(get_starting_gun_frames(game))
    goalframes = np.array(get_goal_frames(game))
    # make sure all goalframes are recorded in game.frames and game.ball
    for i in range(len(goalframes)):
        if goalframes[i] not in game.frames.index:
            warnings.warn("Found goal not in indices")
            game = 0
            return game
        if goalframes[i] not in game.ball.index:
            warnings.warn("Found goal not in indices")
            game = 0
            return game
        for player in game.players:
            if goalframes[i] not in player.data.index:
                warnings.warn("Found goal not in indices")
                game = 0
                return game

    goaltags = np.array(get_goal_tag_sequence(game))

    # combine these two arrays into one
    cutframes = np.concatenate((startingframes, goalframes))

    # in case of OT, there won't be a goal frame between the OT starting frame and the previous starting frame
    if any(game.frames.is_overtime):
        isot = game.frames.is_overtime.fillna(False)
        otframe = np.argmax(isot)  # finds the integer index number of the first True (e.g. 11302)
        otframe = game.frames.index[otframe]  # find the labelled index number of the first True (e.g. 11313)
        cutframes = np.append(cutframes, otframe)  # add the overtime frame into the cutframes array

    cutframes = np.append(cutframes, game.frames.index[0])  # add the first index label into the cutframes array
    cutframes = np.sort(cutframes)  # sort everything before processing cutframes

    # create a 2D array of indices to slice on
    cutstarts = cutframes
    cutends = shift(cutframes, -1)  # shift cutframes up an an index to get the cut ends
    cutends[-1] = np.array(frames.index[-1])  # make the last cut end index the last index of game.frames
    cutindices = np.vstack((cutstarts, cutends))
    cutindices = cutindices.T  # transpose the array in order to iterate over it

    trimmedframes = pd.DataFrame()
    x = 0  # use this to tag index labels, 0 for discard, 1 for keep
    for indices in cutindices:
        keepstart = (indices[0])
        keepend = (indices[1])
        framessubset = frames.loc[keepstart:keepend]
        if x == 1:
            trimmedframes = trimmedframes.append(framessubset)
            # find out the goal indicator, 1, 0.5, 0 and put it in a column here

            if keepend in goalframes:
                goalindex = np.where(goalframes == keepend)  # find the index of the goal in goalframes
                goaltag = goaltags[goalindex]  # goaltags corresponds to goalframes, this index will give the proper tag

            elif keepend not in goalframes:
                goaltag = 0.5  # e.g. at the end of an OT game or ff when no one scores
            # fill who_scores_next in kept interval with goaltag
            trimmedframes.loc[keepstart:keepend, "who_scores_next"] = goaltag

            # in order to count backwards, make a range from len[keepstart:keepend] up to, not incl -1, step down by -1
            framesuntilstoppage = range((len(frames.loc[keepstart:keepend]) - 1), -1, -1)
            trimmedframes.loc[keepstart:keepend, "frames_until_stoppage"] = framesuntilstoppage


            x = 0
        else:
            x = 1

    game.frames = trimmedframes
    # subset game.players[].data with the same indices as trimmedframes
    trimmedframesindices = trimmedframes.index
    for player in game.players:
        player.data = player.data.reindex(index=trimmedframesindices)
    # also subset game.ball by these index labels
    game.ball = game.ball.reindex(index=trimmedframesindices)
    return game
