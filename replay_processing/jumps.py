from carball.json_parser.game import Game
import numpy as np
import pandas as pd


def build_jump_df(game: Game):
    frameslen = len(game.frames)
    for player in game.players:
        jumpbool = player.controls['jump']
        data = player.jumps
        data['jump_classification'] = 0
        data['jump_held_timer'] = 0
        data['jump_2_expiration_timer'] = 0
        data['jumps_exhausted'] = 0
        jbshiftdown = jumpbool.shift(periods=1, fill_value=False)
        jbshiftup = jumpbool.shift(periods=-1, fill_value=False)

        jumpstarts = data.loc[jumpbool & ~jbshiftdown].index.to_list()
        jumpends = data.loc[jumpbool & ~jbshiftup].index.to_list()

        jump1ends = []
        jump1ends_incr = []
        jump2starts = []
        jump2ends = []

        # the first jump is always grounded, the second jump is never grounded.
        # to classify between first and second jumps, test if the car is grounded.
        # sometimes, the initial frame of the first jump is detected the frame after ungrounding.
        # because of this, it is necessary to test if the frame before the jumpstart frame is grounded.
        # however, sometimes the second jump happens very quickly,
        # and the frame before the second jump is also grounded.
        # Since misclassifying a second jump as a first jump only happens when the jumps are close together,
        # a solution is to only test the previous frame if the jumps are sufficiently far apart

        pje = data.index[0]  # previous jump end
        for js, je, in zip(jumpstarts, jumpends):
            js_dec = data.index.get_loc(js) - 1
            js_dec = data.index[js_dec]
            frames_since_jump = js - pje
            if frames_since_jump > 2:
                js_test = js_dec
            else:
                js_test = js
            if any(data['grounded'].loc[js_test:je] == 1):
                data['jump_classification'].loc[js:je] = 1
                cumdeltas = game.frames.delta.loc[js:je].cumsum()
                data['jump_held_timer'].loc[js:je] = cumdeltas
                jump1ends.append(je)
                # build jump1ends_incr to use later on
                je_incr = data.index.get_loc(je) + 1  # getting the raw index location and incr by one
                if je_incr >= frameslen:
                    je_incr -= 1 # undo increment if last frame
                je_incr = data.index[je_incr]  # getting the labelled frame
                jump1ends_incr.append(je_incr)
            else:
                data['jump_classification'].loc[js:je] = 2
                jump2starts.append(js)
                jump2ends.append(je)
            pje = je

        grounded_indices = data.index[np.where(data['grounded'].values == 1)].to_list()
        # add the index of the final frame in order to classify jumps that don't land when the replay ends
        grounded_indices.append(data.index[-1])
        # the j2f + gi data capture both instances of resetting the jump timer --
        # when the second jump is used and when the ground resets the jump
        # TODO: use a resetter like in boost.py in order to vectorize the cumsumming.
        jump2frames = data.index[np.where(data['jump_classification'].values == 2)].to_list()
        j2fgi = jump2frames + grounded_indices
        j2fgi = np.sort(j2fgi)
        for j1e, j1e_incr in zip(jump1ends, jump1ends_incr):
            jtri = np.searchsorted(j2fgi, j1e_incr)  # jump timer reset index
            jtrf = j2fgi[jtri]  # jump timer reset frame
            cumdeltas = game.frames.delta.loc[j1e_incr:jtrf].cumsum()
            data['jump_2_expiration_timer'].loc[j1e_incr:jtrf] = cumdeltas

        # label when jumps are exhausted. Two conditions.
        # first condition is after the second jump is used
        for j2s in jump2starts:
            ngi = np.searchsorted(grounded_indices, j2s)  # next grounded index
            ngf = grounded_indices[ngi]  # next grounded frame
            # since players can jump on the frame they land, decrement ngf by one frame
            ngfi = data.index.get_loc(ngf) - 1
            ngf_dec = data.index[ngfi]
            data['jumps_exhausted'].loc[j2s:ngf_dec] = 1
        # second condition is when the jump_2_expiration_timer is above 1.25
        expirationframes = data.index[np.where(data['jump_2_expiration_timer'].values > 1.25)]
        data['jumps_exhausted'].loc[expirationframes] = 1

        player.jumps = data
        player.data = pd.concat([player.data, player.jumps], axis=1)

    return game
