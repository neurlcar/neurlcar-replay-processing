from carball.json_parser.game import Game
import numpy as np

def append_demo_timers(game: Game):
    for player in game.players:
        data = player.data
        data['demo_timer'] = 0
        xyz = data[['pos_x', 'pos_y', 'pos_z']]
        xyzshiftdown = xyz.shift(periods=1)
        xyzshiftup = xyz.shift(periods=-1)

        demostarts = data.loc[xyz['pos_x'].isnull() & -xyzshiftdown['pos_x'].isnull()].index.to_list()
        demoends = data.loc[xyz['pos_x'].isnull() & -xyzshiftup['pos_x'].isnull()].index.to_list()
        otframe = None
        #  there's a bug here where it's calling ot frames demos, so remove ot frames from demostarts and ends
        if any(game.frames.is_overtime):
            isot = game.frames.is_overtime.fillna(False)
            otframe = np.argmax(isot)  # finds the integer index number of the first True (e.g. 11302)
            otframe = game.frames.index[otframe]  # find the labelled index number of the first True (e.g. 11313)

        demostarts = [x for x in demostarts if x != otframe]
        demoends = [x for x in demoends if x != otframe]

        # if replay ends with car demoed, then there won't be a demoend there, so add one
        lastindex = data.index[-1]
        if len(demostarts) != len(demoends):
            demoends.append(lastindex)



        for demostart, demoend in zip(demostarts, demoends):
            #print(player.name + " demoed on frame " + str(demostart) + " respawned on frame " + str(demoend))
            cumdeltas = game.frames.delta.loc[demostart:demoend].cumsum()
            data['demo_timer'].loc[demostart:demoend] = cumdeltas
        player.data = data
    return game









