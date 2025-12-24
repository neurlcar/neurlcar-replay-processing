from carball.json_parser.game import Game
import numpy as np
import pandas as pd
from .gamefromreplay import game_from_replay
from .hitbox import get_hitbox
import gc


def process_replay(game: Game):

    """

    inputs a Game class object, outputs a processed df to be used for training

    """
    # remove unwanted columns from game.ball
    game.ball = game.ball.drop(['hit_team_no'], axis=1)
    # remove unwanted columns from game.frames
    game.frames = game.frames.drop(['ball_has_been_hit', 'replicated_seconds_remaining'], axis=1)
    # remove unwanted columns from player[N].data
    for player in game.players:
        player.data = player.data.drop(['ping', 'dodge_active', 'double_jump_active',
                                        'boost_active', 'boost_delta', 'jump_active'], axis=1)

    # give unique colnames to game.ball columns
    ball = game.ball
    ballcolnames = []
    for i in range(len(ball.columns)):
        colname = ball.columns[i] + "_ball"
        ballcolnames = np.append(ballcolnames, colname)
    ball.columns = ballcolnames

    # give unique colnames to players[i] columns
    players = game.players
    orange_player_counter = 0
    blue_player_counter = 0
    for i in range(len(players)):
        playercolnames = []
        data = players[i].data
        # get car (hitbox should be latent information in car number)
        car = game.players[i].loadout[0]['car']
        hitbox = get_hitbox(car)
        #name = game.players[i].name
        is_orange = game.players[i].is_orange

        players[i].data['hitbox'] = hitbox
        players[i].data['team'] = is_orange


        # generate colnames
        for n in range(len(data.columns)):
            if is_orange:
                colname = str(data.columns[n]) + "_orange_player_" + str(orange_player_counter)
            else:
                colname = str(data.columns[n]) + "_blue_player_" + str(blue_player_counter)
            playercolnames = np.append(playercolnames, colname)
        players[i].data.columns = playercolnames

        if is_orange:
            orange_player_counter += 1
        else:
            blue_player_counter += 1

    # combine all players into a df
    combinedplayers = pd.DataFrame()
    for i in range(len(players)):
        combinedplayers = pd.concat([combinedplayers, players[i].data], axis=1)

    # combine the frames, ball, futureball, and players into a df to be used for training
    trainingreplay = pd.concat([game.frames, game.ball, game.futureball, combinedplayers], axis=1)

    # add attributes
    trainingreplay.attrs['num_players'] = len(game.players)
    trainingreplay.attrs['team_size'] = game.team_size

    if game.team_size == 1:
        gamemode = 'duel'
    elif game.team_size == 2:
        gamemode = 'doubles'
    elif game.team_size == 3:
        gamemode = 'standard'
    else:
        gamemode = None


    del game
    gc.collect()  # delete the game object then collect garbage

    return trainingreplay, gamemode


if __name__ == '__main__':
    sampledir = "H:/python/Database/replays/replays_sf5_3v3_ssl/5932ecdd-aebb-4b2c-885d-16c1b388611e.replay"

    g = game_from_replay(sampledir)
    df = process_replay(g)






