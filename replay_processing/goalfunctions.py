from carball.json_parser.game import Game

def get_goal_frames(game: Game):
    goalframes = []
    for i in range(0, len(game.goals)):
        goalframe = game.goals[i].frame_number
        goalframes.append(goalframe)
    return goalframes

def get_goal_tag_sequence(game: Game):
    goaltagsequence = []
    for i in range(len(game.goals)):
        team = game.goals[i].player_team
    # orange is 1, blue is 0
        goaltagsequence.append(team)

    return goaltagsequence






