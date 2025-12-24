from carball.json_parser.game import Game
from .hitbox import get_hitbox, get_wheel_displacements, get_wheel_radii

def get_wheel_positions(game: Game):

    for player in game.players:
        car = player.loadout[0]['car']
        hitbox = get_hitbox(car)
        rj_xyz = player.data[['pos_x', 'pos_y', 'pos_z']]
        wheel_displacements = get_wheel_displacements(hitbox)
        wheelradii = get_wheel_radii(hitbox)






