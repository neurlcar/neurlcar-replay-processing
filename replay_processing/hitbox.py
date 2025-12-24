import numpy as np

def get_hitbox(car):
    '''
    takes a car item id and returns the associated hitbox label.
    hitboxlabels = {'Octane': 0, 'Dominus': 1, 'Plank': 2, 'Breakout': 3, 'Hybrid': 4, 'Merc': 5}
    maintained list of items here: https://github.com/RLBot/RLBotGUI/blob/master/rlbot_gui/gui/csv/items.csv
    psyonix maintains hitbox lists for their bodies here:
                    https://support.rocketleague.com/hc/en-us/articles/360029832054-Rocket-League-Car-Hitboxes

    caritemdict last updated 3/3/2022 10AM EST
    '''

    caritemdict = {21: 0, 22: 3, 23: 0, 24: 2, 25: 0, 26: 0, 27: 0, 28: 4, 29: 1, 30: 5, 31: 4, 402: 0, 403: 1, 404: 0,
                   523: 0, 597: 1, 600: 1, 607: 0, 625: 0, 723: 0, 803: 2, 1018: 1, 1159: 4, 1171: 1, 1172: 0, 1286: 1,
                   1295: 0, 1300: 0, 1317: 4, 1416: 3, 1475: 0, 1478: 0, 1533: 0, 1568: 0, 1603: 2, 1623: 0, 1624: 4,
                   1675: 1, 1689: 1, 1691: 2, 1856: 4, 1883: 1, 1894: 3, 1919: 2, 1932: 3, 2070: 1, 2268: 1, 2269: 4,
                   2298: 1, 2313: 0, 2665: 0, 2666: 1, 2853: 0, 2919: 0, 2949: 0, 2950: 1, 2951: 1, 3031: 3, 3155: 1,
                   3156: 1, 3157: 1, 3265: 1, 3311: 3, 3426: 1, 3451: 4, 3582: 4, 3594: 2, 3614: 2, 3622: 2, 3702: 4,
                   3875: 1, 3879: 1, 3880: 1, 4014: 1, 4155: 1, 4268: 2, 4284: 0, 4318: 0, 4319: 0, 4320: 0, 4367: 1,
                   4472: 1, 4473: 1, 4745: 1, 4770: 1, 4780: 5, 4781: 1, 4861: 1, 4864: 1, 4906: 0, 5020: 0, 5039: 0,
                   5188: 0, 5265: 1, 5361: 0, 5470: 4, 5488: 4, 5547: 0, 5709: 1, 5713: 0, 5773: 1, 5823: 1, 5837: 0,
                   5858: 1, 5879: 4, 5951: 0, 5964: 1, 5979: 1, 6122: 0, 6243: 3, 6244: 1, 6247: 1, 6260: 1, 6489: 3,
                   6836: 1, 6939: 0}

    if car in caritemdict.keys():
        hitbox = caritemdict[car]
    else:
        hitbox = 0  # default to an octane hitbox for cars that aren't in the hitboxes dict

    return hitbox

def get_vertex_displacements(hitbox):
    # in order, front left top, front right top, front right bottom, front left bottom, back left top, back right top,
    # back right bottom, back left bottom
    octane_hitbox_xyz = [118.01, 84.20, 36.16]
    dom_hitbox_xyz = [127.93, 83.28, 31.30]
    plank_hitbox_xyz = [128.82, 84.67, 29.39]
    breakout_hitbox_xyz = [131.49, 80.52, 30.30]
    hybrid_hitbox_xyz = [127.02, 82.19, 34.16]
    merc_hitbox_xyz = [120.72, 76.71, 41.66]



    octane_rjdist ={'rj_x_offset': 13.88, 'rj_z_offset': 20.75, 'rj_to_ground': 17.00, 'rj_to_front': 72.88,
                            'rj_to_top': 38.83, 'rj_to_side': 42.10, 'rj_to_back': 45.13}
    octane_rjdist['rj_to_bottom'] = octane_rjdist['rj_to_top'] - octane_hitbox_xyz[2]

    dom_rjdist = {'rj_x_offset': 9.00, 'rj_z_offset': 15.75, 'rj_to_ground': 17.05, 'rj_to_front': 72.96,
                            'rj_to_top': 31.40, 'rj_to_side': 41.64, 'rj_to_back': 54.96}
    dom_rjdist['rj_to_bottom'] = dom_rjdist['rj_to_top'] - dom_hitbox_xyz[2]

    plank_rjdist = {'rj_x_offset':9.01 , 'rj_z_offset':12.09 , 'rj_to_ground':18.65 , 'rj_to_front':73.42 ,
                            'rj_to_top':26.79 , 'rj_to_side':42.34 , 'rj_to_back': 55.4}
    plank_rjdist['rj_to_bottom'] = plank_rjdist['rj_to_top'] - plank_hitbox_xyz[2]

    breakout_rjdist = {'rj_x_offset':12.5 , 'rj_z_offset': 11.75 , 'rj_to_ground': 18.33, 'rj_to_front':78.25 ,
                            'rj_to_top':26.9 , 'rj_to_side': 40.26 , 'rj_to_back': 53.25}
    breakout_rjdist['rj_to_bottom'] = breakout_rjdist['rj_to_top'] - breakout_hitbox_xyz[2]

    hybrid_rjdist = {'rj_x_offset': 13.88 , 'rj_z_offset': 20.75 , 'rj_to_ground': 17.01, 'rj_to_front': 77.39,
                            'rj_to_top': 37.83, 'rj_to_side': 41.09, 'rj_to_back': 49.63}
    hybrid_rjdist['rj_to_bottom'] = hybrid_rjdist['rj_to_top'] - hybrid_hitbox_xyz[2]

    merc_rjdist =  {'rj_x_offset':11.38 , 'rj_z_offset':21.5 , 'rj_to_ground':18.78 , 'rj_to_front': 71.74,
                            'rj_to_top':42.33, 'rj_to_side': 38.36, 'rj_to_back': 48.98}
    merc_rjdist['rj_to_bottom'] = merc_rjdist['rj_to_top'] - merc_hitbox_xyz[2]

    octane_vertex_displacements = [[octane_rjdist['rj_to_front'], -octane_rjdist['rj_to_side'], octane_rjdist['rj_to_top']],
                                    [octane_rjdist['rj_to_front'], octane_rjdist['rj_to_side'], octane_rjdist['rj_to_top']],
                                    [octane_rjdist['rj_to_front'], octane_rjdist['rj_to_side'], octane_rjdist['rj_to_bottom']],
                                    [octane_rjdist['rj_to_front'], -octane_rjdist['rj_to_side'], octane_rjdist['rj_to_bottom']],
                                    [-octane_rjdist['rj_to_back'], -octane_rjdist['rj_to_side'], octane_rjdist['rj_to_top']],
                                    [-octane_rjdist['rj_to_back'], octane_rjdist['rj_to_side'], octane_rjdist['rj_to_top']],
                                    [-octane_rjdist['rj_to_back'], octane_rjdist['rj_to_side'], octane_rjdist['rj_to_bottom']],
                                    [-octane_rjdist['rj_to_back'], -octane_rjdist['rj_to_side'], octane_rjdist['rj_to_bottom']]]
    dom_vertex_displacements = [[dom_rjdist['rj_to_front'], -dom_rjdist['rj_to_side'], dom_rjdist['rj_to_top']],
                                    [dom_rjdist['rj_to_front'], dom_rjdist['rj_to_side'], dom_rjdist['rj_to_top']],
                                    [dom_rjdist['rj_to_front'], dom_rjdist['rj_to_side'], dom_rjdist['rj_to_bottom']],
                                    [dom_rjdist['rj_to_front'], -dom_rjdist['rj_to_side'], dom_rjdist['rj_to_bottom']],
                                    [-dom_rjdist['rj_to_back'], -dom_rjdist['rj_to_side'], dom_rjdist['rj_to_top']],
                                    [-dom_rjdist['rj_to_back'], dom_rjdist['rj_to_side'], dom_rjdist['rj_to_top']],
                                    [-dom_rjdist['rj_to_back'], dom_rjdist['rj_to_side'], dom_rjdist['rj_to_bottom']],
                                    [-dom_rjdist['rj_to_back'], -dom_rjdist['rj_to_side'], dom_rjdist['rj_to_bottom']]]
    plank_vertex_displacements = [[plank_rjdist['rj_to_front'], -plank_rjdist['rj_to_side'], plank_rjdist['rj_to_top']],
                             [plank_rjdist['rj_to_front'], plank_rjdist['rj_to_side'], plank_rjdist['rj_to_top']],
                             [plank_rjdist['rj_to_front'], plank_rjdist['rj_to_side'], plank_rjdist['rj_to_bottom']],
                             [plank_rjdist['rj_to_front'], -plank_rjdist['rj_to_side'], plank_rjdist['rj_to_bottom']],
                             [-plank_rjdist['rj_to_back'], -plank_rjdist['rj_to_side'], plank_rjdist['rj_to_top']],
                             [-plank_rjdist['rj_to_back'], plank_rjdist['rj_to_side'], plank_rjdist['rj_to_top']],
                             [-plank_rjdist['rj_to_back'], plank_rjdist['rj_to_side'], plank_rjdist['rj_to_bottom']],
                             [-plank_rjdist['rj_to_back'], -plank_rjdist['rj_to_side'], plank_rjdist['rj_to_bottom']]]
    breakout_vertex_displacements = [[breakout_rjdist['rj_to_front'], -breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_top']],
                             [breakout_rjdist['rj_to_front'], breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_top']],
                             [breakout_rjdist['rj_to_front'], breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_bottom']],
                             [breakout_rjdist['rj_to_front'], -breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_bottom']],
                             [-breakout_rjdist['rj_to_back'], -breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_top']],
                             [-breakout_rjdist['rj_to_back'], breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_top']],
                             [-breakout_rjdist['rj_to_back'], breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_bottom']],
                             [-breakout_rjdist['rj_to_back'], -breakout_rjdist['rj_to_side'], breakout_rjdist['rj_to_bottom']]]
    hybrid_vertex_displacements = [[hybrid_rjdist['rj_to_front'], -hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_top']],
                             [hybrid_rjdist['rj_to_front'],hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_top']],
                             [hybrid_rjdist['rj_to_front'], hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_bottom']],
                             [hybrid_rjdist['rj_to_front'], -hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_bottom']],
                             [-hybrid_rjdist['rj_to_back'], -hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_top']],
                             [-hybrid_rjdist['rj_to_back'], hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_top']],
                             [-hybrid_rjdist['rj_to_back'], hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_bottom']],
                             [-hybrid_rjdist['rj_to_back'], -hybrid_rjdist['rj_to_side'], hybrid_rjdist['rj_to_bottom']]]
    merc_vertex_displacements = [[merc_rjdist['rj_to_front'], -merc_rjdist['rj_to_side'], merc_rjdist['rj_to_top']],
                             [merc_rjdist['rj_to_front'], merc_rjdist['rj_to_side'], merc_rjdist['rj_to_top']],
                             [merc_rjdist['rj_to_front'], merc_rjdist['rj_to_side'], merc_rjdist['rj_to_bottom']],
                             [merc_rjdist['rj_to_front'], -merc_rjdist['rj_to_side'], merc_rjdist['rj_to_bottom']],
                             [-merc_rjdist['rj_to_back'], -merc_rjdist['rj_to_side'], merc_rjdist['rj_to_top']],
                             [-merc_rjdist['rj_to_back'], merc_rjdist['rj_to_side'], merc_rjdist['rj_to_top']],
                             [-merc_rjdist['rj_to_back'], merc_rjdist['rj_to_side'], merc_rjdist['rj_to_bottom']],
                             [-merc_rjdist['rj_to_back'], -merc_rjdist['rj_to_side'], merc_rjdist['rj_to_bottom']]]





    disp_list = [octane_vertex_displacements, dom_vertex_displacements, plank_vertex_displacements,
                 breakout_vertex_displacements, hybrid_vertex_displacements, merc_vertex_displacements]

    vertices = disp_list[hitbox]

    return vertices


def get_ground_to_root_joint(hitbox):
    ground_to_rj_list = [17.10, 17.15, 18.75, 18.43, 17.11, 18.88]
    ground_to_rj = ground_to_rj_list[hitbox]
    return ground_to_rj

def get_wheel_displacements(hitbox):
    # in order, front left, front right, back right, back left, coords xyz relative to rj
    octane_displacements = [[51.25, -25.9, -6], [51.25, 25.9, -6], [-33.75, 29.5, -4.3], [-33.75, -29.5, -4.3]]
    dominus_displacements = [[50.3,-31.1,-6.2],[50.3, 31.1, -6.2],[-34.75, 33.00, -6.1],[-34.75, -33.00, -6.1]]
    plank_displacements = [[49.97, -27.80, -7.83],[49.97, 27.80, -7.83],[-35.43, 20.28, -3.83],[-35.43, -20.28, -3.83]]
    breakout_displacements = [[51.50, -26.67, -5.95],[51.50, 26.67, -5.95],[-35.75, 35.00, -5.91],[-35.75, -35.00, -5.91]]
    hybrid_displacements = [[51.25, -25.90, -6.00],[51.25, 25.90, -6.00],[-34.00, 29.50, -4.3],[-34.00, -29.50, -4.3]]
    merc_displacements = [[51.25, -25.90, -6.00],[51.25, 25.90, -6.00],[-33.75, 29.50, -5.6],[-33.75, -29.50, -5.6]]

    disp_list = [octane_displacements, dominus_displacements, plank_displacements, breakout_displacements,
                 hybrid_displacements, merc_displacements]
    wheel_displacements = disp_list[hitbox]

    return wheel_displacements



def get_wheel_radii(hitbox):
    frontradii = [12.5, 12, 12.5, 13.5, 12.5, 15]
    backradii = [15, 13.5, 17, 15, 15, 15]
    radii = [frontradii[hitbox], backradii[hitbox]]
    return radii








