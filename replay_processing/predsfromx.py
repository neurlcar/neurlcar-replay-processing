import os

from .Modules import *
import torch
import numpy as np
from replay_processing.get_base_dir import get_base_dir


def blue_c_zero_duel(x):
    controller_columns = [108, 115, 122, 130, 132, 135, 139, 142]
    for i in controller_columns:
        x[:, i] = 0
    return x


def orange_c_zero_duel(x):
    controller_columns = [109, 116, 127, 131, 133, 136, 140, 143]
    for i in controller_columns:
        x[:, i] = 0
    return x


def blue_c_zero_doubles(x, player: int):
    if player == 0:
        controller_columns = [140, 153, 166, 182, 186, 191, 199, 204]
    elif player == 1:
        controller_columns = [141, 154, 167, 183, 187, 192, 200, 205]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def orange_c_zero_doubles(x, player: int):
    if player == 0:
        controller_columns = [142, 155, 176, 184, 188, 193, 201, 206]
    elif player == 1:
        controller_columns = [143, 156, 177, 185, 189, 194, 202, 207]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def blue_c_zero_standard(x, player: int):
    if player == 0:
        controller_columns = [172, 191, 210, 234, 240, 247, 259, 266]
    elif player == 1:
        controller_columns = [173, 192, 211, 235, 241, 248, 260, 267]
    elif player == 2:
        controller_columns = [174, 193, 212, 236, 242, 249, 261, 268]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x

def orange_c_zero_standard(x, player: int):
    if player == 0:
        controller_columns = [175, 194, 225, 237, 243, 250, 262, 269]
    elif player == 1:
        controller_columns = [176, 195, 226, 238, 244, 251, 263, 270]
    elif player == 2:
        controller_columns = [177, 196, 227, 239, 245, 252, 264, 271]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def y_and_cgls_from_x(x):
    numcols = len(x[0])
    x = torch.from_numpy(x).float()
    if numcols == 144: # duel
        model_path = os.path.join(get_base_dir(), "models", "duel.pk1")
        model = PosVel_convSig_futureball_duel_deep()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        y_preds = model(x)

        goalchange_b = model(blue_c_zero_duel(x)) - y_preds
        goalchange_o = model(orange_c_zero_duel(x)) - y_preds
        goalchanges = [goalchange_b, goalchange_o]


    elif numcols == 208: # doubles
        model_path = os.path.join(get_base_dir(), "models", "doubles.pk1")
        model = PosVel_convSig_futureball_doubles_deep()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        y_preds = model(x)

        goalchange_b0 = model(blue_c_zero_doubles(x, 0)) - y_preds
        goalchange_o0 = y_preds - model(orange_c_zero_doubles(x, 0))

        goalchange_b1 = model(blue_c_zero_doubles(x, 1)) - y_preds
        goalchange_o1 = y_preds - model(orange_c_zero_doubles(x, 1))

        goalchanges = [goalchange_b0, goalchange_b1, goalchange_o0, goalchange_o1]

    elif numcols == 272: # standard
        model_path = os.path.join(get_base_dir(), "models", "standard.pk1")
        model = PosVel_convSig_futureball_standard_deep()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        y_preds = model(x)

        goalchange_b0 = model(blue_c_zero_standard(x, 0)) - y_preds
        goalchange_b1 = model(blue_c_zero_standard(x, 1)) - y_preds
        goalchange_b2 = model(blue_c_zero_standard(x, 2)) - y_preds

        goalchange_o0 = y_preds - model(orange_c_zero_standard(x, 0))
        goalchange_o1 = y_preds - model(orange_c_zero_standard(x, 1))
        goalchange_o2 = y_preds - model(orange_c_zero_standard(x, 2))

        goalchanges = [goalchange_b0, goalchange_b1, goalchange_b2, goalchange_o0, goalchange_o1, goalchange_o2]

    else:
        raise Exception

    return y_preds, goalchanges




