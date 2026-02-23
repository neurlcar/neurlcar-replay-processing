import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PosVel_convSig_futureball_standard_deep_relu(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 210
        self.name = "PosVel_convSig_futureball_deep_standard_relu"

        self.combinations = torch.combinations(torch.arange(42), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(11710, 512)  # 42 choose 3 = 11480 ... 11480 + 272 - 42 = 11710
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)
        self.activation = nn.ReLU()

    def separate_physics_inputs(self, inputs):
        pv_cols = [21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 42, 43, 44, 45, 46, 47, 48, 54, 55, 56,
                   78, 79, 80, 81, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103, 104, 105, 111, 112, 113]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)

        other_cols = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  25,  26,  27,  28,  29,
        37,  38,  39,  40,  41,  49,  50,  51,  52,  53,  57,  58,  59,
        60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
        73,  74,  75,  76,  77,  82,  83,  84,  85,  86,  94,  95,  96,
        97,  98, 106, 107, 108, 109, 110, 114, 115, 116, 117, 118, 119,
       120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
       146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
       159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
       172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
       185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
       198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
       211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
       224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
       237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
       250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262,
       263, 264, 265, 266, 267, 268, 269, 270, 271]

        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x



class PosVel_convSig_futureball_standard_deep(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 210
        self.name = "PosVel_convSig_futureball_deep_standard"

        self.combinations = torch.combinations(torch.arange(42), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(11710, 512)  # 42 choose 3 = 11480 ... 11480 + 272 - 42 = 11710
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)

    def separate_physics_inputs(self, inputs):
        pv_cols = [21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 42, 43, 44, 45, 46, 47, 48, 54, 55, 56,
                   78, 79, 80, 81, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103, 104, 105, 111, 112, 113]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)

        other_cols = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  25,  26,  27,  28,  29,
        37,  38,  39,  40,  41,  49,  50,  51,  52,  53,  57,  58,  59,
        60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
        73,  74,  75,  76,  77,  82,  83,  84,  85,  86,  94,  95,  96,
        97,  98, 106, 107, 108, 109, 110, 114, 115, 116, 117, 118, 119,
       120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
       146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
       159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
       172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
       185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
       198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
       211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
       224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
       237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
       250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262,
       263, 264, 265, 266, 267, 268, 269, 270, 271]

        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x


class RulesOfThumbSig4H_doubles(nn.Module):

    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumbsig4H_doubles"
        self.linear1 = nn.Linear(208, 161)
        self.linear2 = nn.Linear(161, 161)
        self.linear3 = nn.Linear(161, 161)
        self.linear4 = nn.Linear(161, 161)
        self.linear5 = nn.Linear(161, 161)
        self.linear6 = nn.Linear(161, 1)
        #self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        x = self.linear6(x)
        x = torch.sigmoid(x)
        return x

class PosVel_convSig_futureball_doubles_deeper(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 208
        self.name = "PosVel_convSig_futureball_deeper"

        self.combinations = torch.combinations(torch.arange(30), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(4238, 1024)  # 30 choose 3 = 4060 ... 4060 + 208 - 30 = 4238
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1024)
        self.linear5 = nn.Linear(1024, 1)

    def separate_physics_inputs(self, inputs):
        pv_cols = [15, 16, 17, 23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 60, 61, 62, 68, 69, 70, 71, 72,
                   78, 79, 80, 81, 82, 88, 89]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)

        other_cols = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  18,  19,  20,  21,  22,  28,  29,  30,  31,  32,  38,
        39,  40,  41,  42,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  57,  58,  59,  63,  64,  65,  66,  67,  73,  74,
        75,  76,  77,  83,  84,  85,  86,  87,  90,  91,  92,  93,  94,
        95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
       134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
       147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
       160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
       173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
       186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
       199, 200, 201, 202, 203, 204, 205, 206, 207]

        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)

        return x

class PosVel_convSig_futureball_doubles_deep(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 210
        self.name = "PosVel_convSig_futureball_deep"

        self.combinations = torch.combinations(torch.arange(30), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(4238, 512)  # 30 choose 3 = 4060 ... 4060 + 208 - 30 = 4238
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)

    def separate_physics_inputs(self, inputs):
        pv_cols = [15, 16, 17, 23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 60, 61, 62, 68, 69, 70, 71, 72,
                   78, 79, 80, 81, 82, 88, 89]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)

        other_cols = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  18,  19,  20,  21,  22,  28,  29,  30,  31,  32,  38,
        39,  40,  41,  42,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  57,  58,  59,  63,  64,  65,  66,  67,  73,  74,
        75,  76,  77,  83,  84,  85,  86,  87,  90,  91,  92,  93,  94,
        95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
       134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
       147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
       160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
       173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
       186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
       199, 200, 201, 202, 203, 204, 205, 206, 207]

        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x


class PosVel_convSig_futureball_duel_deep(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 144
        self.name = "PosVel_convSig_futureball_deep"

        self.combinations = torch.combinations(torch.arange(18), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(942, 512)  # 18 choose 3 = 816 ... 816 + 144 - 18 = 942
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)

    def separate_physics_inputs(self, inputs):
        pv_cols = [9, 10, 16, 17, 18, 24, 25, 26, 32, 42, 43, 49, 50, 51, 57, 58, 59, 65]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)
        other_cols = [0,  1,  2,  3,  4,  5,  6,  7,  8, 11, 12, 13, 14, 15, 19, 20, 21,
                      22, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44,
                      45, 46, 47, 48, 52, 53, 54, 55, 56, 60, 61, 62, 63, 64, 66, 67, 68,
                      69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                      86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                      102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                      113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                      139, 140, 141, 142, 143]
        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x



class PosVel_convSig_futureball_duel(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 144
        self.name = "PosVel_convSig_futureball"

        self.combinations = torch.combinations(torch.arange(18), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(942, 97)  # 18 choose 3 = 816 ... 816 + 144 - 18 = 942
        self.linear2 = nn.Linear(97, 97)  # 97 ~ 2/3 of 144
        self.linear3 = nn.Linear(97, 97)
        self.linear4 = nn.Linear(97, 1)

    def separate_physics_inputs(self, inputs):
        pv_cols = [9, 10, 16, 17, 18, 24, 25, 26, 32, 42, 43, 49, 50, 51, 57, 58, 59, 65]
        # other_cols = np.delete(np.arange(len(inputs)), pv_cols)
        other_cols = [0,  1,  2,  3,  4,  5,  6,  7,  8, 11, 12, 13, 14, 15, 19, 20, 21,
                      22, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44,
                      45, 46, 47, 48, 52, 53, 54, 55, 56, 60, 61, 62, 63, 64, 66, 67, 68,
                      69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                      86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                      102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                      113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                      139, 140, 141, 142, 143]
        physics_inputs = inputs[:, pv_cols]
        other_inputs = inputs[:, other_cols]

        physics_inputs = physics_inputs[:, self.combinations]  # size  (X, 2448)
        physics_inputs = physics_inputs.unsqueeze(1)  # size (X, 1, 2448)
        return physics_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x




class Physics_convSig(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "physics_convSig"

        self.combinations = torch.combinations(torch.arange(36), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(7218, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 1)


    def separate_physics_inputs(self, inputs):
        other_inputs = inputs[:, 36:]
        physics_inputs = inputs[:, self.combinations]  # size  21420
        p_inputs = physics_inputs.unsqueeze(1)  # size 1, 1, 21420
        return p_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = torch.sigmoid(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)

        return x


class Physics_convRELU(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "physics_convRELU"

        self.combinations = torch.combinations(torch.arange(36), r=3, with_replacement=False).flatten()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=3)

        self.linear1 = nn.Linear(7218, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 1)
        self.activation = nn.ReLU

    def separate_physics_inputs(self, inputs):
        other_inputs = inputs[:, 36:]
        physics_inputs = inputs[:, self.combinations]  # size  21420
        p_inputs = physics_inputs.unsqueeze(1)  # size 1, 1, 21420
        return p_inputs, other_inputs

    def forward(self, inputs):
        # split off the physics inputs and 1d convolve all 3wise combinations of them in order to simulate dot prods
        p_inputs, other_inputs = self.separate_physics_inputs(inputs)
        px = self.conv(p_inputs)
        px = px.squeeze()
        px = self.activation(px)

        # reunite the inputs
        x = torch.cat((px, other_inputs), dim=1)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)

        return x


class M101by5Linear(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "101by5"
        self.linear1 = nn.Linear(114, 101)
        self.linear3 = nn.Linear(101, 101)
        self.linear4 = nn.Linear(101, 101)
        self.linear5 = nn.Linear(101, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x


class M10by5Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "10by5"
        # Number of input features is 114
        self.linear1 = nn.Linear(114, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)
        self.linear4 = nn.Linear(10, 10)
        self.linear5 = nn.Linear(10, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x


class RulesOfThumb(nn.Module):
    # from jeffheaton, rulesof thumb are:
    # number of hidden neurons in a layer should be between input size and output size
    # number of hidden neurons should be 2/3 of input size + output size
    # number of hidden neurons should be less than 2x input layer
    # so, num hidden neurons (nhn) is constrained by:
    # 114 > nhn > 1
    # nhn ~= 2/3 * 115 == ~75 (closest prime is 73)
    # total nhn > 228
    # 2 hidden layers of 73 should satisfy this


    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumb"
        self.linear1 = nn.Linear(114, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x

class ControllerInputs_2H(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "CI_2H"
        self.linear1 = nn.Linear(106, 71)
        self.linear2 = nn.Linear(71, 71)
        self.linear3 = nn.Linear(71, 71)
        self.linear4 = nn.Linear(71, 8)


    def forward(self, inputs):
        x = self.linear1(inputs)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        xcopy = torch.sigmoid(x)

        #  normalize controller outputs
        #  controls are in order: boosting, handbrake, jump, pitch, roll, steer, throttle, yaw
        # throttle between -1 and 0.9921875 (254/256) because of bug in throttle feature scaling
        x = torch.clone(xcopy)
        x[:, 6] = (x[:, 6]*1.9921875) - 1
        # steer between -0.9921875 and 1 for similar reason
        x[:, 5] = (x[:, 6]*1.9921875) - 0.9921875

        # yaw, pitch, roll, between -1 and 1
        x[:, (3, 4, 7)] = (x[:, (3, 4, 7)] * 2) - 1
        # handbrake, boosting, jump binary
        x[:, (0, 1, 2)] = (x[:, (0, 1, 2)] > 0.5).int().float()

        return x


class ControllerInputs_2H_relu(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "CI_2H_relu"
        self.linear1 = nn.Linear(106, 71)
        self.linear2 = nn.Linear(71, 71)
        self.linear3 = nn.Linear(71, 71)
        self.linear4 = nn.Linear(71, 8)


    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        xcopy = torch.sigmoid(x)

        #  normalize controller outputs
        #  controls are in order: boosting, handbrake, jump, pitch, roll, steer, throttle, yaw
        # throttle between -1 and 0.9921875 (254/256) because of bug in throttle feature scaling
        x = torch.clone(xcopy)
        x[:, 6] = (x[:, 6]*1.9921875) - 1
        # steer between -0.9921875 and 1 for similar reason
        x[:, 5] = (x[:, 6]*1.9921875) - 0.9921875

        # yaw, pitch, roll, between -1 and 1
        x[:, (3, 4, 7)] = (x[:, (3, 4, 7)] * 2) - 1
        # handbrake, boosting, jump binary
        x[:, (0, 1, 2)] = (x[:, (0, 1, 2)] > 0.5).int().float()

        return x


class ControllerInputs_4H_relu(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "CI_4H_relu"
        self.linear1 = nn.Linear(106, 71)
        self.linear2 = nn.Linear(71, 71)
        self.linear3 = nn.Linear(71, 71)
        self.linear4 = nn.Linear(71, 71)
        self.linear5 = nn.Linear(71, 71)
        self.linear6 = nn.Linear(71, 8)


    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        xcopy = torch.sigmoid(x)

        #  normalize controller outputs
        #  controls are in order: boosting, handbrake, jump, pitch, roll, steer, throttle, yaw
        # throttle between -1 and 0.9921875 (254/256) because of bug in throttle feature scaling
        x = torch.clone(xcopy)
        x[:, 6] = (x[:, 6]*1.9921875) - 1
        # steer between -0.9921875 and 1 for similar reason
        x[:, 5] = (x[:, 6]*1.9921875) - 0.9921875

        # yaw, pitch, roll, between -1 and 1
        x[:, (3, 4, 7)] = (x[:, (3, 4, 7)] * 2) - 1
        # handbrake, boosting, jump binary
        x[:, (0, 1, 2)] = (x[:, (0, 1, 2)] > 0.5).int().float()

        return x




class RulesOfThumb4H(nn.Module):

    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumb4H"
        self.linear1 = nn.Linear(114, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 73)
        self.linear5 = nn.Linear(73, 73)
        self.linear6 = nn.Linear(73, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = torch.sigmoid(x)
        return x


class RulesOfThumbSig4H(nn.Module):

    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumbsig4H"
        self.linear1 = nn.Linear(114, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 73)
        self.linear5 = nn.Linear(73, 73)
        self.linear6 = nn.Linear(73, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        x = self.linear6(x)
        x = torch.sigmoid(x)
        return x

class LargeDeep4H(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "LargeDeep4H"
        self.linear1 = nn.Linear(114, 227)
        self.linear2 = nn.Linear(227, 227)
        self.linear3 = nn.Linear(227, 227)
        self.linear3 = nn.Linear(227, 227)
        self.linear3 = nn.Linear(227, 227)
        self.linear4 = nn.Linear(227, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x


class RulesOfThumbSig3H(nn.Module):
    # use sigmoid and also add a third hidden layer

    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumb_sig3H"
        self.linear1 = nn.Linear(114, 73)
        self.hiddenlayer1 = nn.Linear(73, 73)
        self.hiddenlayer2 = nn.Linear(73, 73)
        self.hiddenlayer3 = nn.Linear(73, 73)
        self.linear2 = nn.Linear(73, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = torch.sigmoid(x)

        x = self.hiddenlayer1(x)
        x = torch.sigmoid(x)
        x = self.hiddenlayer2(x)
        x = torch.sigmoid(x)
        x = self.hiddenlayer3(x)
        x = torch.sigmoid(x)

        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x



class TaperedPrimesTwoThirds(nn.Module):
    # this takes from Rules of thumb and adds tapering layers.
    # three hidden layers with tapering descending primes, tapering by ~2/3
    # 73, 47, 31, 19, 11, 7, 5, 3, 1
    def __init__(self):
        super().__init__()
        self.name = "tapered_primes_23"
        # Number of input features is 114
        self.linear1 = nn.Linear(114, 73)
        self.linear2 = nn.Linear(73, 47)
        self.linear3 = nn.Linear(47, 31)
        self.linear4 = nn.Linear(31, 19)
        self.linear5 = nn.Linear(19, 11)
        self.linear6 = nn.Linear(11, 7)
        self.linear7 = nn.Linear(5, 3)
        self.linear8 = nn.Linear(3, 1)


        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = F.relu(x)
        x = self.linear7(x)
        x = F.relu(x)
        x = self.linear8(x)
        x = torch.sigmoid(x)
        return x

class TaperedPrimesHalved(nn.Module):
    # this takes from Rules of thumb and adds tapering layers.
    # three hidden layers with tapering descending primes, tapering by ~1/2
    # 59, 29, 13, 7, 3, 1
    def __init__(self):
        super().__init__()
        self.name = "tapered_primes_halved"
        # Number of input features is 114
        self.linear1 = nn.Linear(114, 59)
        self.linear2 = nn.Linear(73, 29)
        self.linear3 = nn.Linear(29, 13)
        self.linear4 = nn.Linear(13, 7)
        self.linear5 = nn.Linear(7, 3)
        self.linear6 = nn.Linear(3, 1)

        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = torch.sigmoid(x)
        return x

class TaperedPrimesThirded(nn.Module):
    # this takes from Rules of thumb and adds tapering layers.
    # three hidden layers with tapering descending primes, tapering by ~1/2
    # 37, 13, 5, 1
    def __init__(self):
        super().__init__()
        self.name = "tapered_primes_thirded"
        # Number of input features is 114
        self.linear1 = nn.Linear(114, 37)
        self.linear2 = nn.Linear(37, 13)
        self.linear3 = nn.Linear(13, 5)
        self.linear4 = nn.Linear(5, 1)

        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x


class RulesOfThumbTapered(nn.Module):
    # same as rules of thumb, just with a tapering layer before the output (11 neurons)

    def __init__(self):
        super().__init__()
        # Number of input features is 114
        self.name = "thumb_tapered"
        self.linear1 = nn.Linear(114, 73)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 11)
        self.linear5 = nn.Linear(11, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x

class M73N8H(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "10by5"
        # Number of input features is 114
        self.linear1 = nn.Linear(114, 10)
        self.linear2 = nn.Linear(73, 73)
        self.linear3 = nn.Linear(73, 73)
        self.linear4 = nn.Linear(73, 73)
        self.linear5 = nn.Linear(73, 73)
        self.linear6 = nn.Linear(73, 73)
        self.linear7 = nn.Linear(73, 73)
        self.linear8 = nn.Linear(73, 73)
        self.linear9 = nn.Linear(73, 73)
        self.linear10 = nn.Linear(73, 73)
        self.linear11 = nn.Linear(73, 1)
        self.activation = nn.ReLU
        self.sig = nn.Sigmoid

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.sig(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = F.relu(x)
        x = self.linear7(x)
        x = F.relu(x)
        x = self.linear8(x)
        x = F.relu(x)
        x = self.linear9(x)
        x = F.relu(x)
        x = self.linear10(x)
        x = F.relu(x)
        x = self.linear11(x)
        x = torch.sigmoid(x)
        return x


class TinyPreNormBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D]
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         key_padding_mask=key_padding_mask)
        x = x + self.dropout(y)
        y = self.ffn(self.ln2(x))
        x = x + self.dropout(y)
        return x


class TokenAttnDualHead(nn.Module):
    """
    Expects:
      tokens: list of 37 tensors from ReplayGraphStreamer, each [B, D_i]
      token_mask: [37] or [B, 37] bool (True = padded/dummy to be ignored)

    Token type by index:
      0–5: player nodes
       6 : ball node
       7 : boost node
       8 : global node
      9–29: dyn↔dyn edge tokens
     30–36: dyn↔static edge tokens
    """

    def __init__(self):
        super().__init__()
        self.name = "TokenAttnDualHead"

        self.num_tokens = 37
        self.token_latent_dim = 64

        # Feature dims are determined by your NODE_INDEX_MAP and edge helpers:
        player_dim = 32
        ball_dim = 42
        boost_dim = 34
        global_dim = 4
        dyn_dyn_dim = 18
        dyn_static_dim = 56

        self.embed_player     = nn.Linear(player_dim,     64)
        self.embed_ball       = nn.Linear(ball_dim,       64)
        self.embed_boost      = nn.Linear(boost_dim,      64)
        self.embed_global     = nn.Linear(global_dim,     64)
        self.embed_dyn_dyn    = nn.Linear(dyn_dyn_dim,    64)
        self.embed_dyn_static = nn.Linear(dyn_static_dim, 64)

        self.block1 = TinyPreNormBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.1)
        self.block2 = TinyPreNormBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.1)

        self.linear1 = nn.Linear(self.num_tokens * 64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.head_wsn = nn.Linear(64, 1)
        self.head_imm = nn.Linear(64, 1)

        self.mixing_param_logit = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, tokens, token_mask=None):
        """
        tokens: list length 37, each [B, D_i]
        token_mask: [37] or [B, 37] bool, True = masked/padded
        """
        assert isinstance(tokens, (list, tuple)), "tokens must be a list/tuple of tensors"
        assert len(tokens) == self.num_tokens, f"Expected {self.num_tokens} tokens, got {len(tokens)}"

        batch_size = tokens[0].size(0)

        embedded = []
        for idx, t in enumerate(tokens):
            # t: [B, D_i]
            if idx <= 5:
                # player nodes
                e = self.embed_player(t)
            elif idx == 6:
                # ball node
                e = self.embed_ball(t)
            elif idx == 7:
                # boost node
                e = self.embed_boost(t)
            elif idx == 8:
                # global node
                e = self.embed_global(t)
            elif 9 <= idx <= 29:
                # dyn↔dyn edges
                e = self.embed_dyn_dyn(t)
            else:  # 30–36
                # dyn↔static edges
                e = self.embed_dyn_static(t)

            # shape [B, 64] -> [B, 1, 64]
            embedded.append(e.unsqueeze(1))

        # [B, 37, 64]
        x = torch.cat(embedded, dim=1)

        # Prepare key_padding_mask for MultiheadAttention
        key_padding_mask = None
        if token_mask is not None:
            # token_mask coming from MASKS[...] is [37]
            if token_mask.dim() == 1:
                key_padding_mask = token_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                key_padding_mask = token_mask
            # MultiheadAttention expects bool with True = ignore
            key_padding_mask = key_padding_mask.bool()

        x = self.block1(x, key_padding_mask=key_padding_mask)
        x = self.block2(x, key_padding_mask=key_padding_mask)

        x = x.reshape(batch_size, self.num_tokens * self.token_latent_dim)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        wsn_logit = self.head_wsn(x)  # [B, 1]
        imm_logit = self.head_imm(x)  # [B, 1]

        p_wsn = torch.sigmoid(wsn_logit).squeeze(-1)  # [B]
        p_imm = torch.sigmoid(imm_logit).squeeze(-1)  # [B]

        return p_wsn, p_imm


class TokenAttnDualHeadOnnxWrap(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *args):
        *tokens, token_mask = args
        return self.base(list(tokens), token_mask=token_mask)


