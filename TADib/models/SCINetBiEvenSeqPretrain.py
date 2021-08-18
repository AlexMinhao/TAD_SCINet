import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
from torch.nn.utils import weight_norm
import argparse
import numpy as np
import random
random.seed(80) #80

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        # self.conv_even = lambda x: x[:, ::2, :]
        # self.conv_odd = lambda x: x[:, 1::2, :]

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, args, in_planes, splitting=True, dropout=0.5,
                 simple_lifting=False):
        super(Interactor, self).__init__()
        self.modified = 1
        kernel_size = args.kernel
        dilation = 1
        self.dropout = args.dropout

        pad = dilation * (kernel_size - 1) // 2 + 1  # 2 1 0 0
        # pad = k_size // 2
        self.splitting = splitting
        self.split = Splitting()

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReplicationPad1d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Dropout(self.dropout),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),

                nn.Dropout(self.dropout),
                nn.Tanh()
            ]
        else:
            size_hidden = args.hidden_size
            modules_P += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                          kernel_size=kernel_size, dilation=dilation, stride=1, groups=args.groups),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(int(in_planes * size_hidden), in_planes,
                          kernel_size=3, stride=1, groups=args.groups),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                          kernel_size=kernel_size, dilation=dilation, stride=1, groups=args.groups),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(int(in_planes * size_hidden), in_planes,
                          kernel_size=3, stride=1, groups=args.groups),
                nn.Tanh()
            ]
            if self.modified:
                modules_phi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                              kernel_size=kernel_size, dilation=dilation, stride=1, groups=args.groups),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(int(in_planes * size_hidden), in_planes,
                              kernel_size=3, stride=1, groups=args.groups),
                    nn.Tanh()
                ]
                modules_psi += [
                    nn.ReplicationPad1d(pad),
                    nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                              kernel_size=kernel_size, dilation=dilation, stride=1, groups=args.groups),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(int(in_planes * size_hidden), in_planes,
                              kernel_size=3, stride=1, groups=args.groups),
                    nn.Tanh()
                ]
                self.phi = nn.Sequential(*modules_phi)
                self.psi = nn.Sequential(*modules_psi)

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            # 3  224  112
            # 3  112  112
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x
        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # x_odd_update = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            # x_even_update = x_even.mul(torch.exp(self.psi(x_odd_update))) + self.U(x_odd_update)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, args, in_planes,
                 simple_lifting=False):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(args,
                                in_planes=in_planes,
                                simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        # (L, H)
        (x_even_update, x_odd_update) = self.level(x)  # 10 3 224 224

        return (x_even_update, x_odd_update)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.disable_conv = disable_conv  # in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


class LevelIDCN(nn.Module):
    def __init__(self, args, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelIDCN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.interact = InteractorLevel(args, in_planes,
                                        simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)  # 10 9 128

        if self.bootleneck:
            return self.bootleneck(x_even_update).permute(0, 2, 1), x_odd_update
        else:
            return x_even_update.permute(0, 2, 1), x_odd_update




class EncoderTree(nn.Module):
    def __init__(self, level_layers, level_parts, num_layers=3, Encoder=True, norm_layer=None):
        super(EncoderTree, self).__init__()
        self.level_layers = nn.ModuleList(level_layers)
        self.conv_layers = None  # nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        # self.level_part = [[1, 1], [0, 0], [0, 0]]
        self.level_part = level_parts  # [[0, 1], [0, 0]]
        self.layers = num_layers  # 3 if len(level_parts) == 7 else 2
        print('layer number:', self.layers)
        self.count_levels = 0
        self.ecoder = Encoder

    def reOrder(self, num_of_length, layer=2):
        N = num_of_length
        n = list(range(1, N + 1, 1))
        remain = [i % 2 for i in n]

        n_1 = []
        for i in range(N):
            if remain[i] > 0:
                n_1.append((n[i] + 1) / 2 + N / 2)
            else:
                n_1.append(n[i] / 2)

        remain = [i % 2 for i in n_1]

        n_2 = []
        rem4 = [i % 4 for i in n]

        for i in range(N):
            if rem4[i] == 0:
                n_2.append(int(n[i] / 4))

            elif rem4[i] == 1:

                n_2.append(int((3 * N + 3) / 4 + n[i] / 4))
            elif rem4[i] == 2:
                n_2.append(int((1 * N + 2) / 4 + n[i] / 4))
            elif rem4[i] == 3:
                n_2.append(int((2 * N + 1) / 4 + n[i] / 4))
            else:
                print("Error!")

        n_3 = []
        rem8 = [i % 8 for i in n]
        for i in range(N):
            if rem8[i] == 0:
                n_3.append(int(n[i] / 8))
            elif rem8[i] == 1:
                n_3.append(int(n[i] / 8 + (7 * N + 7) / 8))
            elif rem8[i] == 2:
                n_3.append(int(n[i] / 8 + (3 * N + 6) / 8))
            elif rem8[i] == 3:
                n_3.append(int(n[i] / 8 + (5 * N + 5) / 8))
            elif rem8[i] == 4:
                n_3.append(int(n[i] / 8 + (1 * N + 4) / 8))
            elif rem8[i] == 5:
                n_3.append(int(n[i] / 8 + (6 * N + 3) / 8))
            elif rem8[i] == 6:
                n_3.append(int(n[i] / 8 + (2 * N + 2) / 8))
            elif rem8[i] == 7:
                n_3.append(int(n[i] / 8 + (4 * N + 1) / 8))

            else:
                print("Error!")
        n_4 = []
        rem16 = [i % 16 for i in n]
        for i in range(N):
            if rem16[i] == 0:
                n_4.append(int(n[i] / 16))
            elif rem16[i] == 1:
                n_4.append(int(n[i] / 16 + (15 * N + 15) / 16))
            elif rem16[i] == 2:
                n_4.append(int(n[i] / 16 + (7 * N + 14) / 16))
            elif rem16[i] == 3:
                n_4.append(int(n[i] / 16 + (11 * N + 13) / 16))
            elif rem16[i] == 4:
                n_4.append(int(n[i] / 16 + (3 * N + 12) / 16))
            elif rem16[i] == 5:
                n_4.append(int(n[i] / 16 + (13 * N + 11) / 16))
            elif rem16[i] == 6:
                n_4.append(int(n[i] / 16 + (5 * N + 10) / 16))
            elif rem16[i] == 7:
                n_4.append(int(n[i] / 16 + (9 * N + 9) / 16))
            elif rem16[i] == 8:
                n_4.append(int(n[i] / 16 + (1 * N + 8) / 16))
            elif rem16[i] == 9:
                n_4.append(int(n[i] / 16 + (14 * N + 7) / 16))
            elif rem16[i] == 10:
                n_4.append(int(n[i] / 16 + (6 * N + 6) / 16))
            elif rem16[i] == 11:
                n_4.append(int(n[i] / 16 + (10 * N + 5) / 16))
            elif rem16[i] == 12:
                n_4.append(int(n[i] / 16 + (2 * N + 4) / 16))
            elif rem16[i] == 13:
                n_4.append(int(n[i] / 16 + (12 * N + 3) / 16))
            elif rem16[i] == 14:
                n_4.append(int(n[i] / 16 + (4 * N + 2) / 16))
            elif rem16[i] == 15:
                n_4.append(int(n[i] / 16 + (8 * N + 1) / 16))

            else:
                print("Error!")

        n_5 = []
        rem32 = [i % 32 for i in n]
        for i in range(N):
            if rem32[i] == 0:
                n_5.append(int(n[i] / 32))
            elif rem32[i] == 1:
                n_5.append(int(n[i] / 32 + (31 * N + 31) / 32))
            elif rem32[i] == 2:
                n_5.append(int(n[i] / 32 + (15 * N + 30) / 32))
            elif rem32[i] == 3:
                n_5.append(int(n[i] / 32 + (23 * N + 29) / 32))
            elif rem32[i] == 4:
                n_5.append(int(n[i] / 32 + (7 * N + 28) / 32))
            elif rem32[i] == 5:
                n_5.append(int(n[i] / 32 + (27 * N + 27) / 32))
            elif rem32[i] == 6:
                n_5.append(int(n[i] / 32 + (11 * N + 26) / 32))
            elif rem32[i] == 7:
                n_5.append(int(n[i] / 32 + (19 * N + 25) / 32))
            elif rem32[i] == 8:
                n_5.append(int(n[i] / 32 + (3 * N + 24) / 32))
            elif rem32[i] == 9:
                n_5.append(int(n[i] / 32 + (29 * N + 23) / 32))
            elif rem32[i] == 10:
                n_5.append(int(n[i] / 32 + (13 * N + 22) / 32))
            elif rem32[i] == 11:
                n_5.append(int(n[i] / 32 + (21 * N + 21) / 32))
            elif rem32[i] == 12:
                n_5.append(int(n[i] / 32 + (5 * N + 20) / 32))
            elif rem32[i] == 13:
                n_5.append(int(n[i] / 32 + (25 * N + 19) / 32))
            elif rem32[i] == 14:
                n_5.append(int(n[i] / 32 + (9 * N + 18) / 32))
            elif rem32[i] == 15:
                n_5.append(int(n[i] / 32 + (17 * N + 17) / 32))
            elif rem32[i] == 16:
                n_5.append(int(n[i] / 32 + (1 * N + 16) / 32))
            elif rem32[i] == 17:
                n_5.append(int(n[i] / 32 + (30 * N + 15) / 32))
            elif rem32[i] == 18:
                n_5.append(int(n[i] / 32 + (14 * N + 14) / 32))
            elif rem32[i] == 19:
                n_5.append(int(n[i] / 32 + (22 * N + 13) / 32))
            elif rem32[i] == 20:
                n_5.append(int(n[i] / 32 + (6 * N + 12) / 32))
            elif rem32[i] == 21:
                n_5.append(int(n[i] / 32 + (26 * N + 11) / 32))
            elif rem32[i] == 22:
                n_5.append(int(n[i] / 32 + (10 * N + 10) / 32))
            elif rem32[i] == 23:
                n_5.append(int(n[i] / 32 + (18 * N + 9) / 32))
            elif rem32[i] == 24:
                n_5.append(int(n[i] / 32 + (2 * N + 8) / 32))
            elif rem32[i] == 25:
                n_5.append(int(n[i] / 32 + (28 * N + 7) / 32))
            elif rem32[i] == 26:
                n_5.append(int(n[i] / 32 + (12 * N + 6) / 32))
            elif rem32[i] == 27:
                n_5.append(int(n[i] / 32 + (20 * N + 5) / 32))
            elif rem32[i] == 28:
                n_5.append(int(n[i] / 32 + (4 * N + 4) / 32))
            elif rem32[i] == 29:
                n_5.append(int(n[i] / 32 + (24 * N + 3) / 32))
            elif rem32[i] == 30:
                n_5.append(int(n[i] / 32 + (8 * N + 2) / 32))
            elif rem32[i] == 31:
                n_5.append(int(n[i] / 32 + (16 * N + 1) / 32))
            else:
                print("Error!")

        if layer == 2:
            return [i - 1 for i in n_2]
        if layer == 3:
            return [i - 1 for i in n_3]
        if layer == 4:
            return [i - 1 for i in n_4]
        if layer == 5:
            return [i - 1 for i in n_5]

    def forward(self, x, attn_mask=None):

        # x [B, L, D] torch.Size([16, 336, 512])

        det = []  # List of averaged pooled details
        input = [x, ]
        for l in self.level_layers:

            x_even_update, x_odd_update = l(input[0])

            if self.level_part[self.count_levels][0]:
                input.append(x_even_update)
            else:
                x_even_update = x_even_update.permute(0, 2, 1)
                det += [x_even_update]  ##############################################################################
            if self.level_part[self.count_levels][1]:
                x_odd_update = x_odd_update.permute(0, 2, 1)
                input.append(x_odd_update)
            else:
                det += [x_odd_update]  ##############################################################################
            del input[0]

            self.count_levels = self.count_levels + 1

        for aprox in input:
            aprox = aprox.permute(0, 2, 1)  # b 77 1
            # aprox = self.avgpool(aprox) ##############################################################################
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients

        x = torch.cat(det, 2)  # torch.Size([32, 307, 12])
        index = self.reOrder(x.shape[2], layer=self.layers)
        x_reorder = [x[:, :, i].unsqueeze(2) for i in index]

        x_reorder = torch.cat(x_reorder, 2)

        x = x_reorder.permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)  # torch.Size([16, 512, 336])

        return x


class SeqInterPrediction(nn.Module):
    def __init__(self, args, seq_mask_range, pred_len, in_planes, number_levels, number_level_part, num_layers=3):
        super(SeqInterPrediction, self).__init__()
        self.Left_module = EncoderTree(
            [
                LevelIDCN(args=args, in_planes=in_planes,
                          lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                          share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

                for l in range(number_levels)
            ],

            level_parts=number_level_part,
            num_layers=num_layers,
            Encoder=True
        )

        self.Right_module = EncoderTree(
            [
                LevelIDCN(args=args, in_planes=in_planes,
                          lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
                          share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)

                for l in range(number_levels)
            ],

            level_parts=number_level_part,
            num_layers=num_layers,
            Encoder=True
        )

        self.seq_mask_range = seq_mask_range
        self.Mid_pred_len = pred_len[0]
        self.Edge_pred_len = pred_len[1]

        self.projectionLeft = nn.Conv1d(2*pred_len[0], pred_len[0],
                                    kernel_size=1, stride=1, bias=False)

        self.projectionRight = nn.Conv1d(2*pred_len[0], pred_len[0],
                                     kernel_size=1, stride=1, bias=False)

        self.projectionHead = nn.Conv1d(2*pred_len[1], pred_len[1],
                                        kernel_size=1, stride=1, bias=False)

        self.projectionTail = nn.Conv1d(2*pred_len[1], pred_len[1],
                                         kernel_size=1, stride=1, bias=False)

    def locate_mask_index(self, mask):
        mask_numpy = mask.detach().cpu().numpy()
        mask_index = np.argwhere(mask_numpy[0,:,0] ==True)
        length = mask_index.shape[0]
        if length == 0:
            return None, None, 4
        mask_mid = int(mask_index[int(length/2)])
        type = 0
        if mask_mid<length:
            type = 0
        elif mask_mid>length and mask_mid< 2*length:
            type = 1
        elif mask_mid>2*length and mask_mid< 3*length:
            type = 2
        else:
            type = 3
        return mask_index, mask_mid, type

    def forward(self, x, mask):
        b,l,c = x.shape
        x_temp = x.detach().cpu().numpy()
        x_pred = torch.zeros(b, l, c, device=x.device)
        x_pad = torch.zeros(b, self.Mid_pred_len+l+self.Mid_pred_len, c, device=x.device)
        mask_index, mask_mid, pred_type = self.locate_mask_index(mask)

        if pred_type == 0 or pred_type == 1:
            mask_len = mask_index.shape[0]
            x_index = [int(mask_index[-1])+1, int(mask_index[-1] + 2 * self.Edge_pred_len + 1)]
            x_right = x[:,int(mask_index[-1])+1:int(mask_index[-1] + 2 * self.Edge_pred_len + 1), :]
            x_right_temp = x_right.detach().cpu().numpy()
            x_right_flip = torch.flip(x_right, dims=[1])

            x_right_flip_temp = x_right_flip.detach().cpu().numpy()
            res_right = x_right_flip
            x_right_flip = self.Right_module(x_right_flip)
            x_right_flip += res_right
            x_right_pred = self.projectionRight(x_right_flip)
            x_right_flip_proj_temp = x_right_pred.detach().cpu().numpy()
            # x_right_pred = x_right_flip[:, 0:mask_len, :]
            x_right_pred = torch.flip(x_right_pred, dims=[1])
            x_right_pred_temp = x_right_pred.detach().cpu().numpy()
            x_pred[:, int(mask_index[0]):int(mask_index[-1]) + 1, :] = x_right_pred
            x_pred_temp = x_pred.detach().cpu().numpy()

        elif pred_type == 2 or pred_type == 3:
            mask_len = mask_index.shape[0]
            x_left = x[:, int(mask_index[0])-2*self.Edge_pred_len:int(mask_index[0]), :]
            x_left_temp = x_left.detach().cpu().numpy()
            res_left = x_left
            x_left = self.Left_module(x_left)
            x_left += res_left
            x_left = self.projectionLeft(x_left)
            x_left_pred = x_left[:, 0:mask_len, :]
            x_pred[:, int(mask_index[0]):int(mask_index[-1]) + 1, :] = x_left_pred
            x_pred_temp = x_pred.detach().cpu().numpy()

        else:
            x_pred = x
            return x_pred,0

        return x_pred,1




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SCIMaskEvenPretrain(nn.Module):
    def __init__(self, args, input_len, seq_mask_range_ratio, pred_len, input_dim=9,
                 number_levels=4, number_level_part=[[1, 0], [1, 0], [1, 0]], num_layers=3,
                 no_bootleneck=True):
        super(SCIMaskEvenPretrain, self).__init__()

        self.seq_mask_range_ratio = seq_mask_range_ratio
        self.seq_mask_range = [int(input_len/seq_mask_range_ratio[0]), int(input_len/seq_mask_range_ratio[1])]


        in_planes = input_dim
        out_planes = input_dim * (number_levels + 1)

        # self.blocks = nn.ModuleList([EncoderTree(
        #     [
        #         LevelIDCN(args=args, in_planes=in_planes,
        #                   lifting_size=[2, 1], kernel_size=4, no_bottleneck=True,
        #                   share_weights=False, simple_lifting=False, regu_details=0.01, regu_approx=0.01)
        #
        #         for l in range(number_levels)
        #     ],
        #
        #     level_parts=number_level_part,
        #     num_layers=num_layers,
        #     Encoder=True
        # ) for i in range(6)])



        self.seqPredicts = SeqInterPrediction(args=args, seq_mask_range =self.seq_mask_range,
                                                             pred_len = pred_len, in_planes=in_planes, number_levels = number_levels,
                                             number_level_part= number_level_part, num_layers=num_layers)


        self.linearFusion = nn.Conv2d(in_channels = 3, out_channels =1, kernel_size = 1, stride=1,padding=0, dilation=1, groups=1,bias=True)
        self.error = nn.L1Loss()

        if no_bootleneck:
            in_planes *= 1

        self.num_planes = out_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                m.bias.data.zero_()

    def creatSeqEvenMask(self, x, part = 4):
        b, l, c = x.shape
        PartMask = []
        MaskX = []
        seg_len = int(l / part)
        for i in range(part):
            Mask = torch.ones(b, l, c, device=x.device)
            index =  [i*seg_len,(i+1)*seg_len]
            Mask[:, i*seg_len:(i+1)*seg_len, :] = 0

            Mask = (Mask == 0)
            Mask_temp = Mask.detach().cpu().numpy()

            PartMask.append(Mask)
            MaskX.append(x.masked_fill(Mask, 0))



        rand_selection = list(range(0, len(MaskX)))
        random.shuffle(rand_selection)
        return MaskX, PartMask, rand_selection



    def forward(self, x):

        b, l, c = x.shape

        seq_rand_processed_x = torch.zeros(b, l, c, device=x.device)
        seq_rand_processed_x_1 = torch.zeros(b, l, c, device=x.device)


        seq_rand_mask_x, seq_rand_mask, rand_selection = self.creatSeqEvenMask(x, part=self.seq_mask_range_ratio[0])


        for i in rand_selection:
            seq_mask_i_x_temp = seq_rand_mask_x[i].detach().cpu().numpy()
            seq_rand_mask_temp = seq_rand_mask[i].detach().cpu().numpy()
            mask_x_process,_ = self.seqPredicts(seq_rand_mask_x[i], seq_rand_mask[i])
            mask_x_process_temp = mask_x_process.detach().cpu().numpy()
            seq_rand_processed_x = seq_rand_processed_x + mask_x_process
            seq_processed_x_temp = seq_rand_processed_x.detach().cpu().numpy()

        seq_rand_mask_x, seq_rand_mask, rand_selection = self.creatSeqEvenMask(seq_rand_processed_x, part=self.seq_mask_range_ratio[0])


        # for j in rand_selection:
        #     seq_mask_i_x_temp = seq_rand_mask_x[j].detach().cpu().numpy()
        #     seq_rand_mask_temp = seq_rand_mask[j].detach().cpu().numpy()
        #     mask_x_process,_ = self.seqPredicts(seq_rand_mask_x[j], seq_rand_mask[j])
        #     mask_x_process_temp = mask_x_process.detach().cpu().numpy()
        #     seq_rand_processed_x_1 = seq_rand_processed_x_1 + mask_x_process
        #     seq_processed_x_temp = seq_rand_processed_x_1.detach().cpu().numpy()
        #
        # seq_rand_processed  = (seq_rand_processed_x+seq_rand_processed_x_1)/2
        return seq_rand_processed_x


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


if __name__ == '__main__':
    torch.manual_seed(321)  # reproducible
    torch.cuda.manual_seed_all(321)


    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.5)

    # Action Part

    parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
    parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
    parser.add_argument('--hidden-size', default=3, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)
    parser.add_argument('--groups', type=int, default=1)
    args = parser.parse_args()
    part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
    # part = [[1, 1], [0, 0], [0, 0]]
    # part = [ [0, 0]]

    print('level number {}, level details: {}'.format(len(part), part))     #mid end
    model = SCIMaskEvenPretrain(args, input_len=128, seq_mask_range_ratio = [4,4], pred_len = [32, 32],  input_dim=2,
                   number_levels=len(part),
                   number_level_part=part, num_layers=3).cuda()
    x = torch.randn(1, 128, 2).cuda()
    y = model(x)
    print(y.shape)
