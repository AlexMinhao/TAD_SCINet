import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random

class GRU(nn.Module):
    def __init__(self, hidC, hidR):
        super(GRU, self).__init__()
        # self.use_cuda = args.cuda

        self.hidR = hidR
        self.hidC = hidC

        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.GRU2 = nn.GRU(self.hidR, self.hidC)
        self.lstm1 = nn.LSTM(self.hidC, self.hidR, num_layers=2)
        self.lstm2 = nn.LSTM(self.hidR, self.hidC, num_layers=2)
        self.dropout = nn.Dropout(0.2)

    def creatMask(self, x):
        b, l, c = x.shape
        mask_ratio = nn.Dropout(p=0.8)
        Mask = torch.ones(b, l, c, device=x.device)
        Mask = mask_ratio(Mask)
        Mask = Mask > 0  # torch.Size([8, 1, 48, 48])
        Mask = Mask
        x.masked_fill_(Mask, 0)
        return x

    def creatMaskEvenSplit(self, x, part=6):
        b, l, c = x.shape
        blist = list(range(0, b))
        llist = list(range(0, l))
        clist = list(range(0, c))
        index = []
        for b_ind in blist:
            for l_ind in llist:
                for c_ind in clist:
                    index.append([b_ind, l_ind, c_ind])

        slice_num = int(b * l * c / part)
        PartMask = []
        MaskX = []
        for i in range(part):
            slice = random.sample(index, slice_num)
            Mask = torch.ones(b, l, c, device=x.device)
            for s in slice:
                Mask[s[0], s[1], s[2]] = 0
                index.remove(s)
            Mask = (Mask == 0)
            Mask_temp = Mask.detach().cpu().numpy()

            PartMask.append(Mask)
            mask_x_temp = x.masked_fill(Mask, 0)
            mask_x_temp = mask_x_temp.detach().cpu().numpy()
            MaskX.append(x.masked_fill(Mask, 0))

        return MaskX, PartMask

    def creatMaskSeqMask(self, x, part=6):
        b, l, c = x.shape
        part_len = int(l / part)
        PartMask = []
        MaskX = []
        for i in range(part):
            Mask = torch.ones(b, l, c, device=x.device)
            Mask[:, i * part_len:(i + 1) * part_len, :] = 0

            Mask = (Mask == 0)
            Mask_temp = Mask.detach().cpu().numpy()

            PartMask.append(Mask)
            mask_x_temp = x.masked_fill(Mask, 0)
            mask_x_temp = mask_x_temp.detach().cpu().numpy()
            MaskX.append(x.masked_fill(Mask, 0))

        return MaskX, PartMask

    def forward(self, x):

        b, l, c = x.shape
        point_processed_x = torch.zeros(b, l, c, device=x.device)
        seq_processed_x = torch.zeros(b, l, c, device=x.device)
        point_mask_x, point_mask = self.creatMaskEvenSplit(x, part=6)
        seq_mask_x, seq_mask = self.creatMaskSeqMask(x, part=6)

        for i in range(6):
            x_i = point_mask_x[i]
            mask_x_process, _ = self.GRU1(x_i)
            mask_x_process, _ = self.GRU2(mask_x_process)
            unmask = (point_mask[i] == False)
            mask_x_process = mask_x_process.masked_fill(unmask, 0)
            point_processed_x = point_processed_x + mask_x_process



        return point_processed_x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    # parser.add_argument('--data', type=str, required=True,
    #                     help='location of the data file')
    parser.add_argument('--model', type=str, default='LSTNet',
                        help='')
    parser.add_argument('--hidCNN', type=int, default=100, # 32 64 128
                        help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100,  #64 128 256
                        help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=12,
                        help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=3,
                        help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=6,
                        help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--skip', type=float, default=2)
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--input_dim', default=170, type=int)
    parser.add_argument('--window_size', default=12, type=int)
    args = parser.parse_args()


    model = Model(args)
    x = torch.randn(64, 12, 170)
    y= model(x)
    print(y.shape)