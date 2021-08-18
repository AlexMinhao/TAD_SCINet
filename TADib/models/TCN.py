
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import argparse
import random

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1,  self.relu1, self.dropout1,
        #                          self.conv2,  self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # out1 = self.conv1(x)
        out = self.net(x)
        # print('TemporalBlock')
        # temp = out.detach().cpu().numpy()
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    #self.net(x)
    # Sequential(
    #     (0): Conv1d(2, 30, kernel_size=(7,), stride=(1,), padding=(6,))
    # (1): Chomp1d()
    # (2): ReLU()
    # (3): Dropout(p=0.0, inplace=False)
    # (4): Conv1d(30, 30, kernel_size=(7,), stride=(1,), padding=(6,))
    # (5): Chomp1d()
    # (6): ReLU()
    # (7): Dropout(p=0.0, inplace=False)
    # )
    #self.downsample(x)
    # Conv1d(2, 30, kernel_size=(1,), stride=(1,))





class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) #[30, 30, 30, 30, 30, 30, 30, 30]
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_len, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], input_size)
        self.init_weights()
        self.output_len = output_len

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
    def creatMask(self, x):
        b, l, c = x.shape
        mask_ratio = nn.Dropout(p=0.8)
        Mask = torch.ones(b, l, c, device=x.device)
        Mask = mask_ratio(Mask)
        Mask = Mask > 0  # torch.Size([8, 1, 48, 48])
        Mask = Mask
        x.masked_fill_(Mask, 0)
        return x

    def creatMaskEvenSplit(self, x, part = 6):
        b, l, c = x.shape
        blist = list(range(0, b))
        llist = list(range(0, l))
        clist = list(range(0, c))
        index = []
        for b_ind in blist:
            for l_ind in llist:
                for c_ind in clist:
                    index.append([b_ind,l_ind,c_ind])


        slice_num = int(b*l*c/part)
        PartMask = []
        MaskX = []
        for i in range(part):
            slice = random.sample(index, slice_num)
            Mask = torch.ones(b, l, c, device=x.device)
            for s in slice:
                Mask[s[0],s[1],s[2]] = 0
                index.remove(s)
            Mask = (Mask == 0)
            Mask_temp = Mask.detach().cpu().numpy()

            PartMask.append(Mask)
            mask_x_temp = x.masked_fill(Mask, 0)
            mask_x_temp = mask_x_temp.detach().cpu().numpy()
            MaskX.append(x.masked_fill(Mask, 0))

        return MaskX, PartMask


    def creatMaskSeqMask(self, x, part = 6):
        b, l, c = x.shape
        part_len = int(l/part)
        PartMask = []
        MaskX = []
        for i in range(part):
            Mask = torch.ones(b, l, c, device=x.device)
            Mask[:,i*part_len:(i+1)*part_len,:] = 0

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
            x_i = point_mask_x[i].permute(0, 2, 1)
            mask_x_process = self.tcn(x_i)  # torch.Size([32, 400, 2])  #y1 = torch.Size([32, 30, 400])
            mask_x_process = mask_x_process.permute(0, 2, 1)
            mask_x_process = self.linear(mask_x_process)
            unmask = (point_mask[i] == False)
            mask_x_process = mask_x_process.masked_fill(unmask, 0)
            point_processed_x = point_processed_x + mask_x_process


        #
        #
        # for i in range(6):
        #
        #     mask_x_process = self.tcn(seq_mask_x[i])
        #     mask_x_process = mask_x_process.permute(0, 2, 1)
        #     mask_x_process = self.linear(mask_x_process)
        #     unmask = (seq_mask[i] == False)
        #     mask_x_process = mask_x_process.masked_fill(unmask, 0)
        #     seq_processed_x = seq_processed_x + mask_x_process

        return point_processed_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (default: 0.0)')

    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='sequence length (default: 400)')
    parser.add_argument('--nhid', type=int, default=30,
                        help='number of hidden units per layer (default: 30)')
    parser.add_argument('--horizon', type=int, default=12)


    parser.add_argument('--window_size', type=int, default=400)
    parser.add_argument('--share-weight', default=0, type=int, help='share weight or not in attention q,k,v')
    parser.add_argument('--temp', default=0, type=int, help='Use temporature weights or not, if false, temp=1')
    parser.add_argument('--hidden-size', default=15, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=7, type=int, help='kernel size')
    parser.add_argument('--input_dim', default=170, type=int)

    args = parser.parse_args()

    channel_sizes = [args.nhid] * args.levels
    model = TCN(args.input_dim, args.horizon, channel_sizes, kernel_size=args.kernel, dropout=args.dropout)

    x = torch.randn(32, 12, 170)

    y = model(x)
    print(y.shape)