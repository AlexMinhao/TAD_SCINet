import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import random
import numpy as np
random.seed(80) #80

class SNetDataset(Dataset):
    def __init__(self, raw_data, mode='train', add_anomaly = False, test_slide = None,test_label= None, config=None):
        self.raw_data = raw_data#[:,1].unsqueeze(1)

        self.config = config  # {'slide_win': 15, 'slide_stride': 5}

        self.mode = mode
        if test_slide:
            self.test_slide = test_slide
        else:
            self.test_slide = config['slide_win']

        data = self.raw_data
        # data = x_data

        if mode == 'train':
            labels = [0] * data.shape[0]
            if add_anomaly:
                anomaly_part = 25
                T = config['slide_win']
                a = -1.5
                b = 1.5
                ratio = [T / 8, T / 6]
                start_anomaly = random.sample(range(0, data.shape[0]-2*T), anomaly_part)


                idxs = np.random.randint(0, len(ratio), size=anomaly_part)  # 生成长度为5的随机数组，范围为 [0,10)，作为索引
                end_anomaly =[int(ratio[i]) for i in idxs]  # 按照索引，去l中获取到对应的值


                for i in range(anomaly_part):
                    r1 = torch.from_numpy(a + (b-a)*np.random.random(end_anomaly[i])).float()
                    r2 = torch.from_numpy(a + (b - a) * np.random.random(end_anomaly[i])).float()
                    temp = data[start_anomaly[i]:start_anomaly[i]+end_anomaly[i],0]
                    data[start_anomaly[i]:start_anomaly[i]+end_anomaly[i],0]=data[start_anomaly[i]:start_anomaly[i]+end_anomaly[i],0].mul(r1)
                    data[start_anomaly[i]:start_anomaly[i] + end_anomaly[i], 1] = data[
                                                                                  start_anomaly[i]:start_anomaly[i] +
                                                                                                   end_anomaly[i],
                                                                                  1].mul(r2)
                    labels[start_anomaly[i]:start_anomaly[i]+end_anomaly[i]] = [1] * end_anomaly[i]


        else:


            labels = test_label






        # to tensor float32
        data = torch.tensor(data).double()  # torch.Size([27, 2049])
        labels = torch.tensor(labels).double()  # 2049
        self.raw_data = torch.tensor(self.raw_data).double()

        if add_anomaly:
            self.x, _, self.labels = self.process(data, labels)
            _, self.y, _ = self.process(self.raw_data, labels)
        else:
            self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        is_train = self.mode == 'train'
        if is_train:
            slide_win, slide_stride = [self.config[k] for k
                                       in ['slide_win', 'slide_stride']
                                       ]
        else:
            slide_win, slide_stride = [self.config[k] for k
                                       in ['test_slide_win', 'test_slide_stride']
                                       ]


        total_time_len, node_num = data.shape  # 27  1565    test: 27  1565

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win,
                                                                                     total_time_len, slide_stride)  # range (15 1565 5)  168  3000  168

        for i in rang:
            # 27 1565    10:25
            # print(i)
            ft = data[i - slide_win:i,:]  # 27  15
            tar = ft  # 27
            ft_num = ft.detach().cpu().numpy()
            tar_num = tar.detach().cpu().numpy()
            label = labels[i - slide_win:i]
            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(label)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        # labels = torch.Tensor(labels_arr).contiguous()
        labels = torch.stack(labels_arr).contiguous()
        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()


        label = self.labels[idx].double()

        return feature, y, label





