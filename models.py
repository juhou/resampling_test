import torch, torch.nn as nn, torch.nn.functional as F, random
import torch.fft as fft
import cv2

#
# class ModelBase(nn.Module):
#     def __init__(self, name):
#         super(ModelBase, self).__init__()
#         self.name = name
#
#     def copy_params(self, state_dict):
#         own_state = self.state_dict()
#         for (name, param) in state_dict.items():
#             if name in own_state:
#                 own_state[name].copy_(param.clone())
#
#     def boost_params(self, scale=1.0):
#         if scale == 1.0:
#             return self.state_dict()
#         for (name, param) in self.state_dict().items():
#             self.state_dict()[name].copy_((scale * param).clone())
#         return self.state_dict()
#
#     # self - x
#     def sub_params(self, x):
#         own_state = self.state_dict()
#         for (name, param) in x.items():
#             if name in own_state:
#                 own_state[name].copy_(own_state[name] - param)
#
#     # self + x
#     def add_params(self, x):
#         a = self.state_dict()
#         for (name, param) in x.items():
#             if name in a:
#                 a[name].copy_(a[name] + param)






class MISLnet(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained = False):
        super(MISLnet, self).__init__()

        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[3, 1, 5, 5]), requires_grad=True)
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=4)
        self.conv2 = nn.Conv2d(96, 64, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 1, stride=1)

        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, out_dim)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(3):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0

    def forward(self, x):
        # Constrained-CNN
        self.normalized_F()
        x = F.conv2d(x, self.const_weight)
        # CNN
        x = self.conv1(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv2(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv3(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv4(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv5(x)
        x = self.avg_pool(torch.tanh(x))

        # Fully Connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        logist = self.fc3(x)

        return logist







class JUNet(nn.Module):

    def __init__(self, enet_type, im_size, out_dim=2, pretrained = False):
        super(JUNet, self).__init__()

        self.register_parameter("const_weight", None)
        self.num_const_w = 16
        self.const_weight = nn.Parameter(torch.randn(size=[self.num_const_w, 1, 5, 5]), requires_grad=True)
        self.conv1 = nn.Conv2d(self.num_const_w, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, 1, stride=1, padding=1)

        # layers after fft
        self.row_fc1 = nn.Linear(im_size-2, im_size-2)
        self.row_fc2 = nn.Linear((im_size-2), 64)

        self.col_fc1 = nn.Linear(im_size-2, im_size-2)
        self.col_fc2 = nn.Linear((im_size-2), 64)

        # parameter estimator
        self.regressor_fc1 = nn.Linear(128, 64)
        self.regressor_fc2 = nn.Linear(64, out_dim)

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(self.num_const_w):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0

    def forward(self, x):
        # Constrained-CNN
        self.normalized_F()
        x = F.conv2d(x, self.const_weight)

        # CNN - 2d local area feature extractor
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        # fft layer
        fft_row = torch.abs(fft.fft(x, dim=2))
        mean_fft_row = torch.squeeze(torch.mean(fft_row, dim=3))

        fft_col = torch.abs(fft.fft(x, dim=3))
        mean_fft_col = torch.squeeze(torch.mean(fft_col, dim=2))


        x_row = self.row_fc1(mean_fft_row)
        x_row = torch.relu(x_row)
        x_row = self.row_fc2(x_row)
        x_row = torch.relu(x_row)

        x_col = self.col_fc1(mean_fft_col)
        x_col = torch.relu(x_col)
        x_col = self.col_fc2(x_col)
        x_col = torch.relu(x_col)

        # parameter estimator
        x = torch.cat((x_row, x_col), dim=1)
        x = self.regressor_fc1(x)
        x = torch.relu(x)
        logist = self.regressor_fc2(x)

        return logist