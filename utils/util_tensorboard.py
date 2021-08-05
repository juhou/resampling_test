# imports
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

'''
Command Line에 다음과 같이 입력하여 tensorboard를 본다
tensorboard --logdir=./runs/{kernel_type}
'''


class util_classification_tensorboard():
    ''' classification tensorboard 기록 전용 클래스 '''

    def __init__(self, kernel_type, classes):
        ''' Constructor
        classes = list [name1, name2, ... , name_n]
            e.g. ['normal', 'stone']
        '''
        self.writer = SummaryWriter(f'./runs/{kernel_type}')
        self.classes = classes

    def __del__(self):
        ''' Destructor '''
        self.writer.close()

    def write_batchsamples(self, batchsample_name, images, labels):
        ''' 이미지 배치 샘플들을 기록 '''
        # 이미지 그리드를 만듬.
        img_grid = torchvision.utils.make_grid(images)
        self.__matplotlib_imshow(img_grid, one_channel=True)
        self.writer.add_image(batchsample_name, img_grid)


    def write_net_graph(self, net):
        ''' 뉴럴넷 모델을 기록함 '''
        self.writer.add_graph(net)

    def write_train_epoch(self, loss, inputs, labels, preds, probs, epoch):
        ''' 뉴럴넷 학습 epoch 단계별 이미지/loss를 기록함
            train_epoch() 안쪽에서 호출하는 목적
        '''

        # 학습 중 손실(running loss)을 기록
        self.writer.add_scalar('train/Loss', loss, epoch)
        self.writer.add_scalar('train/Accuracy', loss, epoch)
        self.writer.add_scalar('train/AUC', loss, epoch)

        # 미니배치(mini-batch)에 대한 예측 결과 Figure를 기록
        self.writer.add_figure('train/predict vs. GT',
                               self.__plot_classes_preds(inputs, labels, preds, probs),
                               global_step=epoch)

        # ROC 그리기
        target_index = 1
        self.write_pr_curve_tensorboard(self, target_index, preds, probs, global_step=epoch)

    def write_pr_curve_tensorboard(self, target_index, test_preds, test_probs, global_step=0):
        ''' target_index에 해당하는 ROC curve를 그림 '''
        tensorboard_preds = test_preds == target_index
        tensorboard_probs = test_probs[:, target_index]

        self.writer.add_pr_curve(self.classes[target_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)



    def __plot_classes_preds(self, images, labels, preds, probs):
        '''
        학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
        Figure를 생성. 신경망의 예측 결과 / 확률과 함께 정답(GT)을 보여주며,
        예측 결과가 맞았는지 여부에 따라 색을 다르게 표시
        '''
        # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            self.__matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig

    def __matplotlib_imshow(img, one_channel=False):
        # 이미지를 보여주기 위한 헬퍼(helper) 함수
        # (아래 `plot_classes_preds` 함수에서 사용)
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

