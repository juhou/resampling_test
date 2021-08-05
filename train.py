import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from utils.util import *
# import apex
# from apex import amp
from dataset import get_dataframe, get_transforms, resamplingDataset
from models import JUNet, MISLnet

Precautions_msg = '(주의사항) ---- \n'


'''
- train.py

모델을 학습하는 전과정을 담은 코드

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> train.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

*** def parse_args(): 실행 파라미터에 대한 모든 정보가 있다.  
*** def run(): 학습의 모든과정이 담긴 함수. 이곳에 다양한 trick을 적용하여 성능을 높혀보자. 
** def main(): fold로 나뉜 데이터를 run 함수에 분배해서 실행
* def train_epoch(), def val_epoch() : 완벽히 이해 후 수정하도록


Training list
python train.py --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0
python predict.py --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--data-dir', type=str, default='./data/')
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='256')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--enet-type', type=str, required=True)
    # 학습에 적용할 네트워크 이름
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--use-meta', action='store_true')
    # meta데이터 (사진 외의 나이, 성별 등)을 사용할지 여부

    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    # meta데이터 사용 시 중간레이어 사이즈

    parser.add_argument('--out-dim', type=int, default=2)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')
    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    parser.add_argument('--k-fold', type=int, default=5)
    # data cross-validation
    # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/

    parser.add_argument('--accumulation-step', type=int, default=1)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # 원본데이터에 추가로 외부 데이터를 사용할지 여부


    parser.add_argument('--batch-size', type=int, default=32) # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=8) # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=4e-6) # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값
    parser.add_argument('--n-epochs', type=int, default=100) # epoch 수

    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):

        optimizer.zero_grad()
        data, target = data.to(device), torch.squeeze(target.to(device))
        logits = model(data)

        loss = criterion(logits, target)
        loss.backward()

        # # 그라디언트가 너무 크면 값을 0.5로 잘라준다 (max_grad_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # gradient accumulation (메모리 부족할때)
        if args.accumulation_step:
            if (i + 1) % args.accumulation_step == 0:
                optimizer.step()
                #optimizer.zero_grad()
        else:
            optimizer.step()
            #optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def val_epoch(model, loader):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    val_r_loss = []
    val_s_loss = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            data, target = data.to(device), torch.squeeze(target.to(device))
            logits = model(data)

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

            r_loss = criterion(logits[:, 0], target[:, 0])
            val_r_loss.append(r_loss.detach().cpu().numpy())

            s_loss = criterion(logits[:, 1], target[:, 1])
            val_s_loss.append(s_loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    val_r_loss = np.mean(val_r_loss)
    val_s_loss = np.mean(val_s_loss)
    return val_loss, val_r_loss, val_s_loss

def run(fold, df, transforms_train, transforms_val):
    '''
    학습 진행 메인 함수
    :param fold: cross-validation에서 valid에 쓰일 분할 번호
    :param df: DataFrame 학습용 전체 데이터 목록
    :param transforms_train, transforms_val: 데이터셋 transform 함수
    '''

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    if args.k_fold == 1:
        df_train = df_valid

        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
        if len(df_train) % args.batch_size == 1:
            df_train = df_train.sample(len(df_train)-1)
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)

    # 데이터셋 읽어오기
    dataset_train = resamplingDataset(df_train, 'train', args.image_size, transform=transforms_train)
    dataset_valid = resamplingDataset(df_valid, 'valid', args.image_size, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)


    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    if os.path.isfile(model_file3):
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            pretrained=True,
            im_size = args.image_size
        )
        model.load_state_dict(torch.load(model_file))
    else:
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            pretrained=True,
            im_size=args.image_size
        )

    # GPU 여러개로 병렬처리
    # if DP:
    #     model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    val_loss_max = 99999.

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # if args.use_amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    # amp를 사용하면 버그 (use_amp 비활성화)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer)

        if args.k_fold == 1 :
            # skip validation
            val_loss, acc, auc, auc_no_ext = [999.0, 0.0, 0.0, 0.0]
            if epoch + 5 > args.n_epochs :
                model_file4 = os.path.join(args.model_dir, f'{args.kernel_type}_fold{fold}_e{args.n_epochs-epoch}.pth')
                torch.save(model.state_dict(), model_file4)
        else:
            val_loss, val_r_loss, val_s_loss = val_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid r loss: {(val_r_loss):.5f}, valid s loss: {(val_s_loss):.5f}'

        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step() # bug workaround

        if val_loss < val_loss_max:
            print('val_loss_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss))
            torch.save(model.state_dict(), model_file)
            val_loss_max = val_loss

    if args.k_fold ==1 :
        torch.save(model.state_dict(), model_file)
    torch.save(model.state_dict(), model_file3)



def main():
    df_train, df_test = get_dataframe(args.k_fold, args.data_dir, args.data_folder, args.out_dim)

    # 모델 트랜스폼 가져오기
    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = range(args.k_fold)
    for fold in folds:
        run(fold, df_train, transforms_train, transforms_val)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # 네트워크 타입 설정
    if 'JUNet' in args.enet_type:
        ModelClass = JUNet
    elif 'MISLnet' in args.enet_type:
        ModelClass = MISLnet
    else:
        raise NotImplementedError()

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(2359)
    device = torch.device('cuda')
    criterion = nn.MSELoss()

    # 메인 기능 수행
    main()