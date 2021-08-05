import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def get_dataframe(k_fold, data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'train.csv'))

    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}train', x))  # f'{x}.jpg'

    # 원본데이터=0, 외부데이터=1
    df_train['is_ext'] = 0

    '''
    ####################################################
    교차 검증 구현 (k-fold cross-validation)
    ####################################################
    '''
    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    img_ids = len(df_train['img_id'].unique())
    print(f'Original dataset의 이미지수 : {img_ids}')

    # 데이터 인덱스 : fold 번호. (fold)번 분할뭉치로 간다
    # train.py input arg에서 k-fold를 수정해줘야함 (default:5)
    print(f'Dataset: {k_fold}-fold cross-validation')
    img_id2fold = {i: i % k_fold for i in range(img_ids)}
    df_train['fold'] = df_train['img_id'].map(img_id2fold)

    # test data (학습이랑 똑같게 함)
    df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    # 쓰지 않는 변수
    meta_features = None
    n_meta_features = 0
    target_idx = 1

    return df_train, df_test



class resamplingDataset(Dataset):
    def __init__(self, csv, mode, image_size=1024, transform=None):
        self.csv = pd.concat([csv]*10, ignore_index=True).reset_index(drop=True)
        self.mode = mode # train / valid
        self.transform = transform

        self.r_test_delta = 180.0 # 테스트가 필요없다면 180, 아니면 1
        self.s_test_delta = 0.01 # 0.01 단위로 변경
        self.s_test_low = 0.5
        self.s_test_high = 2.0
        self.image_size = image_size

        # 시간 절약을 위해 메모리에 데이터셋 미리 읽어들임
        self.image_list = []
        print('데이터 로딩 시작')
        for i in tqdm(range(len(self.csv))):
            # 이미지 읽어들임
            temp_img = cv2.cvtColor(cv2.imread(self.csv.iloc[i].filepath), cv2.COLOR_RGB2BGR)

            # Color filter array 영향을 줄이기 위해 0.5배로 리사이징
            temp_img = cv2.resize(temp_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            # center crop
            h, w, _ = temp_img.shape
            h_half, w_half = int(h/2), int(w/2)
            temp_img = temp_img[h_half-self.image_size:h_half+self.image_size, w_half-self.image_size:w_half+self.image_size, :]

            # append image list
            self.image_list.append(temp_img)
        print('데이터 로딩 끝')

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        image = self.image_list[index]
        
        # 변형 값 얻어내기
        r_test_value = np.random.uniform(low=0, high=180, size=(1,)) // self.r_test_delta
        s_test_value = np.random.uniform(low=self.s_test_low, high=self.s_test_high, size=(1,))
        s_test_value -= (s_test_value % self.s_test_delta)


        # 변형
        matrix = cv2.getRotationMatrix2D((int(self.image_size/2), int(self.image_size/2)), float(r_test_value), float(s_test_value))
        image = cv2.warpAffine(image, matrix, (self.image_size, self.image_size), cv2.INTER_LINEAR)

        # albumentation 적용
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)

        # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        # 변경 값 리턴하기
        target_list = [r_test_value, s_test_value*100]

        return data, torch.tensor(target_list).float()



def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        albumentations.RandomBrightness(limit=0.1, p=0.75),
        albumentations.RandomContrast(limit=0.1, p=0.75),
        # albumentations.CLAHE(clip_limit=2.0, p=0.3),
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        # albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

def get_meta_data_stoneproject(df_train, df_test):
    '''
    ####################################################
                        안씀
    ####################################################
    '''

    return 0,0,0,0

