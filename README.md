# Image resampling detection and estimation을 위한 baseline


## SOFTWARE 

Pytorch 1.8 이상 (*fft module)

CUDA Version 11.1

(`requirements.txt`안의 python package 상세 참조)

1. Nvidia 드라이버, CUDA toolkit 11.1, Anaconda 설치

2. pytorch 설치

3. 각종 필요 패키지 설치

        pip install opencv-contrib-python albumentations pillow scikit-learn scikit-image pandas tqdm 

4. apex 설치

        conda install -c conda-forge nvidia-apex
        
5. git, LR 관련 패키지 설치

        conda install git
        pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git


## DATA SETUP 

`./data` 하위 경로를 기준으로 `DB명/{test,train,test.csv,train.csv}`

아래는 Database의 샘플 구조이며, 접미사 ext는 외부데이터 (크롤링 혹은 생성한)를 뜻한다. 

IMD2020_prnu 데이터셋의 canon-powershot-a495 이미지 100장을 이용함



```
./data/images/test/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/images/train/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/images/test.csv
./data/images/train.csv

```

아래는 샘플 csv 구조

```
,image_name,camera_name,img_id
0,img_001.jpg,canon-powershot-a495,0
1,img_002.jpg,canon-powershot-a495,1
2,img_003.jpg,canon-powershot-a495,2
3,img_004.jpg,canon-powershot-a495,3
4,img_005.jpg,canon-powershot-a495,4
5,img_006.jpg,canon-powershot-a495,5
6,img_007.jpg,canon-powershot-a495,6
7,img_008.jpg,canon-powershot-a495,7
8,img_009.jpg,canon-powershot-a495,8
9,img_010.jpg,canon-powershot-a495,9
10,img_011.jpg,canon-powershot-a495,10

...

```



## Training



Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행

        python train.py --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0
        python predict.py --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0

pycharm의 경우: 

        Run 메뉴
        -> Edit Configuration 
        -> train.py 가 선택되었는지 확인 
        -> parameters 이동 후 아래를 입력 

        --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0

        -> 적용하기 후 실행/디버깅



## (Optional) Evaluating

학습한 모델을 k-fold cross validation 진행한다. 앞에서 학습에 사용한 모델을 사용하거나, `--model-dir`안에 있는 모델을 명시하여 평가할 수 있다. 

평가 결과는 `./logs/`에 저장되며, 전체 합쳐진 결과 (Out-of-folds)는  `./oofs/` 폴더에 저장된다. 


## Predicting

Test 데이터에 따른 예측을 실시한다. 

각 모델의 평가 결과는 아래 `./subs/` 폴더에 저장된다. 


## Ensembling (앙상블)

최종 평가 결과를 위해, 18개의 모델의 예측 결과를 모아서 최종 예측을 진행한다. 

root 폴더에 `final_sub1.csv`라는 형태로 출력을 만들어준다. 

```
python ensemble.py
```

