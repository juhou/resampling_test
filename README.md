# MMC image classification base-kernel

MMC 연구실의 image classification 연구실험을 위한 base-kernel. 


## SOFTWARE 


Python 3.6.9 ~ 3.7.9

CUDA Version 10.2.89

cuddn 7.6.5

(`requirements.txt`안의 python package 상세 참조)

1. Nvidia 드라이버, CUDA toolkit 10.2, Anaconda 설치

2. pytorch 설치

        conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

3. 각종 필요 패키지 설치

        pip install opencv-contrib-python kaggle resnest geffnet albumentations pillow scikit-learn scikit-image pandas tqdm pretrainedmodels

4. apex 설치

        conda install -c conda-forge nvidia-apex
        
5. git, LR 관련 패키지 설치

        conda install git
        pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git


## DATA SETUP 

`./data` 하위 경로를 기준으로 `DB명/{test,train,test.csv,train.csv}`

아래는 Database의 샘플 구조이며, 접미사 ext는 외부데이터 (크롤링 혹은 생성한)를 뜻한다. 


```
./data/database01/test/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/database01/train/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/database01/test.csv
./data/database01/train.csv

./data/database_ext01/train/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/database_ext01/train.csv

./data/database_ext02/train/{im1.jpg, im2.jpg, ... ,imn.jpg}
./data/database_ext02/train.csv
```

아래는 샘플 csv 구조

```
,image_name,target,patient_id,image_number
3332,51-1.jpg,1,51,1.jpg
3333,51-2.jpg,1,51,2.jpg
3334,51-3.jpg,1,51,3.jpg
3335,51-4.jpg,1,51,4.jpg
3344,52-1.jpg,1,52,1.jpg
3345,52-2.jpg,1,52,2.jpg
3346,52-3.jpg,1,52,3.jpg
3347,52-4.jpg,1,52,4.jpg
...

```



## Training


SIIM-ISIC Melanoma Classification 커널 구조를 기반으로 작성하였다. https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412



Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행

        python train.py --kernel-type 5fold_b3_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 

        Run 메뉴
        -> Edit Configuration 
        -> train.py 가 선택되었는지 확인 
        -> parameters 이동 후 아래를 입력 

        --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b7_ns --n-epochs 30

        -> 적용하기 후 실행/디버깅


아래는 18개 모델에 대한 샘플예제 command이다. 

학습이 진행되면서 폴더 `./weights/` 에, best, final weight가 저장된다. 학습 로그는 `./logs/` 폴더에 저장된다. 

아래는 타 대회의 우승팀이 조합한 18개 모델에 대한 학습 스크립트이다. (https://www.kaggle.com/boliu0/melanoma-winning-models)

```
python train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3  --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns --use-amp --init-lr 2e-5 --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5

python train.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0

python train.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns  --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4 --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns  --n-meta-dim 128,32 --use-amp --CUDA_VISIBLE_DEVICES 0

python train.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7

python train.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns  --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101 --init-lr 2e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101 --use-amp --CUDA_VISIBLE_DEVICES 0,1
```

## (Optional) Evaluating

학습한 모델을 k-fold cross validation 진행한다. 앞에서 학습에 사용한 모델을 사용하거나, `--model-dir`안에 있는 모델을 명시하여 평가할 수 있다. 

평가 결과는 `./logs/`에 저장되며, 전체 합쳐진 결과 (Out-of-folds)는  `./oofs/` 폴더에 저장된다. 

```
python evaluate.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 

python evaluate.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns 

python evaluate.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4

python evaluate.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns

python evaluate.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns

python evaluate.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns  --n-meta-dim 128,32

python evaluate.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns

python evaluate.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns

python evaluate.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns 

python evaluate.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101

python evaluate.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101
```

## Predicting

Test 데이터에 따른 예측을 실시한다. 

각 모델의 평가 결과는 아래 `./subs/` 폴더에 저장된다. 

```
python predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 

python predict.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns 

python predict.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4

python predict.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns

python predict.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns

python predict.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns  --n-meta-dim 128,32

python predict.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns

python predict.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns

python predict.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns 

python predict.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101

python predict.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101
```

## Ensembling (앙상블)

최종 평가 결과를 위해, 18개의 모델의 예측 결과를 모아서 최종 예측을 진행한다. 

root 폴더에 `final_sub1.csv`라는 형태로 출력을 만들어준다. 

```
python ensemble.py
```

