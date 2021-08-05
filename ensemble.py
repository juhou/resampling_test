import pandas as pd
import numpy as np
from glob import glob
import argparse
import os
from sklearn.metrics import roc_auc_score


'''
# 모델 앙상블

총 18개 학습한 모델을 predict를 통해 평가한 결과가 저장된 csv파일
      ./subs/*.csv
를 읽어온 뒤, 모든 값의 평균을 내서 결과를 평가함

# When ensembling different folds, or different models,
# we first rank all the probabilities of each model/fold,
# to ensure they are evenly distributed.
# In pandas, it can be done by df['pred'] = df['pred'].rank(extract)

'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, default='original_stone/')
    args, _ = parser.parse_known_args()
    return args

def reject_outliers(data, m=1.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


if __name__ == '__main__':
    args = parse_args()

    # 폴더에서 csv읽어오기
    subs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.sub_dir, '*csv')))]
    sub_probs_target_valence = [sub.Valence for sub in subs]
    sub_probs_target_arousal = [sub.Arousal for sub in subs]
    sub_probs_target_stress1 = [sub.Stress1 for sub in subs]
    sub_probs_target_stress2 = [sub.Stress2 for sub in subs]

    # 앙상블을 위한 균등 가중치
    ensem_number = len(sub_probs_target_valence)
    wts = [1/ensem_number]*ensem_number

    # 가중치 반영하여 결과 평균내기
    PROBS_target_valence = np.sum([wts[i] * sub_probs_target_valence[i] for i in range(len(wts))], axis=0)
    PROBS_target_arousal = np.sum([wts[i] * sub_probs_target_arousal[i] for i in range(len(wts))], axis=0)
    PROBS_target_stress1 = np.sum([wts[i] * sub_probs_target_stress1[i] for i in range(len(wts))], axis=0)
    PROBS_target_stress2 = np.sum([wts[i] * sub_probs_target_stress2[i] for i in range(len(wts))], axis=0)

    remove_outlier = True

    minus_bound = 1
    if remove_outlier:
        # 앙상블 최종 결과를 저장함
        df_sub = subs[0]
        df_sub['Valence'] = [np.mean(np.sort(sub)[minus_bound:-minus_bound]) for sub in np.transpose(sub_probs_target_valence)]
        df_sub['Arousal'] = [np.mean(np.sort(sub)[minus_bound:-minus_bound]) for sub in np.transpose(sub_probs_target_arousal)]
        df_sub['Stress1'] = [np.mean(np.sort(sub)[minus_bound:-minus_bound]) for sub in np.transpose(sub_probs_target_stress1)]
        df_sub['Stress2'] = [np.mean(np.sort(sub)[minus_bound:-minus_bound]) for sub in np.transpose(sub_probs_target_stress1)]
        df_sub.to_csv(f"final_sub1_select_med_{minus_bound}.csv", index=False)


    # Test에 대한 정답이 없는 경우
    # 앙상블 최종 결과를 저장함
    df_sub = subs[0]
    df_sub['Valence'] = PROBS_target_valence
    df_sub['Arousal'] = PROBS_target_arousal
    df_sub['Stress1'] = PROBS_target_stress1
    df_sub['Stress2'] = PROBS_target_stress2
    df_sub.to_csv(f"final_sub1.csv", index=False)




