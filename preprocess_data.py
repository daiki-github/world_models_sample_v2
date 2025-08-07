# preprocess_data.py
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from config import DATA_DIR

def run_preprocessing():
    """
    収集した生データを、memmapで扱える単一の巨大なバイナリファイルに前処理して保存する。
    """
    print("--- データの前処理を開始します (memmap形式) ---")
    
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_memmap")
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    raw_file_list = sorted(glob.glob(os.path.join(DATA_DIR, "series_data_part_*.npz")))
    if not raw_file_list:
        print(f"エラー: 前処理対象のデータが {DATA_DIR} に見つかりません。")
        return

    # 最初に全データ数を計算
    total_steps = sum([len(np.load(f)['observations']) for f in raw_file_list])
    print(f"合計ステップ数: {total_steps}")

    # ★★★ memmapファイルを作成 ★★★
    # observationsは巨大なのでmemmapで扱う
    obs_shape = (total_steps, 64, 64, 3)
    obs_memmap_path = os.path.join(PROCESSED_DATA_DIR, 'observations.mmap')
    processed_obs = np.memmap(obs_memmap_path, dtype=np.uint8, mode='w+', shape=obs_shape)
    
    # actionsとdonesは比較的小さいので、通常のnpyファイルでOK
    all_actions = []
    all_dones = []
    
    current_pos = 0
    for file_path in tqdm(raw_file_list, desc="前処理中"):
        with np.load(file_path) as data:
            observations = data['observations']
            
            # 画像を64x64にリサイズ
            resized_obs = np.array([cv2.resize(obs, (64, 64)) for obs in observations], dtype=np.uint8)
            
            # memmapファイルの対応するスライスに書き込む
            num_obs = len(resized_obs)
            processed_obs[current_pos:current_pos + num_obs] = resized_obs
            current_pos += num_obs
            
            all_actions.append(data['actions'])
            all_dones.append(data['dones'])

    # actionsとdonesを結合して保存
    actions = np.concatenate(all_actions, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'actions.npy'), actions)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'dones.npy'), dones)

    print(f"前処理が完了しました。データは {PROCESSED_DATA_DIR} に保存されました。")

if __name__ == "__main__":
    run_preprocessing()