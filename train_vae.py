# train_vae.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.optim as optim
# ★★★ Datasetクラスをインポート ★★★
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import glob

from config import (DEVICE, VAE_PATH, DATA_DIR, WEIGHTS_DIR,
                    LATENT_DIM, VAE_BATCH_SIZE, VAE_LEARNING_RATE, VAE_EPOCHS)
from models.vae import VAE, vae_loss_function

# --- ★★★ ここからカスタムDatasetクラスを新規作成 ★★★ ---
class VAEDataset(Dataset):
    """
    ディスク上の複数のデータチャンクを、メモリに全て読み込むことなく扱うためのカスタムDataset。
    """
    def __init__(self, data_dir):
        super().__init__()
        processed_dir = os.path.join(data_dir, "processed_memmap")
        
        obs_path = os.path.join(processed_dir, 'observations.mmap')
        actions_path = os.path.join(processed_dir, 'actions.npy')

        if not os.path.exists(obs_path) or not os.path.exists(actions_path):
             raise FileNotFoundError(f"前処理済みデータが {processed_dir} に見つかりません。preprocess_data.pyを先に実行してください。")

        # actions.npyから全体の長さを取得
        self.length = len(np.load(actions_path))
        
        # memmapファイルを開く（データはまだメモリに読み込まれない）
        self.observations = np.memmap(obs_path, dtype=np.uint8, mode='r', shape=(self.length, 64, 64, 3))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 与えられたインデックス(idx)が、どのファイルの何番目にあたるかを計算
        obs = self.observations[idx]
        
        # 前処理
        obs_float = obs.astype(np.float32) / 255.0
        obs_tensor = torch.from_numpy(obs_float).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        
        return obs_tensor

def run_train_vae():
    """ VAEモデルの学習を実行する """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    # --- ★★★ ここからデータ読み込み部分を全面的に修正 ★★★ ---
    # np.concatenateは使わず、カスタムDatasetを初期化するだけ
    try:
        dataset = VAEDataset(data_dir=DATA_DIR)
        print(f"データセットの準備完了。合計ステップ数: {len(dataset)}")
    except FileNotFoundError as e:
        print(e)
        print("先に collect_data.py を実行してください。")
        return
    # --- ★★★ 修正ここまで ★★★ ---
    
    # DataLoaderは、このカスタムDatasetからバッチを賢く読み込んでくれる
    dataloader = DataLoader(dataset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4) # Windowsではnum_workers=0が安定
    
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=VAE_LEARNING_RATE)

    if os.path.exists(VAE_PATH):
        try:
            vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
            print(f"既存のVAEモデルをロードしました: {VAE_PATH}")
            print("学習を再開します...")
        except Exception as e:
            print(f"VAEモデルのロードに失敗しました: {e}")
            print("新規に学習を開始します。")
    else:
        print("新規にVAEの学習を開始します。")
    
    vae.train()
    for epoch in range(VAE_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{VAE_EPOCHS}")
        for obs_batch in pbar:
            # print(f"batch loade!!!")
            obs_batch = obs_batch.to(DEVICE) # DataLoaderが返すのは1つのテンソル
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(obs_batch)
            loss = vae_loss_function(recon_batch, obs_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item() / len(obs_batch):.4f}")
            
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{VAE_EPOCHS}, 平均損失: {avg_loss:.4f}")

    torch.save(vae.state_dict(), VAE_PATH)
    print(f"学習済みVAEモデルが {VAE_PATH} に保存されました。")

if __name__ == "__main__":
    run_train_vae()