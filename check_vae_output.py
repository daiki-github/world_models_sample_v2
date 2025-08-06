# check_vae_output.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import cv2
from tqdm import tqdm

from config import DEVICE, VAE_PATH, SERIES_PATH, LATENT_DIM
from models.vae import VAE

def run_check():
    """
    VAEをロードし、データの一部を潜在ベクトルzに変換して、
    その値がnanやinfになっていないかを確認する。
    """
    print("--- VAE出力の健全性チェックを開始します ---")

    # 1. 学習済みのVAEモデルをロード
    try:
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        vae.eval()
        print(f"VAEモデル ({VAE_PATH}) のロードに成功しました。")
    except FileNotFoundError:
        print(f"エラー: VAEモデル ({VAE_PATH}) が見つかりません。")
        return

    # 2. データファイルをロード
    try:
        data = np.load(SERIES_PATH)
        observations = data['observations']
        print(f"データファイル ({SERIES_PATH}) のロードに成功しました。")
    except FileNotFoundError:
        print(f"エラー: データファイル ({SERIES_PATH}) が見つかりません。")
        return

    # 3. データを前処理してVAEに入力し、出力をチェック
    print("データをエンコードしてnan/infの有無を確認中...")
    all_z = []
    with torch.no_grad():
        batch_size = 256
        for i in tqdm(range(0, len(observations), batch_size), desc="エンコード進捗"):
            obs_batch_np = observations[i:i+batch_size]
            
            # 前処理
            obs_list = [cv2.resize(obs, (64, 64)) for obs in obs_batch_np]
            obs_tensor = torch.from_numpy(np.array(obs_list)).float().permute(0, 3, 1, 2) / 255.0
            obs_tensor = obs_tensor.to(DEVICE)

            # VAEでエンコード
            mu, logvar = vae.encode(obs_tensor)
            
            # nan/inf チェック
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! 致命的なエラー: VAEの出力 (mu) に nan または inf が検出されました。")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return

            if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! 致命的なエラー: VAEの出力 (logvar) に nan または inf が検出されました。")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return

    print("\n--- チェック完了 ---")
    print("✅ VAEの出力は正常です。nanやinfは検出されませんでした。")
    print("問題の原因は、MDN-RNNの学習プロセス内部にある可能性が高いです。")


if __name__ == "__main__":
    run_check()