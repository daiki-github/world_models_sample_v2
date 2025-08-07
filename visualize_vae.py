# visualize_vae.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt

from config import DEVICE, VAE_PATH, DATA_DIR, LATENT_DIM
from models.vae import VAE

def visualize_reconstruction(num_images=10):
    """
    学習済みのVAEをロードし、元の画像と再構成された画像を並べて表示する。
    """
    print("--- VAEの再構成結果を可視化します ---")

    # 1. 学習済みのVAEモデルをロード
    try:
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        vae.eval()
        print(f"VAEモデル ({VAE_PATH}) のロードに成功しました。")
    except FileNotFoundError:
        print(f"エラー: VAEモデル ({VAE_PATH}) が見つかりません。")
        print("先に train_vae.py を実行して、モデルを学習させてください。")
        return

    # 2. データファイル（最初のチャンク）をロード
    file_list = sorted(glob.glob(os.path.join(DATA_DIR, "series_data_part_*.npz")))
    if not file_list:
        print(f"エラー: データファイルが {DATA_DIR} に見つかりません。")
        return
    
    with np.load(file_list[0]) as data:
        observations = data['observations']
    print(f"データ ({file_list[0]}) のロードに成功しました。")

    # 3. ランダムに画像を選択
    sample_indices = random.sample(range(len(observations)), num_images)
    original_images_np = observations[sample_indices]

    # 4. 画像を前処理
    processed_images = []
    for img in original_images_np:
        resized = cv2.resize(img, (64, 64))
        processed_images.append(resized)
    
    processed_images_np = np.array(processed_images, dtype=np.float32) / 255.0
    images_tensor = torch.from_numpy(processed_images_np).permute(0, 3, 1, 2).to(DEVICE)

    # 5. VAEで画像を再構成
    with torch.no_grad():
        recon_images_tensor, _, _ = vae(images_tensor)

    # 6. 表示用にテンソルをNumPy配列に戻す
    recon_images_np = recon_images_tensor.cpu().permute(0, 2, 3, 1).numpy()

    # 7. Matplotlibで結果を表示
    fig, axes = plt.subplots(num_images, 2, figsize=(6, num_images * 2))
    fig.suptitle("VAE Reconstruction Results", fontsize=16)
    
    for i in range(num_images):
        # 元の画像
        axes[i, 0].imshow(processed_images_np[i])
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')
        
        # 再構成された画像
        axes[i, 1].imshow(recon_images_np[i])
        axes[i, 1].set_title(f"Reconstructed {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # Matplotlibをインストールする必要がある場合があります:
    # pip install matplotlib
    visualize_reconstruction(num_images=8)