# visualize_vae.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from config import DEVICE, VAE_PATH, DATA_DIR, LATENT_DIM
from models.vae import VAE

def visualize_reconstruction(num_images=10, output_filename="vae_reconstruction_result.png"):
    """
    学習済みのVAEをロードし、元の画像と再構成された画像を並べて表示し、
    結果を画像ファイルとして保存する。
    """
    print("--- VAEの再構成結果を可視化し、ファイルに保存します ---")

    # 1. 学習済みのVAEモデルをロード
    try:
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        vae.eval()
        print(f"VAEモデル ({VAE_PATH}) のロードに成功しました。")
    except FileNotFoundError:
        print(f"エラー: VAEモデル ({VAE_PATH}) が見つかりません。")
        return

    # 2. 前処理済みのデータファイルをロード
    try:
        processed_dir = os.path.join(DATA_DIR, "processed_memmap")
        obs_path = os.path.join(processed_dir, 'observations.mmap')
        actions_path = os.path.join(processed_dir, 'actions.npy')
        
        if not os.path.exists(obs_path): raise FileNotFoundError
        
        total_length = len(np.load(actions_path))
        observations_mmap = np.memmap(obs_path, dtype=np.uint8, mode='r', shape=(total_length, 64, 64, 3))
        print(f"前処理済みデータ ({processed_dir}) のロードに成功しました。")
    except FileNotFoundError:
        print(f"エラー: 前処理済みデータが {processed_dir} に見つかりません。")
        return

    # 3. ランダムに画像を選択
    sample_indices = random.sample(range(total_length), num_images)
    original_images_np = observations_mmap[sample_indices]

    # 4. PyTorchテンソルに変換
    images_float = original_images_np.astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images_float).permute(0, 3, 1, 2).to(DEVICE)

    # 5. VAEで画像を再構成
    with torch.no_grad():
        recon_images_tensor, _, _ = vae(images_tensor)

    # 6. 表示用にテンソルをNumPy配列に戻す
    recon_images_np = recon_images_tensor.cpu().permute(0, 2, 3, 1).numpy()

    # 7. Matplotlibで結果をプロット
    fig, axes = plt.subplots(num_images, 2, figsize=(6, num_images * 2))
    fig.suptitle("VAE Reconstruction Results\n(Left: Original / Right: Reconstructed)", fontsize=16)
    
    for i in range(num_images):
        axes[i, 0].imshow(images_float[i])
        axes[i, 0].set_title(f"Original #{i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon_images_np[i])
        axes[i, 1].set_title(f"Reconstructed #{i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- ★★★ ここから修正 ★★★ ---
    # 画面に表示する代わりに、ファイルに保存する
    try:
        plt.savefig(output_filename, dpi=150) # dpiで解像度を指定
        print(f"\n可視化結果を {output_filename} に保存しました。")
    except Exception as e:
        print(f"\n画像の保存中にエラーが発生しました: {e}")
    
    # メモリを解放するためにプロットを閉じる
    plt.close()
    # --- ★★★ 修正ここまで ★★★ ---


if __name__ == "__main__":
    visualize_reconstruction(num_images=8, output_filename="vae_reconstruction_result.png")