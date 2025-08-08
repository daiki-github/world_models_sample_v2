# save_video.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium
import numpy as np
import torch
import cv2
import imageio # ★★★ imageioをインポート

# --- 必要な設定とモデルをインポート ---
from config import (DEVICE, VAE_PATH, MDNRNN_PATH, CONTROLLER_PATH,
                    LATENT_DIM, ACTION_DIM, HIDDEN_UNITS, N_GAUSSIANS)
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

def save_video(filename="carracing_play.mp4", use_finetuned=False, num_episodes=1):
    """
    学習済みのモデルでCarRacing-v3をプレイし、その様子をMP4動画として保存する。
    """
    
    # --- モデルの初期化と重みのロード ---
    print("学習済みモデルをロードしています...")
    try:
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))

        mdnrnn = MDNRNN(
            latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
            hidden_units=HIDDEN_UNITS, n_gaussians=N_GAUSSIANS
        ).to(DEVICE)
        mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))

        controller = Controller(
            latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_units=HIDDEN_UNITS
        ).to(DEVICE)
        
        controller_path = CONTROLLER_PATH
        print(f"事前学習済みモデルを使用します: {controller_path}")
            
        best_params_vector = torch.load(controller_path, map_location=DEVICE)
        torch.nn.utils.vector_to_parameters(best_params_vector, controller.parameters())

    except FileNotFoundError as e:
        print(f"モデルのロードエラー: {e}")
        return
        
    vae.eval(); mdnrnn.eval(); controller.eval()
    print("モデルのロードが完了しました。")

    # --- ゲーム環境の初期化 ---
    # ★★★ render_modeを'rgb_array'に変更 ★★★
    # ★★★ 環境名を "CarRacing-v2" に修正 ★★★
    env = gymnasium.make("CarRacing-v3", continuous=True, render_mode='rgb_array')

    # ★★★ imageioの動画ライターを初期化 ★★★
    # fpsで使用するフレームレート（1秒あたりのコマ数）を指定
    writer = imageio.get_writer(filename, fps=50)

    with torch.no_grad():
        for episode in range(num_episodes):
            print(f"\n--- 動画生成のためのエピソード {episode + 1} を開始 ---")
            total_reward = 0.0
            
            obs, _ = env.reset()
            hidden = (torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE),
                      torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE))

            done = False
            time_steps = 0
            while not done:
                # ★★★ 毎ステップ、画面をキャプチャして動画に追加 ★★★
                frame = env.render()
                writer.append_data(frame)

                # --- 以下は通常のプレイロジック ---
                obs_resized = cv2.resize(obs, (64, 64))
                obs_tensor = torch.from_numpy(obs_resized).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

                mu, _ = vae.encode(obs_tensor)
                z = mu

                h = hidden[0].squeeze(0)
                action_tensor = controller(z, h)
                action = action_tensor.squeeze().cpu().numpy()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                time_steps += 1
                
                action_tensor_seq = action_tensor.unsqueeze(1)
                z_seq = z.unsqueeze(1)
                _, _, _, hidden = mdnrnn(z_seq, action_tensor_seq, hidden)

                if time_steps > 1000:
                    done = True

            print(f"エピソード終了: 総リワード = {total_reward:.2f}, ステップ数 = {time_steps}")

    # ★★★ ライターを閉じて、動画ファイルを完成させる ★★★
    writer.close()
    env.close()
    print(f"\n動画が正常に保存されました: {filename}")


if __name__ == "__main__":
    # 1エピソード分のプレイを動画として保存する
    save_video(filename="carracing_play.mp4", use_finetuned=False, num_episodes=1)