# play.py
import gymnasium
import numpy as np
import torch
import cv2
import time

# --- 必要な設定とモデルをインポート ---
from config import (DEVICE, VAE_PATH, MDNRNN_PATH, CONTROLLER_PATH,
                    LATENT_DIM, ACTION_DIM, HIDDEN_UNITS, N_GAUSSIANS)
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

def play_game(num_episodes=5):
    """
    学習済みのモデルをロードして、CarRacing-v3をプレイする。
    """
    
    # --- モデルの初期化と重みのロード ---
    print("学習済みモデルをロードしています...")
    try:
        # VAE
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))

        # MDN-RNN
        mdnrnn = MDNRNN(
            latent_dim=LATENT_DIM,
            action_dim=ACTION_DIM,
            hidden_units=HIDDEN_UNITS,
            n_gaussians=N_GAUSSIANS
        ).to(DEVICE)
        mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))

        # Controller
        controller = Controller(
            latent_dim=LATENT_DIM,
            action_dim=ACTION_DIM,
            hidden_units=HIDDEN_UNITS
        ).to(DEVICE)
        # Controllerの重みはCMA-ESから保存された1次元ベクトルなので、モデルにロードする
        best_params_vector = torch.load(CONTROLLER_PATH, map_location=DEVICE)
        torch.nn.utils.vector_to_parameters(best_params_vector, controller.parameters())

    except FileNotFoundError as e:
        print(f"モデルのロードエラー: {e}")
        print("必要なモデルファイルが 'weights' フォルダに存在するか確認してください。")
        print("学習が完了していない場合は、先に main.py で学習を実行してください。")
        return
        
    # すべてのモデルを評価モードに設定
    vae.eval()
    mdnrnn.eval()
    controller.eval()
    print("モデルのロードが完了しました。")

    # --- ゲーム環境の初期化 ---
    # render_mode='human' でゲーム画面を表示
    env = gymnasium.make("CarRacing-v3", continuous=True, render_mode='human')

    with torch.no_grad():
        for episode in range(num_episodes):
            print(f"\n--- エピソード {episode + 1} 開始 ---")
            total_reward = 0.0
            
            # --- 環境とRNNの隠れ状態をリセット ---
            obs, _ = env.reset()
            hidden = (torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE),
                      torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE))

            # 画面が表示されるまで少し待機
            time.sleep(1)

            done = False
            time_steps = 0
            while not done:
                # 1. 観測（画像）を前処理
                obs_resized = cv2.resize(obs, (64, 64))
                obs_tensor = torch.from_numpy(obs_resized).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

                # 2. VAEで潜在ベクトルzを取得
                mu, _ = vae.encode(obs_tensor)
                z = mu

                # 3. Controllerで行動を決定
                h = hidden[0].squeeze(0)
                action_tensor = controller(z, h)
                action = action_tensor.squeeze().cpu().numpy()

                # 4. 環境を1ステップ進める
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                time_steps += 1
                
                # 5. MDN-RNNで次の隠れ状態を更新
                action_tensor_seq = action_tensor.unsqueeze(1)
                z_seq = z.unsqueeze(1)
                _, _, _, hidden = mdnrnn(z_seq, action_tensor_seq, hidden)

                # ゲームの終了条件（任意）
                if time_steps > 1000:
                    print("最大ステップ数に達しました。")
                    done = True

            print(f"エピソード {episode + 1} 終了: 総リワード = {total_reward:.2f}, ステップ数 = {time_steps}")
            time.sleep(2) # 次のエピソードの前に少し待機

    env.close()
    print("\nプレイが完了しました。")


if __name__ == "__main__":
    # 5回ゲームをプレイする
    play_game(num_episodes=5)