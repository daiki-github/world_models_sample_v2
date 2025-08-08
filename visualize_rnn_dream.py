# visualize_dream.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import cv2
import gymnasium
from tqdm import trange
import argparse

# 既存のスクリプトから設定とモデルクラスをインポート
from config import (DEVICE, VAE_PATH, MDNRNN_PATH, CONTROLLER_PATH,
                    LATENT_DIM, ACTION_DIM, HIDDEN_UNITS, N_GAUSSIANS)
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

def load_models():
    """学習済みのVAE, MDN-RNN, Controllerモデルをロードする"""
    print("学習済みモデルのロードを開始します...")
    try:
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        vae.eval()

        mdnrnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_units=HIDDEN_UNITS, n_gaussians=N_GAUSSIANS).to(DEVICE)
        mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))
        mdnrnn.eval()

        controller = Controller(latent_dim=LATENT_DIM, hidden_units=HIDDEN_UNITS, action_dim=ACTION_DIM).to(DEVICE)
        if os.path.exists(CONTROLLER_PATH):
            controller_params = torch.load(CONTROLLER_PATH, map_location=DEVICE)
            torch.nn.utils.vector_to_parameters(controller_params, controller.parameters())
            controller.eval()
            print("✔ Controller をロードしました。")
        else:
            # この可視化にはControllerが必須
            print("✘ Controller が見つかりませんでした。この可視化はControllerがないと実行できません。")
            return None, None, None
            
        print("✔ VAE, MDN-RNN, Controller のモデルを正常にロードしました。")
        return vae, mdnrnn, controller

    except FileNotFoundError as e:
        print(f"モデルファイルのロードに失敗しました: {e}")
        print("エラー: VAE, MDN-RNN, Controllerの学習が完了していることを確認してください。")
        return None, None, None

def preprocess_obs(obs):
    """観測画像をVAEの入力形式に前処理する"""
    obs_resized = cv2.resize(obs, (64, 64))
    obs_tensor = torch.from_numpy(obs_resized).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    return obs_tensor

def tensor_to_image(tensor):
    """PyTorchの画像テンソルをOpenCVで表示可能なNumpy配列に変換する"""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def select_next_z(pi, mu):
    """MDN-RNNの出力から最も確率の高い平均(mu)を次の潜在ベクトルとして選択する"""
    best_pi_idx = torch.argmax(pi.squeeze(), dim=-1)
    selected_mu = mu.squeeze()[best_pi_idx, :]
    return selected_mu.unsqueeze(0)

def draw_text_on_image(image, text, color=(255, 255, 255)):
    """画像にラベルテキストを描画する"""
    header = np.full((25, image.shape[1], 3), (0,0,0), dtype=np.uint8)
    cv2.putText(header, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack((header, image))

def run_dream_visualization(vae, mdnrnn, controller, prime_steps=100, dream_steps=300):
    """モデルの「夢」と「現実」を比較する動画を生成する"""
    env = gymnasium.make("CarRacing-v3", render_mode='rgb_array', continuous=True)
    
    output_filename = "dream_vs_reality.mp4"
    print(f"シミュレーション結果は '{output_filename}' に動画として保存されます。")
    
    obs, _ = env.reset()
    # 最初の数フレームは少しだけアクセルを踏む
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.2, 0.0]))

    video_frames = []
    hidden = (torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE),
              torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE))

    # --- 1. 準備(Priming)フェーズ ---
    # モデルに現在の状況を認識させる
    print(f"--- 準備フェーズ ({prime_steps}ステップ) ---")
    for t in trange(prime_steps, desc="準備中"):
        obs_tensor = preprocess_obs(obs)
        with torch.no_grad():
            z, _ = vae.encode(obs_tensor)
            h = hidden[0].squeeze(0)
            action_tensor = controller(z, h)
            action = action_tensor.squeeze().cpu().numpy()
            
            # RNNの隠れ状態を更新
            _, _, _, hidden = mdnrnn(z.unsqueeze(0), action_tensor.unsqueeze(0), hidden)
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print("準備中にエピソードが終了しました。")
            return
            
        # 準備中は左右に同じ現実の画像を表示
        ground_truth_image = cv2.resize(obs, (64, 64))
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_RGB2BGR)
        
        left_panel = draw_text_on_image(ground_truth_image, "Reality (Priming)")
        right_panel = draw_text_on_image(ground_truth_image, "Dream (Waiting...)")
        video_frames.append(np.hstack([left_panel, right_panel]))

    # 夢の開始時点での潜在ベクトルと隠れ状態を保存
    z_dream_start, _ = vae.encode(preprocess_obs(obs))
    h_dream_start = hidden

    # --- 2. 想像(Dreaming)フェーズ ---
    # モデルの想像と現実を比較する
    print(f"\n--- 想像フェーズ ({dream_steps}ステップ) ---")
    z_dream = z_dream_start
    hidden_dream = h_dream_start
    
    for t in trange(dream_steps, desc="想像中"):
        with torch.no_grad():
            # Controllerは「夢の中の状況」から行動を決定
            h_dream_controller = hidden_dream[0].squeeze(0)
            action_tensor = controller(z_dream, h_dream_controller)
            action = action_tensor.squeeze().cpu().numpy()

            # --- 右パネル: 夢の世界を生成 ---
            # 夢の中のzと行動から、次の夢のzを予測
            _, mu_pred, _, hidden_dream = mdnrnn(z_dream.unsqueeze(0), action_tensor.unsqueeze(0), hidden_dream)
            z_dream = select_next_z(pi=torch.ones_like(mu_pred[:,:,:,0]), mu=mu_pred) # select_next_zのpiは使わないのでダミーを渡す
            dream_image = tensor_to_image(vae.decode(z_dream))
            right_panel = draw_text_on_image(dream_image, "Dream", color=(100, 255, 100))
            
            # --- 左パネル: 現実の世界を記録 ---
            # 夢の中で決定した行動を、現実の環境に適用する
            obs, _, terminated, truncated, _ = env.step(action)
            real_image = cv2.resize(obs, (64, 64))
            real_image = cv2.cvtColor(real_image, cv2.COLOR_RGB2BGR)
            left_panel = draw_text_on_image(real_image, "Reality")

        video_frames.append(np.hstack([left_panel, right_panel]))
        
        if terminated or truncated:
            print("\n想像中にエピソードが終了しました。")
            break

    # --- 3. 動画の保存 ---
    if not video_frames:
        print("警告: 生成されたフレームがありません。")
        return

    print("\nフレーム生成完了。動画ファイルに書き出しています...")
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))
    for frame in video_frames:
        video_writer.write(frame)
    video_writer.release()
    print(f"✔ 動画が '{output_filename}' として正常に保存されました。")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="World Models Dream Visualization")
    parser.add_argument('--prime', type=int, default=100, help="モデルに現実を見せる準備ステップ数")
    parser.add_argument('--dream', type=int, default=300, help="モデルが夢を見る想像ステップ数")
    args = parser.parse_args()

    vae_model, mdnrnn_model, controller_model = load_models()
    if vae_model and mdnrnn_model and controller_model:
        run_dream_visualization(vae_model, mdnrnn_model, controller_model, prime_steps=args.prime, dream_steps=args.dream)