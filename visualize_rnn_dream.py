# visualize_future.py
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
        # VAE
        vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        vae.eval()

        # MDN-RNN
        mdnrnn = MDNRNN(
            latent_dim=LATENT_DIM,
            action_dim=ACTION_DIM,
            hidden_units=HIDDEN_UNITS,
            n_gaussians=N_GAUSSIANS
        ).to(DEVICE)
        mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))
        mdnrnn.eval()

        # Controller (オプション)
        controller = Controller(
            latent_dim=LATENT_DIM,
            hidden_units=HIDDEN_UNITS,
            action_dim=ACTION_DIM
        ).to(DEVICE)
        
        if os.path.exists(CONTROLLER_PATH):
            controller_params = torch.load(CONTROLLER_PATH, map_location=DEVICE)
            torch.nn.utils.vector_to_parameters(controller_params, controller.parameters())
            controller.eval()
            print("✔ Controller をロードしました。学習済み方策で動作します。")
        else:
            controller = None
            print("✘ Controller が見つかりませんでした。ランダムアクションで動作します。")

        print("✔ VAE と MDN-RNN のモデルを正常にロードしました。")
        return vae, mdnrnn, controller

    except FileNotFoundError as e:
        print(f"モデルファイルのロードに失敗しました: {e}")
        print("エラー: VAEとMDN-RNNの学習が完了していることを確認してください。")
        print("先に `python main.py --steps vae rnn` を実行してください。")
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
    # 色空間をRGBからBGRに変換してOpenCVで正しく表示
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def select_next_z(pi, mu):
    """MDN-RNNの出力(混合ガウス分布)から、最も確率の高い平均(mu)を次の潜在ベクトルとして選択する"""
    # pi: [1, 1, n_gaussians], mu: [1, 1, n_gaussians, latent_dim]
    best_pi_idx = torch.argmax(pi.squeeze(), dim=-1)
    selected_mu = mu.squeeze()[best_pi_idx, :]
    return selected_mu.unsqueeze(0) # [1, latent_dim]

def draw_text_on_image(image, text):
    """画像にラベルテキストを描画する"""
    height, width, _ = image.shape
    bg_color = (0, 0, 0) # 黒背景
    font_color = (255, 255, 255) # 白文字
    
    # テキスト用の黒帯を上部に追加
    header = np.full((25, width, 3), bg_color, dtype=np.uint8)
    
    cv2.putText(header, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
    
    # 元の画像と結合
    return np.vstack((header, image))

def run_visualization(vae, mdnrnn, controller, episodes=5):
    """
    【動画保存・比較版】
    実際の観測と未来予測を並べて比較する動画を保存する。
    """
    env = gymnasium.make("CarRacing-v3", render_mode='rgb_array', continuous=True)
    
    output_filename = "future_prediction_comparison.mp4"
    print(f"シミュレーション結果は '{output_filename}' に動画として保存されます。")

    for ep in range(episodes):
        print(f"\n--- エピソード {ep + 1}/{episodes} ---")
        
        obs, _ = env.reset()
        done = False
        
        video_frames = []
        
        hidden = (torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE),
                  torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE))

        predicted_obs_image = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(10):
            initial_action = np.array([0.0, 0.2, 0.0]) 
            obs, _, _, _, _ = env.step(initial_action)
        
        for t in trange(1000, desc="比較動画フレーム生成中"):
            obs_tensor = preprocess_obs(obs)
            
            with torch.no_grad():
                z, _ = vae.encode(obs_tensor)
                if controller:
                    h = hidden[0].squeeze(0)
                    action_tensor = controller(z, h)
                    action = action_tensor.squeeze().cpu().numpy()
                else:
                    action = env.action_space.sample()
                
                action_tensor_seq = torch.from_numpy(action).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
                z_seq = z.unsqueeze(0)
                pi, mu_pred, _, next_hidden = mdnrnn(z_seq, action_tensor_seq, hidden)
                
                z_pred = select_next_z(pi, mu_pred)
                predicted_obs_tensor = vae.decode(z_pred)

            ground_truth_image = cv2.resize(obs, (64, 64))
            ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_RGB2BGR)

            # ★★★ ここからが変更点 ★★★
            # 「実際の観測」と「予測」のコンポーネントをそれぞれ作成
            gt_component = draw_text_on_image(ground_truth_image, "Ground Truth (t)")
            prediction_component = draw_text_on_image(predicted_obs_image, "MDN-RNN Prediction (t)")

            # 2つのコンポーネントを水平に結合して比較フレームを作成
            comparison_frame = np.hstack([gt_component, prediction_component])
            
            # 生成した比較フレームをリストに追加
            video_frames.append(comparison_frame)
            # ★★★ 変更点ここまで ★★★

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
                
            hidden = next_hidden
            predicted_obs_image = tensor_to_image(predicted_obs_tensor)

        if not video_frames:
            print("警告: 生成されたフレームがありません。動画は作成されませんでした。")
            continue

        print("フレームの生成が完了しました。動画ファイルに書き出しています...")
        height, width, _ = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

        for frame in video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✔ 動画が '{output_filename}' として正常に保存されました。")

    env.close()
    print("すべての処理が完了しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="World Models Future Prediction Visualization")
    parser.add_argument('--episodes', type=int, default=5, help="実行するエピソード数")
    args = parser.parse_args()

    vae_model, mdnrnn_model, controller_model = load_models()
    if vae_model and mdnrnn_model:
        run_visualization(vae_model, mdnrnn_model, controller_model, episodes=args.episodes)