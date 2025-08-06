# train_controller.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium
import numpy as np # <- numpyをインポート
import torch
import cv2
import cma
from tqdm import tqdm

from config import (DEVICE, VAE_PATH, MDNRNN_PATH, CONTROLLER_PATH, WEIGHTS_DIR,
                    LATENT_DIM, HIDDEN_UNITS, CMA_POPULATION_SIZE, CMA_SIGMA, CMA_GENERATIONS,
                    ACTION_DIM, N_GAUSSIANS) # モデルの引数用に追記
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

def run_train_controller():
    """ Controllerの学習をCMA-ESで実行する """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # --- モデルのロード ---
    # configから設定を読み込むように修正
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    mdnrnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_units=HIDDEN_UNITS,
        n_gaussians=N_GAUSSIANS
    ).to(DEVICE)
    controller = Controller(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_units=HIDDEN_UNITS
    ).to(DEVICE)

    try:
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"モデルのロードエラー: {e}。先に学習スクリプトを実行してください。")
        return

    vae.eval()
    mdnrnn.eval()

    def rollout(params_vector):
        torch.nn.utils.vector_to_parameters(torch.tensor(params_vector, dtype=torch.float32, device=DEVICE), controller.parameters())
        controller.eval()
        # 環境名を v2 に修正
        env = gymnasium.make("CarRacing-v3", continuous=True, render_mode=None)
        total_reward = 0.0
        with torch.no_grad():
            obs, _ = env.reset()
            hidden = (torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE), torch.zeros(1, 1, HIDDEN_UNITS).to(DEVICE))
            
            # --- 修正点 ---
            # np.array() でNumPy配列に変換する
            initial_action = np.array([0.0, 0.2, 0.0]) 
            for _ in range(10):
                obs, _, _, _, _ = env.step(initial_action)

            time_steps = 0
            done = False
            while not done and time_steps < 1000:
                obs_resized = cv2.resize(obs, (64, 64))
                obs_tensor = torch.from_numpy(obs_resized).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
                mu, _ = vae.encode(obs_tensor)
                z = mu
                h = hidden[0].squeeze(0)
                action_tensor = controller(z, h)
                # ここは元から正しく .numpy() で変換されている
                action = action_tensor.squeeze().cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if reward < -0.1 and time_steps > 20:
                    done = True
                action_tensor_seq = action_tensor.unsqueeze(1)
                z_seq = z.unsqueeze(1)
                _, _, _, hidden = mdnrnn(z_seq, action_tensor_seq, hidden)
                time_steps += 1
        env.close()
        return total_reward

    n_params = sum(p.numel() for p in controller.parameters())
    initial_params = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
    
    es = cma.CMAEvolutionStrategy(initial_params, CMA_SIGMA, {'popsize': CMA_POPULATION_SIZE})
    
    print("CMA-ESによるControllerの学習を開始します...")
    best_reward = -np.inf
    
    for g in range(CMA_GENERATIONS):
        solutions = es.ask()
        rewards = [rollout(s) for s in tqdm(solutions, desc=f"Generation {g+1}/{CMA_GENERATIONS}")]
        es.tell(solutions, [-r for r in rewards])
        es.disp()
        current_best_reward = -es.best.f
        if current_best_reward > best_reward:
            best_reward = current_best_reward
            print(f"新しいベスト報酬: {best_reward:.2f}")
            best_params = torch.tensor(es.best.x, dtype=torch.float32)
            torch.save(best_params, CONTROLLER_PATH)
            print(f"ベストパラメータが {CONTROLLER_PATH} に保存されました。")

    print("Controllerの学習が完了しました。")

if __name__ == "__main__":
    run_train_controller()