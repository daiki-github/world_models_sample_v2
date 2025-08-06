# train_mdn_rnn.py
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import cv2
from tqdm import tqdm

from config import (DEVICE, VAE_PATH, MDNRNN_PATH, SERIES_PATH, WEIGHTS_DIR,
                    LATENT_DIM, ACTION_DIM, HIDDEN_UNITS, N_GAUSSIANS,
                    MDNRNN_BATCH_SIZE, MDNRNN_SEQ_LEN, MDNRNN_LEARNING_RATE, MDNRNN_EPOCHS)
from models.vae import VAE
from models.mdn_rnn import MDNRNN, mdn_loss_function

def create_rnn_dataset(vae, data):
    # (この関数の中身は変更なし)
    obs_list = [cv2.resize(obs, (64, 64)) for obs in data['observations']]
    obs_tensor = torch.from_numpy(np.array(obs_list)).float().permute(0, 3, 1, 2) / 255.0
    
    z_list = []
    with torch.no_grad():
        batch_size = 512
        for i in tqdm(range(0, len(obs_tensor), batch_size), desc="観測データをエンコード中"):
            obs_batch = obs_tensor[i:i+batch_size].to(DEVICE)
            mu, _ = vae.encode(obs_batch)
            z_list.append(mu.cpu())
    
    z = torch.cat(z_list, dim=0)
    actions = torch.from_numpy(data['actions'])
    dones = torch.from_numpy(data['dones'])

    sequences_z, sequences_a, sequences_z_next = [], [], []
    for i in range(len(z) - MDNRNN_SEQ_LEN):
        if not dones[i : i + MDNRNN_SEQ_LEN].any():
            sequences_z.append(z[i : i + MDNRNN_SEQ_LEN])
            sequences_a.append(actions[i : i + MDNRNN_SEQ_LEN])
            sequences_z_next.append(z[i+1 : i+1 + MDNRNN_SEQ_LEN])
            
    return TensorDataset(torch.stack(sequences_z), torch.stack(sequences_a), torch.stack(sequences_z_next))

def run_train_rnn():
    """ MDN-RNNモデルの学習を実行する """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"VAEモデルが見つかりません: {VAE_PATH}。先に train_vae.py を実行してください。")
        return
    vae.eval()

    data = np.load(SERIES_PATH)
    dataset = create_rnn_dataset(vae, data)
    dataloader = DataLoader(dataset, batch_size=MDNRNN_BATCH_SIZE, shuffle=True)
    
    mdnrnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_units=HIDDEN_UNITS,
        n_gaussians=N_GAUSSIANS
    ).to(DEVICE)
    optimizer = optim.Adam(mdnrnn.parameters(), lr=MDNRNN_LEARNING_RATE)

    print("MDN-RNNの学習を開始します...")
    mdnrnn.train()
    for epoch in range(MDNRNN_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{MDNRNN_EPOCHS}")
        for z_batch, a_batch, z_next_batch in pbar:
            z_batch, a_batch, z_next_batch = z_batch.to(DEVICE), a_batch.to(DEVICE), z_next_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            pi, mu, sigma, _ = mdnrnn(z_batch, a_batch)
            loss = mdn_loss_function(z_next_batch, pi, mu, sigma)
            
            # --- 修正点: lossがnanでない場合のみ逆伝播と更新を行う ---
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                # --- 修正点: 勾配クリッピングを追加 ---
                torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0) # 上限値1.0でクリッピング
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                print("警告: 損失が nan または inf になりました。このバッチをスキップします。")

        # 1エポックで一度も有効な学習がなければ、損失が0になる可能性を考慮
        if len(dataloader) > 0 and total_loss != 0:
            avg_loss = total_loss / len(pbar)
            print(f"Epoch {epoch+1}/{MDNRNN_EPOCHS}, 平均損失: {avg_loss:.4f}")
        else:
             print(f"Epoch {epoch+1}/{MDNRNN_EPOCHS}, 有効な学習ステップがありませんでした。")


    torch.save(mdnrnn.state_dict(), MDNRNN_PATH)
    print(f"学習済みMDN-RNNモデルが {MDNRNN_PATH} に保存されました。")

if __name__ == "__main__":
    run_train_rnn()