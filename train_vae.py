# train_vae.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

from config import (DEVICE, VAE_PATH, SERIES_PATH, WEIGHTS_DIR,
                    LATENT_DIM, VAE_BATCH_SIZE, VAE_LEARNING_RATE, VAE_EPOCHS)
from models.vae import VAE, vae_loss_function

def preprocess_observations(obs_array):
    """ 観測データをVAEの入力形式に前処理 """
    processed = np.array([cv2.resize(obs, (64, 64)) for obs in obs_array], dtype=np.float32)
    processed = processed / 255.0
    processed = torch.from_numpy(processed).permute(0, 3, 1, 2)
    return processed

def run_train_vae():
    """ VAEモデルの学習を実行する """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    if not os.path.exists(SERIES_PATH):
        print(f"データファイルが見つかりません: {SERIES_PATH}。先に collect_data.py を実行してください。")
        return
        
    data = np.load(SERIES_PATH)
    observations = data['observations']
    
    print("観測データの前処理中...")
    dataset = TensorDataset(preprocess_observations(observations))
    dataloader = DataLoader(dataset, batch_size=VAE_BATCH_SIZE, shuffle=True)
    
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=VAE_LEARNING_RATE)

    print("VAEの学習を開始します...")
    vae.train()
    for epoch in range(VAE_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{VAE_EPOCHS}")
        for batch in pbar:
            obs_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(obs_batch)
            loss = vae_loss_function(recon_batch, obs_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item() / len(obs_batch):.4f}")
            
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{VAE_EPOCHS}, 平均損失: {avg_loss:.4f}")

    torch.save(vae.state_dict(), VAE_PATH)
    print(f"学習済みVAEモデルが {VAE_PATH} に保存されました。")

if __name__ == "__main__":
    run_train_vae()