# train_mdn_rnn.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import glob

from config import (DEVICE, VAE_PATH, MDNRNN_PATH, DATA_DIR, WEIGHTS_DIR,
                    LATENT_DIM, ACTION_DIM, HIDDEN_UNITS, N_GAUSSIANS,
                    MDNRNN_BATCH_SIZE, MDNRNN_SEQ_LEN, MDNRNN_LEARNING_RATE, MDNRNN_EPOCHS)
from models.vae import VAE
from models.mdn_rnn import MDNRNN, mdn_loss_function

def create_rnn_dataset(vae, observations_mmap, actions, dones):
    """ VAEで観測を潜在ベクトルに変換し、RNN用のデータセットを作成 """
    print("観測データ(memmap)を潜在ベクトルにエンコードしています...")
    
    z_list = []
    with torch.no_grad():
        batch_size = 512
        # ★★★ memmap配列をバッチ処理するため、torch.from_numpyはループ内で行う ★★★
        for i in tqdm(range(0, len(observations_mmap), batch_size), desc="エンコード進捗"):
            # memmapから必要なバッチサイズ分だけを読み込む -> メモリ効率が良い
            obs_batch_np = observations_mmap[i:i+batch_size]
            obs_batch_tensor = torch.from_numpy(obs_batch_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            
            mu, _ = vae.encode(obs_batch_tensor)
            z_list.append(mu.cpu())
    
    z = torch.cat(z_list, dim=0)
    actions = torch.from_numpy(actions)
    dones = torch.from_numpy(dones)

    print("RNN用のシーケンスデータを作成しています...")
    sequences_z, sequences_a, sequences_z_next = [], [], []
    for i in tqdm(range(len(z) - MDNRNN_SEQ_LEN), desc="シーケンス作成進捗"):
        if not dones[i : i + MDNRNN_SEQ_LEN].any():
            sequences_z.append(z[i : i + MDNRNN_SEQ_LEN])
            sequences_a.append(actions[i : i + MDNRNN_SEQ_LEN])
            sequences_z_next.append(z[i+1 : i+1 + MDNRNN_SEQ_LEN])
    
    if not sequences_z:
        raise ValueError("有効なシーケンスが1つも見つかりませんでした。データ収集期間が短すぎる可能性があります。")
        
    return TensorDataset(torch.stack(sequences_z), torch.stack(sequences_a), torch.stack(sequences_z_next))

def run_train_rnn():
    """ MDN-RNNモデルの学習を実行する """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # --- ★★★ データ読み込み部分をmemmap対応に全面的に修正 ★★★ ---
    processed_dir = os.path.join(DATA_DIR, "processed_memmap")
    obs_path = os.path.join(processed_dir, 'observations.mmap')
    actions_path = os.path.join(processed_dir, 'actions.npy')
    dones_path = os.path.join(processed_dir, 'dones.npy')

    try:
        if not os.path.exists(obs_path): raise FileNotFoundError
        print("前処理済みデータを読み込んでいます...")
        actions = np.load(actions_path)
        dones = np.load(dones_path)
        # memmapファイルを開く
        observations_mmap = np.memmap(obs_path, dtype=np.uint8, mode='r', shape=(len(actions), 64, 64, 3))
        print(f"データの準備完了。合計ステップ数: {len(actions)}")
    except FileNotFoundError:
        print(f"前処理済みデータが {processed_dir} に見つかりません。")
        print("先に collect_data.py と preprocess_data.py を実行してください。")
        return
    # --- ★★★ 修正ここまで ★★★ ---

    # --- VAEモデルのロード ---
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"VAEモデルが見つかりません: {VAE_PATH}。先に train_vae.py を実行してください。")
        return
    vae.eval()
    
    # --- データセットの準備 ---
    dataset = create_rnn_dataset(vae, observations_mmap, actions, dones)
    # メモリ解放
    del observations_mmap, actions, dones

    dataloader = DataLoader(dataset, batch_size=MDNRNN_BATCH_SIZE, shuffle=True, num_workers=0) # Windows/Macの安定性のため0に
    
    # --- MDN-RNNの初期化と学習ループ ---
    mdnrnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_units=HIDDEN_UNITS,
        n_gaussians=N_GAUSSIANS
    ).to(DEVICE)
    optimizer = optim.Adam(mdnrnn.parameters(), lr=MDNRNN_LEARNING_RATE)

    # 既存の重みがあればロード
    if os.path.exists(MDNRNN_PATH):
        try:
            mdnrnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))
            print(f"既存のMDN-RNNモデルをロードしました: {MDNRNN_PATH}")
            print("学習を再開します...")
        except Exception as e:
            print(f"MDN-RNNモデルのロードに失敗しました: {e}")
            print("新規に学習を開始します。")
    else:
        print("新規にMDN-RNNの学習を開始します。")

    mdnrnn.train()
    for epoch in range(MDNRNN_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{MDNRNN_EPOCHS}")
        for z_batch, a_batch, z_next_batch in pbar:
            z_batch, a_batch, z_next_batch = z_batch.to(DEVICE), a_batch.to(DEVICE), z_next_batch.to(DEVICE)
            optimizer.zero_grad()
            pi, mu, sigma, _ = mdnrnn(z_batch, a_batch)
            loss = mdn_loss_function(z_next_batch, pi, mu, sigma)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                print("警告: 損失が nan または inf になりました。このバッチをスキップします。")

        if len(pbar) > 0 and total_loss != 0:
            avg_loss = total_loss / len(pbar)
            print(f"Epoch {epoch+1}/{MDNRNN_EPOCHS}, 平均損失: {avg_loss:.4f}")
        else:
             print(f"Epoch {epoch+1}/{MDNRNN_EPOCHS}, 有効な学習ステップがありませんでした。")

    torch.save(mdnrnn.state_dict(), MDNRNN_PATH)
    print(f"学習済みMDN-RNNモデルが {MDNRNN_PATH} に保存されました。")

if __name__ == "__main__":
    run_train_rnn()