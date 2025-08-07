# collect_data.py
import gymnasium
import numpy as np
import os
from tqdm import tqdm
from config import DATA_DIR, COLLECT_EPISODES, DATA_CHUNK_SIZE # ★★★ DATA_CHUNK_SIZEをインポート

def run_collection():
    """ ランダムな方策でデータを収集し、チャンクごとに保存する """
    print(f"--- データ収集中: {COLLECT_EPISODES} エピソードを {DATA_CHUNK_SIZE} ごとのチャンクで保存 ---")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    env = gymnasium.make("CarRacing-v3", continuous=True)
    
    observations, actions, dones = [], [], []
    file_counter = 0

    for i in tqdm(range(COLLECT_EPISODES), desc="Collecting Data"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observations.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dones.append(done)
        
        # ★★★ ここからチャンク保存のロジック ★★★
        # チャンクサイズに達したか、最後のエピソードであればファイルを保存
        if (i + 1) % DATA_CHUNK_SIZE == 0 or (i + 1) == COLLECT_EPISODES:
            chunk_path = os.path.join(DATA_DIR, f"series_data_part_{file_counter}.npz")
            print(f"\nチャンクを保存中: {chunk_path} ({len(observations)} ステップ)")
            
            np.savez_compressed(
                chunk_path,
                observations=np.array(observations, dtype=np.uint8),
                actions=np.array(actions, dtype=np.float32),
                dones=np.array(dones, dtype=bool)
            )
            
            # メモリを解放するためにリストをクリア
            observations.clear()
            actions.clear()
            dones.clear()
            file_counter += 1
            
    env.close()
    print(f"\nデータ収集が完了しました。合計 {file_counter} 個のファイルが生成されました。")

if __name__ == "__main__":
    run_collection()