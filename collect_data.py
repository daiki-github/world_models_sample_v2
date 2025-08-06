# collect_data.py
import gymnasium
import numpy as np
import os
from tqdm import tqdm
from config import DATA_DIR, SERIES_PATH, COLLECT_EPISODES

def run_collection():
    """ ランダムな方策でデータを収集し、保存する """
    print(f"--- データ収集中: {COLLECT_EPISODES} エピソード ---")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    env = gymnasium.make("CarRacing-v3", continuous=True)
    
    observations = []
    actions = []
    dones = []

    for i in tqdm(range(COLLECT_EPISODES), desc="Collecting Data"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() # ランダムな行動
            
            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dones.append(done)
    
    env.close()

    np.savez_compressed(
        SERIES_PATH,
        observations=np.array(observations, dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        dones=np.array(dones, dtype=bool)
    )
    print(f"データが {SERIES_PATH} に保存されました。")
    print(f"合計ステップ数: {len(observations)}")

# このファイル単体で実行された場合に、上記の関数を呼び出す
if __name__ == "__main__":
    run_collection()