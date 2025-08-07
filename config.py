# config.py
import torch

# --- 全体設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
WEIGHTS_DIR = "weights"
SERIES_PATH = f"{DATA_DIR}/series_data.npz"

# --- データ収集設定 ---
COLLECT_EPISODES = 1000 # 収集するエピソード数
DATA_CHUNK_SIZE = 100 # この行を追加: 100エピソード毎にファイルを保存する

# --- VAE設定 ---
VAE_PATH = f"{WEIGHTS_DIR}/vae.pth"
LATENT_DIM = 32
VAE_BATCH_SIZE = 256
VAE_LEARNING_RATE = 0.0001
VAE_EPOCHS = 30 # データセットのサイズに応じて調整

# --- MDN-RNN設定 ---
MDNRNN_PATH = f"{WEIGHTS_DIR}/mdn_rnn.pth"
ACTION_DIM = 3 # CarRacing: [steering, gas, brake]
HIDDEN_UNITS = 256
N_GAUSSIANS = 5
MDNRNN_BATCH_SIZE = 256 # シーケンスをバッチにする
MDNRNN_SEQ_LEN = 32    # 1つの学習で見るシーケンス長
MDNRNN_LEARNING_RATE = 0.0003 # 学習率
MDNRNN_EPOCHS = 50 # データセットのサイズに応じて調整

# --- Controller設定 ---
CONTROLLER_PATH = f"{WEIGHTS_DIR}/controller.pth"
# CMA-ES用の設定
CMA_POPULATION_SIZE = 32 # 1世代あたりの個体数
CMA_SIGMA = 0.5          # 初期標準偏差
CMA_GENERATIONS = 500    # 進化の世代数