# controller.py
import torch
import torch.nn as nn

class Controller(nn.Module):
    # --- 修正点: __init__で引数を受け取る ---
    def __init__(self, latent_dim, hidden_units, action_dim):
        super(Controller, self).__init__()
        
        self.fc = nn.Linear(latent_dim + hidden_units, action_dim)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        # CarRacingの行動空間: [ステアリング(-1~1), アクセル(0~1), ブレーキ(0~1)]
        action = self.fc(x)
        
        # 活性化関数で各値を範囲内に収める
        steering = torch.tanh(action[:, 0])
        gas = torch.sigmoid(action[:, 1])
        brake = torch.sigmoid(action[:, 2])
        
        return torch.stack([steering, gas, brake], dim=-1)