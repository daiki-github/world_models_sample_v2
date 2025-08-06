# mdn_rnn.py
import torch
import torch.nn as nn

class MDNRNN(nn.Module):
    # --- 修正点: __init__で引数を受け取る ---
    def __init__(self, latent_dim, action_dim, hidden_units, n_gaussians):
        super(MDNRNN, self).__init__()
        
        # --- 修正点: 引数を使ってクラスのプロパティとして保存 ---
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.n_gaussians = n_gaussians
        
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, (2 * latent_dim + 1) * n_gaussians)

    def forward(self, z, a, hidden=None):
        # 入力の形状: z [batch, seq, 32], a [batch, seq, 3]
        x = torch.cat([z, a], dim=-1)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # MDNのパラメータを計算
        mdn_params = self.fc(lstm_out)
        
        # パラメータを分割
        pi_logits, mu, log_sigma = torch.split(
            mdn_params, [self.n_gaussians, self.n_gaussians * self.latent_dim, self.n_gaussians * self.latent_dim], dim=-1
        )
        
        # log_sigmaが極端な値にならないように値を制限する
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        
        # 活性化関数を適用
        pi = torch.softmax(pi_logits, dim=-1)
        sigma = torch.exp(log_sigma)
        mu = mu.view(-1, mu.size(1), self.n_gaussians, self.latent_dim)
        sigma = sigma.view(-1, sigma.size(1), self.n_gaussians, self.latent_dim)
        
        return pi, mu, sigma, hidden

def mdn_loss_function(y, pi, mu, sigma):
    # y: 実際の次の潜在ベクトル [batch, seq, 32]
    y = y.unsqueeze(2).expand_as(mu) # 比較のために次元を拡張
    
    # ガウス分布の確率密度を計算
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y)
    
    # 確率密度の和を計算し、混合係数を乗算
    log_prob = torch.sum(log_prob, dim=-1) # sum over latent dim
    log_pi = torch.log(pi)
    
    # logsumexpトリックで数値的安定性を確保
    loss = -torch.logsumexp(log_pi + log_prob, dim=-1)
    
    return loss.mean()