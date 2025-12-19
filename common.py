import torch
import torch.nn as nn
import torch.nn.functional as Fun

import math

# ------------------ 重写 DDI（接受 [B, Seq, F]） ------------------
class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=3, alpha=0.0, layernorm=True):
        """
        input_shape: [seq_len, feature_dim]
        x expected shape: [B, Seq, F]
        """
        super(DDI, self).__init__()
        self.seq_len = input_shape[0]
        self.feat_dim = input_shape[1]
        self.patch = patch
        self.alpha = alpha
        self.n_history = 1  # 保持原语义

        # feed-forward used when alpha > 0
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.feat_dim))
            # apply MLP on feature dim (last dim)
            self.fc_block = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # aggregation from history window -> new patch
        hist_window = self.n_history * self.patch
        self.agg = nn.Linear(hist_window, self.patch, bias=True)  # operate across time-window
        self.dropout_t = nn.Dropout(dropout)

        # layer norms for stability
        self.layernorm = layernorm
        if self.layernorm:
            # normalize per-token (feature-wise)
            self.norm_token = nn.LayerNorm(self.feat_dim)
        # norm before fc_block (if used)
        if self.alpha > 0.0:
            self.norm_before_ff = nn.LayerNorm(self.feat_dim)

    def forward(self, x):
        # x: [B, Seq, F]
        B, Seq, F = x.shape
        # assert Seq == self.seq_len and F == self.feat_dim, f"Expected seq_len={self.seq_len}, feat_dim={self.feat_dim}"

        # 初始保留前 hist_window tokens
        hist_window = self.n_history * self.patch
        output = torch.zeros_like(x)

        # 直接拷贝最前 hist_window tokens
        output[:, :hist_window, :] = x[:, :hist_window, :].clone()

        # 逐 patch 滑动生成后续 patch
        for i in range(hist_window, self.seq_len, self.patch):
            # 取历史窗口
            prev = output[:, i - hist_window:i, :]  # [B, hist_window, F]

            # 线性聚合（要换成 [B, F, hist_window]）
            prev_t = prev.permute(0, 2, 1)  # [B, F, hist_window]
            aggregated = self.agg(prev_t)  # [B, F, patch]
            aggregated = Fun.gelu(aggregated)
            aggregated = self.dropout_t(aggregated)
            aggregated = aggregated.permute(0, 2, 1)  # [B, patch, F]

            # 对应 slice
            cur_slice = x[:, i:i + self.patch, :]  # [B, patch, F]
            res = aggregated + cur_slice  # 残差融合

            tmp = res
            if self.alpha > 0.0:
                tmp = self.norm_before_ff(tmp)
                tmp = self.fc_block(tmp)  # [B, patch, F]

            # 更新到 output（自回归式）
            output[:, i:i + self.patch, :] = res + self.alpha * tmp
        return output

