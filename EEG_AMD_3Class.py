from torch import nn
import torch
from common import DDI
import torch.nn.functional as F
import torch.optim as optim
import math


# --- 时序编码器 ---
class TemporalEncoder(nn.Module):
    def __init__(self, T=400, F=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=4, padding=12), nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, eeg):  # eeg: [B,N,T_seg]
        B, N, T = eeg.shape
        x = eeg.reshape(B*N, 1, T)
        x = self.net(x).squeeze(-1)  # [B*N, F]
        return x.reshape(B, N, -1)   # [B,N,F]


# --- 多频段GCN层 ---
class GraphConv(nn.Module):
    def __init__(self, Fin, Fout, K=4, shared_W=True, use_attn=True):
        super().__init__()
        self.K = K
        self.use_attn = use_attn
        if shared_W:
            self.W = nn.Linear(Fin, Fout, bias=False)
        else:
            self.W_list = nn.ModuleList([nn.Linear(Fin, Fout, bias=False) for _ in range(K)])
            self.W = None
        if use_attn:
            self.q = nn.Parameter(torch.zeros(K))
        else:
            self.register_buffer('alpha_fixed', torch.ones(K)/K)
        self.bn = nn.BatchNorm1d(Fout)

    def forward(self, X, A_hat):

        alpha = torch.softmax(self.q, dim=0) if self.use_attn else self.alpha_fixed
        outs = []
        for k in range(self.K):
            AX = torch.matmul(A_hat[:,k], X)
            Hk = self.W(AX) if self.W is not None else self.W_list[k](AX)
            outs.append(alpha[k] * Hk)
        H = sum(outs)
        H = self.bn(H.transpose(1,2)).transpose(1,2)
        return F.relu(H)


class GraphPLVEncoder(nn.Module):
    def __init__(self, T_seg=400, Fin=64, Hidden=64, K=4):
        super().__init__()
        self.encoder = TemporalEncoder(T=T_seg, F=Fin)
        self.gcn1 = GraphConv(Fin, Hidden, K=K, shared_W=True, use_attn=True)
        self.gcn2 = GraphConv(Hidden, Fin, K=K, shared_W=True, use_attn=True)

    def forward(self, eeg_seg, plv_seg):
        """
        eeg_seg: [B, Seg, N, T_seg]  例如 [64,15,19,400]
        plv_seg: [B, K, Seg, N, N]  例如 [64,4,15,19,19]
        返回: [B, Seg, Hidden]
        """
        B, Seg, N, T = eeg_seg.shape


        out_seg = []
        for s in range(Seg):
            X = self.encoder(eeg_seg[:,s])        # [B,N,F]
            A_hat = plv_seg[:,:,s]                # [B,K,N,N]
            H = self.gcn1(X, A_hat)
            H = self.gcn2(H, A_hat)

            g = H.mean(dim=1)                     # [B, Hidden]
            out_seg.append(g.unsqueeze(1))        # [B,1,Hidden]

        out = torch.cat(out_seg, dim=1)          # [B,Seg,Hidden]
        return out


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=64, seq_len=15, num_heads=4, ff_dim=128, num_layers=2, num_classes=3, dropout=0.1, use_cls=True):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.use_cls = use_cls

        # 如果使用 cls token，则 pos_embedding 长度为 seq_len + 1
        pos_len = seq_len + 1 if use_cls else seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, pos_len, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, cls_token=None):
        """
        x: [B, Seq, F]
        cls_token: optional tensor of shape [1,1,F] (module-level param expanded per-batch)
        returns: [B, F] (pooled representation)
        """
        B, Seq, F = x.shape
        if self.use_cls:
            assert cls_token is not None, "use_cls=True but no cls_token provided"
            # expand cls token to batch
            cls_expanded = cls_token.expand(B, -1, -1)  # [B,1,F]
            x = torch.cat([cls_expanded, x], dim=1)     # [B, Seq+1, F]

        # add positional embedding (slice to length)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)  # [B, Seq(+1), F]

        if self.use_cls:
            # read cls token
            out = x[:, 0, :]  # [B, F]
        else:
            out = x.mean(dim=1)
        return out


# ------------------ 在主模型里使用新的 DDI 与新的 Transformer ------------------
class EEG_AMD_3Class(nn.Module):
    def __init__(self, n_block=2, patch=3, input_dim=64, seq_len=15, num_heads=4, ff_dim=128, num_layers=4,
                 num_classes=3, dropout=0.2, alpha=0.0, n_classes=3, norm=True, layernorm=True, T_seg=400, Fin=64, Hidden=128, K=4):
        super(EEG_AMD_3Class, self).__init__()
        self.multiBand_GCN = GraphPLVEncoder(T_seg=T_seg, Fin=Fin, Hidden=Hidden, K=K)
        self.seq_len = seq_len
        self.norm = norm
        # class token used by transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        self.fc_blocks = nn.ModuleList([
            DDI([self.seq_len, input_dim], dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
            for _ in range(n_block)
        ])

        # transformer expects input_dim features; use cls_token we defined
        self.transformerClassifier = EEGTransformer(input_dim=input_dim, seq_len=seq_len, num_heads=num_heads,
                                                              ff_dim=ff_dim, num_layers=num_layers,
                                                              num_classes=num_classes, dropout=dropout, use_cls=True)

        # final classifier
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, X, A_hat):
        X = X.view(-1, 19, 15, 400)
        X = X.permute(0,2,1,3)
        X_pro = self.multiBand_GCN(X, A_hat)  # [B, Seq, Hidden]  (注意：Hidden vs input_dim 要匹配)
        # X_pro = X.mean(dim=2)

        # apply DDI blocks (they expect [B, Seq, F])
        for fc in self.fc_blocks:
            X_pro = fc(X_pro)  # [B, Seq, input_dim]

        # Transformer (supply cls token)
        X_trans = self.transformerClassifier(X_pro, cls_token=self.cls_token)  # [B, input_dim]
        # X_trans = X_pro.mean(dim=1)   # 屏蔽Transformer

        out = self.classifier(X_trans)  # [B, n_classes]
        return out


def get_optimizer_and_scheduler(model, base_lr=1e-3, weight_decay=5e-4, total_epochs=100, warmup_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        # linear warmup
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine decay after warmup, epochs indexed from 0
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def getModel():
    model = EEG_AMD_3Class(input_dim=64)
    return model


if __name__ == '__main__':
    B, S, N, T = 64, 15, 19, 400
    eeg = torch.randn(B, S, N, T)

    A_hat = torch.rand(B, 4, S, N, N)
    model = EEG_AMD_3Class()
    out = model(eeg, A_hat)  # logits: [8, 3]
    print(out.shape)
