import torch
import torch.nn.functional as F
import numpy as np


class SelfAttention2d(torch.nn.Module):
    def __init__(self, input_size, n_heads, dim_per_head):
        super().__init__()
        self.n_channels = n_channels = input_size[0]
        self.height = input_size[1]
        self.width = input_size[2]
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head

        n_pos_emb_channels = 4
        self.pos_emb = torch.nn.Parameter(
            torch.randn(1, n_pos_emb_channels, input_size[1], input_size[2]) \
            / np.sqrt(np.prod(input_size)))

        self.K_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head, kernel_size=1, padding=0)
        self.Q_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head, kernel_size=1, padding=0)
        self.V_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head, kernel_size=1, padding=0)

        self.linear = torch.nn.Conv2d(n_heads * dim_per_head, n_channels, kernel_size=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()

        timesteps = H * W

        x_plus_pos = torch.cat([x, self.pos_emb.repeat(B, 1, 1, 1)], dim=1)
        K = self.K_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, timesteps)
        Q = self.Q_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, timesteps)
        V = self.V_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, timesteps)

        QK = torch.matmul(Q.transpose(-1, -2), K)
        # assert QK.size() == (B, self.n_heads, timesteps, timesteps)
        # QK_old = torch.sum(
        #     Q.view(B, self.n_heads, self.dim_per_head, timesteps, 1) *
        #     K.view(B, self.n_heads, self.dim_per_head, 1, timesteps), dim=2
        # )
        # print(torch.mean(torch.abs(QK - QK_old)))

        QK = QK.view(B, self.n_heads, timesteps, timesteps)

        QK = F.softmax(QK / np.sqrt(timesteps), dim=-1)
        A = torch.matmul(V, QK.transpose(-1, -2))
        # assert A.size() == (B, self.n_heads, self.dim_per_head, timesteps)

        # A_old = torch.sum(
        #     QK.view(B, self.n_heads, 1, timesteps, timesteps) *
        #     V.view(B, self.n_heads, self.dim_per_head, 1, timesteps), dim=-1)

        A_flat = A.view(B, self.n_heads * self.dim_per_head, H, W)

        return x + self.linear(A_flat)


class SparseSelfAttention2d(torch.nn.Module):

    def __init__(self, input_size, n_heads, dim_per_head):
        super().__init__()
        self.n_channels = n_channels = input_size[0]
        self.height = input_size[1]
        self.width = input_size[2]
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head

        n_pos_emb_channels = 4
        self.pos_emb = torch.nn.Parameter(
            torch.randn(1, n_pos_emb_channels, input_size[1], input_size[2]) \
            / np.sqrt(np.prod(input_size)))

        self.K_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head,
            kernel_size=1, padding=0)
        self.Q_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head,
            kernel_size=1, padding=0)
        self.V_nn = torch.nn.Conv2d(
            n_channels + n_pos_emb_channels, n_heads * dim_per_head,
            kernel_size=1, padding=0)

        self.linear = torch.nn.Conv2d(n_heads * dim_per_head, n_channels, kernel_size=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()

        x_plus_pos = torch.cat([x, self.pos_emb.repeat(B, 1, 1, 1)], dim=1)
        K = self.K_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, H, W)
        Q = self.Q_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, H, W)
        V = self.V_nn(x_plus_pos).view(B, self.n_heads, self.dim_per_head, H, W)

        QK_W = torch.matmul(
            Q.permute(0, 1, 3, 4, 2),
            K.permute(0, 1, 3, 2, 4)
        )
        # QK_W_old = torch.sum(
        #     Q.view(B, self.n_heads, self.dim_per_head, H, W, 1) *
        #     K.view(B, self.n_heads, self.dim_per_head, H, 1, W), dim=2
        # )
        # assert QK_W.size() == (B, self.n_heads, H, W, W)

        QK_H = torch.matmul(
            Q.permute(0, 1, 4, 3, 2),
            K.permute(0, 1, 4, 2, 3),
        ).permute(0, 1, 3, 2, 4)

        # QK_H_old = torch.sum(
        #     Q.view(B, self.n_heads, self.dim_per_head, H, 1, W) *
        #     K.view(B, self.n_heads, self.dim_per_head, 1, H, W), dim=2
        # ).permute(0, 1, 2, 4, 3)
        # print(torch.mean(torch.abs(QK_H - QK_H_old)))
        # assert QK_H.size() == (B, self.n_heads, H, W, H)

        # Shape: [B, heads, dims, H, W, W+H]
        QK = torch.cat([QK_W, QK_H], dim=-1)
        # assert QK.size() == (B, self.n_heads, H, W, W + H)

        QK = F.softmax(QK / np.sqrt(W+H), dim=-1)

        QK_W = QK[..., :W]
        # assert QK_W.size() == (B, self.n_heads, H, W, W)

        QK_H = QK[..., W:]
        # assert QK_H.size() == (B, self.n_heads, H, W, H)

        # V_W = V.view(B, self.n_heads, self.dim_per_head, H, 1, W)
        # V_H = V.permute(0, 1, 2, 4, 3).view(B, self.n_heads, self.dim_per_head, 1, W, H)
        # A_old = torch.sum(
        #     QK_W.view(B, self.n_heads, 1, H, W, W) * V_W, dim=-1) \
        #     + torch.sum(
        #     QK_H.view(B, self.n_heads, 1, H, W, H) * V_H, dim=-1
        # )

        A_W = torch.matmul(QK_W, V.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        QK_H = QK_H.permute(0, 1, 3, 2, 4)
        V_H = V.permute(0, 1, 4, 3, 2)
        A_H = torch.matmul(QK_H, V_H).permute(0, 1, 4, 3, 2)

        A = (A_W + A_H).contiguous()

        # assert A.size() == (B, self.n_heads, self.dim_per_head, H, W)

        A = A.view(B, self.n_heads * self.dim_per_head, H, W)

        return x + self.linear(A)

if __name__ == '__main__':
    B, C, H, W = 7, 3, 13, 11
    inp = torch.rand(size=(B, C, H, W))

    layer = SparseSelfAttention2d(input_size=(C, H, W), n_heads=19, dim_per_head=2)

    out = layer(inp)

