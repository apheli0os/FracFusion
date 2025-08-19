import torch
import torch.nn as nn
import torch.nn.functional as F

class FractionalFusionUp(nn.Module):
    """
    分数阶自适应上采样（替代最终 logits 的双线性上采样）
    - 仅放大空间维度，不改变通道数
    - 8方向分数阶核 + 门控高通增强 + 低通残差融合
    返回: y [B,C,H,W], orders [B,8,h,w], gate [B,1,H,W]
    """
    def __init__(self, in_channels, up_factor=4, kernel_size=3, vmax=1.0, hidden=48,
                 mode='both', beta=1.6, tau=0.5, center_bias=2.5, smooth_residual=0.15):
        super().__init__()
        self.k = int(kernel_size)
        self.s = int(up_factor)
        self.vmax = float(vmax)
        self.mode = mode
        self.center = self.k // 2
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.tau = float(tau)
        self.center_bias = float(center_bias)
        self.smooth_residual = float(smooth_residual)

        # 8个方向单位向量
        dirs = torch.tensor([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]], dtype=torch.float32)
        self.register_buffer('dirs', dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8))

        # 预测分数阶次与门控（在类别通道上）
        self.order_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 8, 3, 1, 1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1), nn.Sigmoid()
        )

        # 预计算方向-步长基底 [8,2,M+1,K,K]
        base = []
        with torch.no_grad():
            yy, xx = torch.meshgrid(torch.arange(self.k), torch.arange(self.k), indexing='ij')
            rel = torch.stack([yy - self.center, xx - self.center], dim=-1).float()
            max_m = self.center
            for d in range(8):
                v = self.dirs[d]
                proj = rel[..., 0] * v[1] + rel[..., 1] * v[0]
                step = proj.round().clamp(min=-max_m, max=max_m).int()
                pos_masks, neg_masks = [], []
                for m in range(max_m + 1):
                    pos_masks.append((step == m).float())
                    neg_masks.append((step == -m).float())
                base.append(torch.stack(pos_masks, 0))
                base.append(torch.stack(neg_masks, 0))
        dir_step_base = torch.stack([torch.stack(base[2*i:2*i+2], 0) for i in range(8)], 0)
        self.register_buffer('dir_step_base', dir_step_base)

    @staticmethod
    def gl_coeffs(v, M):
        # Grunwald–Letnikov 系数
        if v.dim() == 3:
            v = v.unsqueeze(1)
        B, _, H, W = v.shape
        coeffs = [torch.ones(B,1,H,W, device=v.device, dtype=v.dtype)]
        w = coeffs[0]
        for m in range(1, M+1):
            w = w * (v - m + 1.0)/m * (-1.0)
            coeffs.append(w)
        return torch.cat(coeffs, dim=1)  # [B,M+1,H,W]

    def _direction_kernel(self, v_dir, d):
        M = self.center
        coeff = self.gl_coeffs(v_dir, M)  # [B,M+1,H,W]
        base_pos = self.dir_step_base[d,0]  # [M+1,K,K]
        base_neg = self.dir_step_base[d,1]  # [M+1,K,K]
        ker = (coeff[:, :, None, None, ...] *
               (base_pos[None, :, :, :, None, None] - base_neg[None, :, :, :, None, None])).sum(1)
        return ker  # [B,K,K,H,W]

    def _build_kernels(self, v):
        # 8方向取均值
        ks = [self._direction_kernel(v[:, d:d+1], d) for d in range(8)]
        return torch.stack(ks, 0).mean(0)  # [B,K,K,H,W]

    def forward(self, x):
        B, C, h, w = x.shape
        raw_v = self.order_head(x)
        g = self.gate(x)                         # [B,1,h,w]
        v = self.vmax * torch.tanh(raw_v)        # [B,8,h,w]
        ker = self._build_kernels(v)             # [B,K,K,h,w]
        K, pad = self.k, self.center

        # 低通权重（中心偏置 + 温度 softmax）
        mask_pre = ker.view(B, K*K, h, w)
        center_idx = self.center * self.k + self.center
        mask_pre[:, center_idx:center_idx+1] = mask_pre[:, center_idx:center_idx+1] + self.center_bias
        mask_lp = torch.softmax(mask_pre / self.tau, dim=1)  # [B,K^2,h,w]

        # 邻域展开
        xp = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        patches = F.unfold(xp, kernel_size=K, stride=1, padding=0).view(B, C, K*K, h, w)

        # 放大空间维（仅空间，保留通道）
        if self.s > 1:
            neigh   = patches.repeat_interleave(self.s, dim=3).repeat_interleave(self.s, dim=4)  # [B,C,K^2,H,W]
            mask_lp = mask_lp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)  # [B,K^2,H,W]
            g_up    = g.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)        # [B,1,H,W]
            x_up    = F.interpolate(x, scale_factor=self.s, mode='bilinear', align_corners=False) # [B,C,H,W]
        else:
            neigh, g_up, x_up = patches, g, x

        # 低通输出 + 边缘自适应残差（降低模糊）
        y_lp_out = (neigh * mask_lp[:, None]).sum(dim=2)
        lambda_eff = self.smooth_residual * (1.0 - g_up)
        y_lp = x_up + lambda_eff * (y_lp_out - x_up)

        if self.mode == 'lowpass':
            return y_lp, v, g_up

        # 高通（零直流 + L1归一）
        ker_hp = ker - ker.mean(dim=(1, 2), keepdim=True)
        ker_hp = ker_hp / (ker_hp.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        mask_hp = ker_hp.view(B, K*K, h, w)
        if self.s > 1:
            mask_hp = mask_hp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)
        y_hp = (neigh * mask_hp[:, None]).sum(dim=2)

        if self.mode == 'highpass':
            return y_hp, v, g_up

        # 融合
        y = y_lp + torch.tanh(self.beta) * g_up * y_hp
        return y, v, g_up