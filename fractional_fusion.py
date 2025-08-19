import torch
import torch.nn as nn
import torch.nn.functional as F


class FractionalFusionUp(nn.Module):
    """
    分数阶自适应上采样（用于解码器特征融合）
    - 预测8方向分数阶次 v ∈ [-vmax, vmax]
    - 边界门控 g ∈ [0,1]（g大→更偏向高通增强，g小→低通平滑）
    - 基于GL系数生成K×K方向核，聚合为二维动态核
    - 使用CARAFE式重组：unfold邻域 × 动态核 → 上采样

    输出：
      up: 上采样后的特征 [B,C,H*s,W*s]
      orders: 分数阶次 [B,8,H,W]
      gate: 边界门控 [B,1,H,W]
    """

    def __init__(self, in_channels, up_factor=2, kernel_size=5, vmax=1.2, hidden=64, mode='both', beta=0.6):
        super().__init__()
        assert mode in ['lowpass', 'highpass', 'both']
        self.k, self.s, self.vmax, self.mode = kernel_size, up_factor, vmax, mode
        self.center = kernel_size // 2
        self.beta = nn.Parameter(torch.tensor(float(beta)))  # 高通强度

        # 方向单位向量（8方向）
        dirs = torch.tensor(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=torch.float32
        )
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('dirs', dirs)

        # 阶次与门控
        self.order_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 8, 3, 1, 1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1), nn.Sigmoid()
        )

        # 预计算“方向-步长-位置”基底 [8,2,M+1,K,K]
        base = []
        with torch.no_grad():
            yy, xx = torch.meshgrid(torch.arange(self.k), torch.arange(self.k), indexing='ij')
            rel = torch.stack([yy - self.center, xx - self.center], dim=-1).float()  # [K,K,2]
            max_m = self.center
            for d in range(8):
                v = self.dirs[d]  # 统一使用 self.dirs
                # 用(y,x)与(dir_y,dir_x)点乘作为步长投影
                proj = rel[..., 0] * v[1] + rel[..., 1] * v[0]
                step = proj.round().clamp(min=-max_m, max=max_m).int()
                pos_masks, neg_masks = [], []
                for m in range(max_m + 1):
                    pos_masks.append((step == m).float())
                    neg_masks.append((step == -m).float())
                base.append(torch.stack(pos_masks, 0))
                base.append(torch.stack(neg_masks, 0))
        dir_step_base = torch.stack([torch.stack(base[2 * i:2 * i + 2], 0) for i in range(8)], 0)
        self.register_buffer('dir_step_base', dir_step_base)

    @staticmethod
    def gl_coeffs(v, M):
        """
        Grünwald–Letnikov 系数递推：W_m = W_{m-1} * (v - m + 1)/m * (-1)
        v: [B,1,H,W] → [B,M+1,H,W]
        """
        if v.dim() == 3:
            v = v.unsqueeze(1)
        B, _, H, W = v.shape
        coeffs = [torch.ones(B, 1, H, W, device=v.device, dtype=v.dtype)]
        w = coeffs[0]
        for m in range(1, M + 1):
            w = w * (v - m + 1.0) / m * (-1.0)
            coeffs.append(w)
        return torch.cat(coeffs, dim=1)  # [B,M+1,H,W]

    def _direction_kernel(self, v_dir, d):
        """按方向d生成二维核: [B,K,K,H,W]"""
        M = self.center
        coeff = self.gl_coeffs(v_dir, M)  # [B,M+1,H,W]
        base_pos = self.dir_step_base[d, 0]  # [M+1,K,K]
        base_neg = self.dir_step_base[d, 1]  # [M+1,K,K]
        ker = (coeff[:, :, None, None, ...] *
               (base_pos[None, :, :, :, None, None] - base_neg[None, :, :, :, None, None])).sum(1)
        return ker  # [B,K,K,H,W]

    def _build_kernels(self, v):
        """聚合8方向，得到 [B,K,K,H,W] 的二维核（可含正负）"""
        kernels = []
        for d in range(8):
            vd = v[:, d:d + 1]
            ker = self._direction_kernel(vd, d)
            kernels.append(ker)
        ker = torch.stack(kernels, 0).mean(0)  # [B,K,K,H,W]
        return ker

    def forward(self, x, skip=None):
        """
        x: [B,C,H,W]
        return: up [B,C,H*s,W*s], orders [B,8,H,W], gate_up [B,1,H*s,W*s]
        """
        B, C, H, W = x.shape

        # 预测阶次与门控
        raw_v = self.order_head(x)             # [B,8,H,W]
        g = self.gate(x)                       # [B,1,H,W]
        v = self.vmax * torch.tanh(raw_v)      # [-vmax, vmax]

        # 二维核 [B,K,K,H,W]
        ker = self._build_kernels(v)
        K, pad = self.k, self.center

        # 低通权重：中心先验 + softmax 温度
        tau = getattr(self, 'tau', 0.6)
        center_bias = getattr(self, 'center_bias', 2.0)
        mask_pre = ker.view(B, K*K, H, W)
        center_idx = self.center * self.k + self.center
        mask_pre[:, center_idx:center_idx+1] = mask_pre[:, center_idx:center_idx+1] + center_bias
        mask_lp = torch.softmax(mask_pre / tau, dim=1)         # [B,K^2,H,W]

        # 邻域展开 -> [B,C,K^2,H,W]
        xp = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        patches = F.unfold(xp, kernel_size=K, stride=1, padding=0).view(B, C, K*K, H, W)

        # 仅放大空间维（避免对 K^2 维插值）
        if self.s > 1:
            neigh   = patches.repeat_interleave(self.s, dim=3).repeat_interleave(self.s, dim=4)  # [B,C,K^2,Hs,Ws]
            mask_lp = mask_lp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)  # [B,K^2,Hs,Ws]
            g_up    = g.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)        # [B,1,Hs,Ws]
            x_up    = F.interpolate(x, scale_factor=self.s, mode='bilinear', align_corners=False)
            Hs, Ws = H * self.s, W * self.s
        else:
            neigh = patches
            mask_lp = mask_lp
            g_up = g
            x_up = x
            Hs, Ws = H, W

        # 低通输出 + 边缘自适应残差（减弱模糊）
        y_lp_out = (neigh * mask_lp[:, None]).sum(dim=2)       # [B,C,Hs,Ws]
        smooth_residual = getattr(self, 'smooth_residual', 0.2)
        lambda_eff = smooth_residual * (1.0 - g_up)            # [B,1,Hs,Ws]
        y_lp = x_up + lambda_eff * (y_lp_out - x_up)

        if getattr(self, 'mode', 'both') == 'lowpass':
            return y_lp, v, g_up

        # 高通分支：零直流 + L1 归一
        ker_hp = ker - ker.mean(dim=(1, 2), keepdim=True)
        ker_hp = ker_hp / (ker_hp.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        mask_hp = ker_hp.view(B, K*K, H, W)
        if self.s > 1:
            mask_hp = mask_hp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)  # [B,K^2,Hs,Ws]
        y_hp = (neigh * mask_hp[:, None]).sum(dim=2)           # [B,C,Hs,Ws]

        if getattr(self, 'mode', 'both') == 'highpass':
            return y_hp, v, g_up

        # 混合输出：y = y_lp + alpha * g * y_hp
        beta = self.beta
        alpha = torch.tanh(beta if torch.is_tensor(beta) else torch.tensor(beta, device=x.device, dtype=x.dtype))
        y = y_lp + alpha * g_up * y_hp
        return y, v, g_up
    
    def apply_fractional_operator(self, x, orders=None, return_parts=False):
        """
        同分辨率分数阶处理（用于可视化/增强）
        return_parts=True 时返回 (y, y_lp, y_hp, orders, gate)
        """
        B, C, H, W = x.shape
        if orders is None:
            raw_v = self.order_head(x)
            orders = self.vmax * torch.tanh(raw_v)
        g = self.gate(x)                                    # [B,1,H,W]

        # 二维核与低通权重
        ker = self._build_kernels(orders)                   # [B,K,K,H,W]
        K = self.k
        tau = getattr(self, 'tau', 0.6)
        center_bias = getattr(self, 'center_bias', 2.0)
        mask_pre = ker.view(B, K*K, H, W)
        center_idx = self.center * self.k + self.center
        mask_pre[:, center_idx:center_idx+1] = mask_pre[:, center_idx:center_idx+1] + center_bias
        mask_lp = torch.softmax(mask_pre / tau, dim=1)      # [B,K^2,H,W]

        # 邻域展开
        xp = F.pad(x, [self.center, self.center, self.center, self.center], mode='reflect')
        neigh = F.unfold(xp, kernel_size=K, stride=1, padding=0).view(B, C, K*K, H, W)

        # 低通输出 + 边缘自适应残差
        y_lp_out = (neigh * mask_lp[:, None]).sum(dim=2)
        smooth_residual = getattr(self, 'smooth_residual', 0.2)
        lambda_eff = smooth_residual * (1.0 - g)            # [B,1,H,W]
        y_lp = x + lambda_eff * (y_lp_out - x)

        if getattr(self, 'mode', 'both') == 'lowpass':
            return (y_lp, orders, g) if not return_parts else (y_lp, y_lp, torch.zeros_like(y_lp), orders, g)

        # 高通
        ker_hp = ker - ker.mean(dim=(1, 2), keepdim=True)
        ker_hp = ker_hp / (ker_hp.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        mask_hp = ker_hp.view(B, K*K, H, W)
        y_hp = (neigh * mask_hp[:, None]).sum(dim=2)

        if getattr(self, 'mode', 'both') == 'highpass':
            return (y_hp, orders, g) if not return_parts else (y_hp, torch.zeros_like(y_hp), y_hp, orders, g)

        # 混合
        beta = self.beta
        alpha = torch.tanh(beta if torch.is_tensor(beta) else torch.tensor(beta, device=x.device, dtype=x.dtype))
        y = y_lp + alpha * g * y_hp
        return (y, orders, g) if not return_parts else (y, y_lp, y_hp, orders, g)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W, device=device)
    up = FractionalFusionUp(C, up_factor=2, kernel_size=5, vmax=1.2, hidden=64, mode='both').to(device)
    y, v, g = up(x)
    print('in:', x.shape, 'out:', y.shape, 'orders:', v.shape, 'gate:', g.shape)
