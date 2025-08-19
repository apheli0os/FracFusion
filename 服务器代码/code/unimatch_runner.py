import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FractionalOrderMaskGenerator(nn.Module):
    """
    分数阶微积分自适应掩膜生成器
    - 学习每像素8方向的分数阶次 v_dir
    - 生成 K×K 掩膜（方向加权融合）或真正的分数阶微积分二维核
    - 支持注意力缩放与掩膜归一
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        num_directions=8,
        order_range=(-2.0, 2.0),
        encoder_channels=64,
        use_attention=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 需为奇数"
        assert num_directions == 8, "当前实现固定8个方向"

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_directions = num_directions
        self.order_range = order_range
        self.use_attention = use_attention
        self.center = kernel_size // 2
        self.max_m = self.center  # GL 系数的最大阶（半径）

        # 8个方向单位向量（右、右下、下、左下、左、左上、上、右上）
        self.register_buffer("direction_vectors", self._init_direction_vectors())

        # 预计算：LUT 的分数阶系数表（用于 generate_directional_masks）
        self.register_buffer("fractional_coeffs_table", self._precompute_coeffs_table())

        # 预计算：方向-步长-位置的二值基底（用于真正分数阶二维核的高效合成）
        self.register_buffer("dir_step_base", self._precompute_dir_step_base())

        # 阶次预测网络
        self.order_predictor = self._build_order_predictor(in_channels, encoder_channels)

        # 注意力（可选）
        if use_attention:
            attention_channels = max(1, in_channels // 4)
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, attention_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attention_channels, 1, 1),
                nn.Sigmoid(),
            )

        self.init_weights()

    def _init_direction_vectors(self):
        """初始化8个方向的单位向量（dx, dy）"""
        angles = [0, 45, 90, 135, 180, 225, 270, 315]  # deg
        vectors = []
        for angle in angles:
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            vectors.append([dx, dy])
        v = torch.tensor(vectors, dtype=torch.float32)
        return v / (v.norm(dim=1, keepdim=True) + 1e-8)

    def _precompute_coeffs_table(self):
        """
        预计算分数阶系数 LUT（用于插值）:
        W_m = (-1)^m * v(v-1)...(v-m+1)/m!
        m = 0..M（M=半径=center）
        """
        num_samples = 200
        order_samples = torch.linspace(self.order_range[0], self.order_range[1], num_samples)
        M = self.max_m
        table = torch.zeros(num_samples, M + 1)
        for i, v in enumerate(order_samples):
            table[i, 0] = 1.0
            acc = 1.0
            for m in range(1, M + 1):
                acc = acc * (v - (m - 1)) / m  # v(v-1)..(v-m+1)/m!
                table[i, m] = ((-1) ** m) * acc
        return table  # [S, M+1]

    def _precompute_dir_step_base(self):
        """
        预计算每个方向下，“步长m对应哪些K×K位置”的二值掩膜基底。
        返回形状: [8, 2, M+1, K, K]
        维度2表示正向/负向。m=0..M。
        """
        K = self.kernel_size
        M = self.max_m
        cy = cx = self.center
        yy, xx = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
        rel = torch.stack([yy - cy, xx - cx], dim=-1).float()  # [K, K, 2]

        bases = []
        for d in range(8):
            vx, vy = self.direction_vectors[d]  # (dx, dy)
            # 图像网格中用 (y, x)·(dy, dx) 做投影
            proj = rel[..., 0] * vy + rel[..., 1] * vx  # [K, K]
            step = proj.round().clamp(min=-M, max=M).to(torch.int32)  # [-M, M]

            pos_masks = [(step == m).float() for m in range(M + 1)]
            neg_masks = [(step == -m).float() for m in range(M + 1)]
            bases.append(torch.stack([torch.stack(pos_masks, 0), torch.stack(neg_masks, 0)], 0))
        return torch.stack(bases, 0)  # [8,2,M+1,K,K]

    def _build_order_predictor(self, in_channels, encoder_channels):
        """构建分数阶次预测网络"""
        return nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels, 3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, encoder_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, encoder_channels, 3, padding=1, groups=encoder_channels),
            nn.Conv2d(encoder_channels, encoder_channels, 1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, self.num_directions, 1),
            nn.Tanh(),  # [-1,1] -> 之后映射到 order_range
        )

    def init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 给最终预测层设置不同偏置，增加方向多样性
        final_conv = None
        for m in self.order_predictor.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.num_directions:
                final_conv = m
                break
        if final_conv is not None and final_conv.bias is not None:
            direction_biases = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.0, 0.15], dtype=final_conv.bias.dtype)
            with torch.no_grad():
                final_conv.bias.copy_(direction_biases)
                nn.init.normal_(final_conv.weight, std=0.02)

    def interpolate_coefficients(self, orders):
        """
        LUT 插值方式得到系数（用于 generate_directional_masks）
        输入: orders [B,8,H,W]
        输出: [B,8,M+1,H,W]
        """
        B, D, H, W = orders.shape
        table = self.fractional_coeffs_table  # [S,M+1]
        S, M1 = table.shape

        t = (orders - self.order_range[0]) / (self.order_range[1] - self.order_range[0])
        t = torch.clamp(t, 0, 1)
        idx = t * (S - 1)
        idx0 = idx.floor().long()
        idx1 = idx.ceil().long()
        w1 = idx - idx0.float()
        w0 = 1.0 - w1

        # 按索引取表并线性插值
        coeff0 = table[idx0.flatten()].view(B, D, H, W, M1)
        coeff1 = table[idx1.flatten()].view(B, D, H, W, M1)
        coeff = w0.unsqueeze(-1) * coeff0 + w1.unsqueeze(-1) * coeff1
        # [B,D,H,W,M+1] -> [B,D,M+1,H,W]
        return coeff.permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def gl_coefficients(v, M):
        """
        计算 GL 递推系数（向量化）
        v: [B,1,H,W] 或 [B,H,W]
        返回: [B,M+1,H,W]
        """
        if v.dim() == 3:
            v = v.unsqueeze(1)
        B, _, H, W = v.shape
        coeffs = [torch.ones(B, 1, H, W, device=v.device, dtype=v.dtype)]
        w = coeffs[0]
        for m in range(1, M + 1):
            w = w * (v - (m - 1)) / m * (-1.0)
            coeffs.append(w)
        return torch.cat(coeffs, dim=1)  # [B,M+1,H,W]

    def generate_directional_masks(self, fractional_orders):
        """
        用 LUT + 投影步长生成8方向掩膜（更偏向“可视化/启发式”）
        返回: [B, K*K, H, W] 之前的中间形状为 [B,8,K,K,H,W]
        """
        B, _, H, W = fractional_orders.shape
        K = self.kernel_size
        M = self.max_m

        # [B,8,M+1,H,W]
        coefficients = self.interpolate_coefficients(fractional_orders)

        # 使用预计算的基底，将 1D 系数映射为 2D K×K 掩膜
        # dir_step_base: [8,2,M+1,K,K] （pos/neg）
        masks_dir = []
        for d in range(8):
            base_pos = self.dir_step_base[d, 0]  # [M+1,K,K]
            base_neg = self.dir_step_base[d, 1]  # [M+1,K,K]
            coeff = coefficients[:, d]  # [B,M+1,H,W]
            # [B,M+1,1,1,H,W] * [1,M+1,K,K,1,1] -> sum_m -> [B,K,K,H,W]
            mask_d = (
                coeff[:, :, None, None, :, :]
                * (base_pos[None, :, :, :, None, None] - base_neg[None, :, :, :, None, None])
            ).sum(1)
            masks_dir.append(mask_d)
        # [8,B,K,K,H,W] -> [B,8,K,K,H,W]
        masks_dir = torch.stack(masks_dir, 0).permute(1, 0, 2, 3, 4, 5).contiguous()

        # 基于梯度方向进行方向加权融合 -> [B, K*K, H, W]
        fused = self.adaptive_mask_fusion(masks_dir, feature_maps=None, shape=(B, H, W))
        return fused

    def adaptive_mask_fusion(self, directional_masks, feature_maps=None, shape=None):
        """
        融合8方向掩膜:
        - 若给定 feature_maps，则用 Sobel 梯度方向做加权；
        - 否则使用各方向均匀平均。
        输入: directional_masks [B,8,K,K,H,W]
        返回: [B,K*K,H,W]
        """
        B, _, K, _, H, W = directional_masks.shape
        if feature_maps is None:
            # 均匀平均
            fused_mask = directional_masks.mean(dim=1).view(B, K * K, H, W)
            return fused_mask

        # 若需基于梯度做权重，这里保留接口（当前 evaluation 多数不传入 feature_maps）
        # 计算灰度梯度
        sobel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        sobel_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        feature_gray = feature_maps.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(feature_gray, sobel_x, padding=1)
        grad_y = F.conv2d(feature_gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 梯度方向量化到 0..7
        grad_dir = torch.atan2(grad_y, grad_x)  # [-pi,pi]
        grad_dir = (grad_dir + math.pi) / (2 * math.pi) * 8.0
        grad_dir = grad_dir.squeeze(1)  # [B,H,W]

        weights = torch.zeros(B, 8, H, W, device=feature_maps.device)
        for d in range(8):
            diff = torch.abs(grad_dir - d)
            diff = torch.min(diff, 8 - diff)
            weights[:, d] = torch.exp(-diff) * (1 + grad_mag.squeeze(1))
        weights = F.softmax(weights, dim=1)  # [B,8,H,W]

        fused = torch.zeros(B, K * K, H, W, device=feature_maps.device, dtype=feature_maps.dtype)
        for d in range(8):
            m = directional_masks[:, d].view(B, K * K, H, W)
            fused += m * weights[:, d : d + 1]
        return fused

    def forward(self, feature_maps):
        """
        返回:
        - adaptive_mask: [B,K^2,H,W]
        - fractional_orders: [B,8,H,W]
        """
        B, C, H, W = feature_maps.shape
        raw_orders = self.order_predictor(feature_maps)  # [-1,1]
        fractional_orders = raw_orders * (self.order_range[1] - self.order_range[0]) / 2.0 + (
            self.order_range[1] + self.order_range[0]
        ) / 2.0

        # 方向掩膜（启发式/LUT 版），适合作为 CARAFE 权重的初值
        directional_masks = self.generate_directional_masks(fractional_orders)

        # 注意力
        if self.use_attention:
            att = self.attention(feature_maps)
            directional_masks = directional_masks * att

        # 归一化（和为1，避免数值漂移）
        adaptive_mask = self.normalize_mask(directional_masks)
        return adaptive_mask, fractional_orders

    def normalize_mask(self, mask):
        """按 K^2 归一，确保每位置系数和为1"""
        return mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

    # ========== 真正的分数阶二维核（GL + 方向基底，向量化实现） ==========

    def generate_fractional_differential_masks(self, fractional_orders):
        """
        基于 GL 递推 + 方向投影合成真正的二维 K×K 分数阶核（每像素不同）
        输入: fractional_orders [B,8,H,W]
        输出: [B,K^2,H,W]
        """
        B, _, H, W = fractional_orders.shape
        K, M = self.kernel_size, self.max_m
        device = fractional_orders.device
        dtype = fractional_orders.dtype

        kernels = []
        for d in range(8):
            v = fractional_orders[:, d : d + 1]  # [B,1,H,W]
            coeff = self.gl_coefficients(v, M)  # [B,M+1,H,W]
            base_pos = self.dir_step_base[d, 0]  # [M+1,K,K]
            base_neg = self.dir_step_base[d, 1]  # [M+1,K,K]
            # 合成二维核：sum_m C_m*(pos-neg) -> [B,K,K,H,W]
            ker = (
                coeff[:, :, None, None, :, :]
                * (base_pos[None, :, :, :, None, None] - base_neg[None, :, :, :, None, None])
            ).sum(1)
            # 高通属性（微分型）零直流
            ker = ker - ker.mean(dim=(1, 2), keepdim=True)
            kernels.append(ker)

        # 8方向平均（也可换成方向加权）
        kernel_2d = torch.stack(kernels, 0).mean(0)  # [B,K,K,H,W]
        masks = kernel_2d.view(B, K * K, H, W)
        # 为稳定性，做一次 L1 归一（不改变零直流性质）
        masks = masks / (masks.abs().sum(dim=1, keepdim=True) + 1e-8)
        return masks.to(device=device, dtype=dtype)

    def apply_fractional_operator(self, image, fractional_orders):
        """
        应用二维核到输入（同分辨率处理）
        image: [B,C,H,W]；orders: [B,8,H,W]
        返回: [B,C,H,W]
        """
        B, C, H, W = image.shape
        K = self.kernel_size
        pad = self.center

        # 生成每像素二维核（K^2）
        frac_masks = self.generate_fractional_differential_masks(fractional_orders)  # [B,K^2,H,W]

        # unfold 邻域并做逐像素加权
        xpad = F.pad(image, [pad, pad, pad, pad], mode="reflect")
        patches = F.unfold(xpad, kernel_size=K, stride=1, padding=0)  # [B, C*K^2, H, W]
        patches = patches.view(B, C, K * K, H, W)
        out = (patches * frac_masks.unsqueeze(1)).sum(dim=2)  # [B,C,H,W]
        return out

    # 可视化用的小工具
    def visualize_mask_effect(self, image_tensor, mask_tensor):
        mask_variance = mask_tensor.var(dim=1, keepdim=True)  # [B,1,H,W]
        return mask_variance.squeeze().detach().cpu().numpy()


class FractionalCarafe(nn.Module):
    """
    分数阶 CARAFE（使用学习到的掩膜做内容自适应重组）
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        up_factor=2,
        order_range=(-1.5, 1.5),
        encoder_channels=64,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor

        self.feature_compressor = (
            nn.Conv2d(in_channels, encoder_channels, 1) if in_channels != encoder_channels else nn.Identity()
        )
        self.mask_generator = FractionalOrderMaskGenerator(
            in_channels=encoder_channels,
            kernel_size=kernel_size,
            order_range=order_range,
            encoder_channels=encoder_channels,
        )

    def forward(self, x):
        """
        x: [B,C,H,W]
        返回: 上采样特征 [B,C,H*s,W*s] 与 orders（便于可视化）
        """
        B, C, H, W = x.shape

        feat = self.feature_compressor(x)
        mask, orders = self.mask_generator(feat)  # [B,K^2,H,W], [B,8,H,W]

        # unfold 邻域
        pad = self.mask_generator.center
        K = self.kernel_size
        xpad = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        patches = F.unfold(xpad, kernel_size=K, stride=1, padding=0)  # [B, C*K^2, H, W]

        # 上采样到目标分辨率（近似 CARAFE 的重组网格）
        if self.up_factor > 1:
            s = self.up_factor
            patches = F.interpolate(patches, scale_factor=s, mode="nearest")
            mask = F.interpolate(mask, scale_factor=s, mode="nearest")

        Hs, Ws = H * self.up_factor, W * self.up_factor
        patches = patches.view(B, C, K * K, Hs, Ws)
        mask = mask.view(B, 1, K * K, Hs, Ws)

        out = (patches * mask).sum(dim=2)  # [B,C,Hs,Ws]
        return out, orders


class FractionalDifferentialCarafe(nn.Module):
    """
    使用真正分数阶二维核进行增强，再可选上采样（与 CARAFE 路径对照）
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        up_factor=2,
        order_range=(-1.5, 1.5),
        encoder_channels=64,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor

        self.feature_compressor = (
            nn.Conv2d(in_channels, encoder_channels, 1) if in_channels != encoder_channels else nn.Identity()
        )
        self.mask_generator = FractionalOrderMaskGenerator(
            in_channels=encoder_channels, kernel_size=kernel_size, order_range=order_range, encoder_channels=encoder_channels
        )

    def forward(self, x):
        feat = self.feature_compressor(x)
        _, orders = self.mask_generator(feat)  # 仅用来预测阶次
        out = self.mask_generator.apply_fractional_operator(x, orders)  # 同分辨率增强

        if self.up_factor > 1:
            out = F.interpolate(out, scale_factor=self.up_factor, mode="bilinear", align_corners=False)
        return out, orders


# ===================== 演示与自测 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机特征
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W, device=device)

    print("=" * 60)
    print("测试分数阶掩膜生成器")
    print("=" * 60)
    gen = FractionalOrderMaskGenerator(in_channels=C, kernel_size=5, order_range=(-1.5, 1.5), encoder_channels=32).to(device)
    mask, orders = gen(x)
    print(f"输入特征形状: {x.shape}")
    print(f"输出掩膜形状: {mask.shape}")
    print(f"分数阶次形状: {orders.shape}")
    print(f"分数阶次范围: [{orders.min().item():.3f}, {orders.max().item():.3f}]")
    s = mask.sum(dim=1)
    print(f"掩膜归一化检查 - 最小和: {s.min().item():.6f}, 最大和: {s.max().item():.6f}")

    print("\n" + "=" * 60)
    print("测试分数阶CARAFE（上采样）")
    print("=" * 60)
    carafe = FractionalCarafe(in_channels=C, kernel_size=5, up_factor=2, order_range=(-1.5, 1.5), encoder_channels=32).to(device)
    y, o1 = carafe(x)
    print(f"CARAFE输入: {x.shape} -> 输出: {y.shape} (x{y.shape[2] // x.shape[2]})")
    print(f"CARAFE 阶次范围: [{o1.min().item():.3f}, {o1.max().item():.3f}]")
    print(f"总参数量: {sum(p.numel() for p in carafe.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试真正的分数阶二维核（同分辨率增强）")
    print("=" * 60)
    diff = FractionalDifferentialCarafe(in_channels=C, kernel_size=5, up_factor=1, order_range=(-1.5, 1.5), encoder_channels=32).to(device)
    y2, o2 = diff(x)
    print(f"分数阶增强 输入: {x.shape} -> 输出: {y2.shape}")
    print(f"阶次范围: [{o2.min().item():.3f}, {o2.max().item():.3f}]")

    in_std = x.std().item()
    out_std = y2.std().item()
    print(f"输入标准差: {in_std:.4f} -> 输出标准差: {out_std:.4f} (变化 {((out_std-in_std)/in_std*100):.2f}%)")

    print("\n模型测试完成！")