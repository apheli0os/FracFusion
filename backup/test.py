import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FractionalOrderMaskGenerator(nn.Module):
    """
    分数阶微积分自适应掩膜生成器
    
    该模块为每个像素位置学习8个方向的分数阶次，并基于分数阶微积分理论
    生成自适应掩膜，用于保持细节信息并平滑低频区域。
    
    核心思想：
    - 强边缘处：较大的微分阶次，增强边缘
    - 弱边缘/纹理处：较小的微分阶次，保持纹理
    - 平滑区域：微小的微分阶次，轻微平滑
    - 噪声区域：负阶次，强平滑去噪
    """
    
    def __init__(self, 
                 in_channels,
                 kernel_size=5,
                 num_directions=8,
                 order_range=(-2.0, 2.0),
                 encoder_channels=64,
                 use_attention=True):
        """
        初始化分数阶掩膜生成器
        
        Args:
            in_channels: 输入特征通道数
            kernel_size: 掩膜核大小 (建议3, 5, 7)
            num_directions: 方向数量 (固定为8个主要方向)
            order_range: 分数阶次范围 (min_order, max_order)
            encoder_channels: 编码器中间通道数
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_directions = num_directions
        self.order_range = order_range
        self.use_attention = use_attention
        
        # 定义8个主要方向的单位向量 (右、右下、下、左下、左、左上、上、右上)
        self.register_buffer('direction_vectors', self._init_direction_vectors())
        
        # 预计算分数阶系数查找表 (用于加速计算)
        self.register_buffer('fractional_coeffs_table', self._precompute_coeffs_table())
        
        # 分数阶次预测网络
        self.order_predictor = self._build_order_predictor(in_channels, encoder_channels)
        
        # 可选的注意力机制
        if use_attention:
            attention_channels = max(1, in_channels // 4)  # 确保至少有1个通道
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, attention_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attention_channels, 1, 1),
                nn.Sigmoid()
            )
        
        self.init_weights()
    
    def _init_direction_vectors(self):
        """初始化8个方向的单位向量"""
        angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 度
        vectors = []
        for angle in angles:
            rad = math.radians(angle)
            # 将角度转换为网格坐标系下的方向向量
            dx = math.cos(rad)
            dy = math.sin(rad)
            vectors.append([dx, dy])
        return torch.tensor(vectors, dtype=torch.float32)
    
    def _precompute_coeffs_table(self):
        """
        预计算分数阶系数查找表
        
        基于分数阶微积分理论，系数计算公式为：
        W_m = (-1)^m * v(v-1)(v-2)...(v-m+1) / m!
        其中 v 是分数阶次，m 是系数索引
        """
        # 创建阶次采样点 (用于插值)
        order_samples = torch.linspace(self.order_range[0], self.order_range[1], 200)
        max_m = self.kernel_size  # 最大系数索引
        
        # 初始化系数表 [num_orders, max_m+1]
        coeffs_table = torch.zeros(len(order_samples), max_m + 1)
        
        for i, v in enumerate(order_samples):
            for m in range(max_m + 1):
                if m == 0:
                    coeffs_table[i, m] = 1.0
                else:
                    # 计算 v(v-1)(v-2)...(v-m+1)
                    numerator = 1.0
                    for k in range(m):
                        numerator *= (v - k)
                    
                    # 计算 m!
                    denominator = math.factorial(m)
                    
                    # 应用符号 (-1)^m
                    sign = (-1) ** m
                    
                    coeffs_table[i, m] = sign * numerator / denominator
        
        return coeffs_table
    
    def _build_order_predictor(self, in_channels, encoder_channels):
        """构建分数阶次预测网络"""
        return nn.Sequential(
            # 多尺度特征提取
            nn.Conv2d(in_channels, encoder_channels, 3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            
            # 扩张卷积捕获更大感受野
            nn.Conv2d(encoder_channels, encoder_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            
            # 深度可分离卷积减少参数
            nn.Conv2d(encoder_channels, encoder_channels, 3, padding=1, groups=encoder_channels),
            nn.Conv2d(encoder_channels, encoder_channels, 1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            
            # 输出8个方向的分数阶次
            nn.Conv2d(encoder_channels, self.num_directions, 1),
            nn.Tanh()  # 输出范围 [-1, 1]，后续会缩放到order_range
        )
    
    def init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 改进初始化：让不同方向有不同的初始倾向
        final_conv = None
        for m in self.order_predictor.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.num_directions:
                final_conv = m
                break
        
        if final_conv is not None:
            # 为8个方向设置不同的初始偏置，增加多样性
            direction_biases = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.0, 0.15])
            with torch.no_grad():
                final_conv.bias.copy_(direction_biases)
                # 权重使用较小的随机值
                nn.init.normal_(final_conv.weight, std=0.02)
    
    def interpolate_coefficients(self, orders):
        """
        通过插值获取分数阶系数
        
        Args:
            orders: 分数阶次张量 [B, num_directions, H, W]
            
        Returns:
            coefficients: 插值得到的系数 [B, num_directions, kernel_size+1, H, W]
        """
        B, D, H, W = orders.shape
        
        # 将阶次缩放到查找表范围
        table_size = self.fractional_coeffs_table.shape[0]
        normalized_orders = (orders - self.order_range[0]) / (self.order_range[1] - self.order_range[0])
        normalized_orders = torch.clamp(normalized_orders, 0, 1)
        
        # 计算插值索引
        indices = normalized_orders * (table_size - 1)
        indices_low = torch.floor(indices).long()
        indices_high = torch.ceil(indices).long()
        
        # 插值权重
        weight_high = indices - indices_low.float()
        weight_low = 1.0 - weight_high
        
        # 执行插值
        coeffs_low = self.fractional_coeffs_table[indices_low.flatten()].view(B, D, H, W, -1)
        coeffs_high = self.fractional_coeffs_table[indices_high.flatten()].view(B, D, H, W, -1)
        
        interpolated_coeffs = weight_low.unsqueeze(-1) * coeffs_low + weight_high.unsqueeze(-1) * coeffs_high
        
        return interpolated_coeffs.permute(0, 1, 4, 2, 3)  # [B, D, kernel_size+1, H, W]
    
    def generate_directional_masks(self, fractional_orders):
        """
        基于分数阶次生成8个方向的掩膜
        
        Args:
            fractional_orders: 每个方向的分数阶次 [B, 8, H, W]
            
        Returns:
            masks: 8个方向的掩膜 [B, 8, kernel_size, kernel_size, H, W]
        """
        B, _, H, W = fractional_orders.shape
        
        # 获取分数阶系数
        coefficients = self.interpolate_coefficients(fractional_orders)  # [B, 8, kernel_size+1, H, W]
        
        # 初始化掩膜
        masks = torch.zeros(B, 8, self.kernel_size, self.kernel_size, H, W, 
                           device=fractional_orders.device, dtype=fractional_orders.dtype)
        
        center = self.kernel_size // 2
        
        # 为每个方向生成掩膜
        for dir_idx in range(8):
            dx, dy = self.direction_vectors[dir_idx][0].item(), self.direction_vectors[dir_idx][1].item()
            
            # 在每个核位置填充系数
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    # 计算相对于中心的距离
                    rel_i, rel_j = i - center, j - center
                    
                    # 计算在当前方向上的投影距离
                    projection = abs(rel_i * dx + rel_j * dy)
                    
                    # 根据投影距离选择系数
                    coeff_idx = min(int(round(projection)), self.kernel_size)
                    
                    if coeff_idx < coefficients.shape[2]:
                        masks[:, dir_idx, i, j, :, :] = coefficients[:, dir_idx, coeff_idx, :, :]
        
        return masks
    
    def adaptive_mask_fusion(self, directional_masks, feature_maps):
        """
        自适应融合8个方向的掩膜
        
        根据局部特征自适应地组合不同方向的掩膜，
        在边缘处强化主方向，在纹理处平衡多方向。
        
        Args:
            directional_masks: 8个方向的掩膜 [B, 8, K, K, H, W]
            feature_maps: 输入特征图 [B, C, H, W]
            
        Returns:
            fused_mask: 融合后的掩膜 [B, K*K, H, W]
        """
        B, _, K, _, H, W = directional_masks.shape
        
           # 修改：使用正确的padding确保输出尺寸不变
        # 创建固定的梯度计算核
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                          device=feature_maps.device, dtype=feature_maps.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                          device=feature_maps.device, dtype=feature_maps.dtype)
    
        # 计算梯度，使用padding=1保持尺寸
        feature_gray = feature_maps.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_x = F.conv2d(feature_gray, sobel_x, padding=1)    # [B, 1, H, W]
        grad_y = F.conv2d(feature_gray, sobel_y, padding=1)    # [B, 1, H, W]
    
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
        # 计算主导方向
        gradient_direction = torch.atan2(grad_y, grad_x)  # [-π, π]
        gradient_direction = (gradient_direction + math.pi) / (2 * math.pi) * 8  # [0, 8]
        
        # 为每个方向计算权重
        direction_weights = torch.zeros(B, 8, H, W, device=feature_maps.device)
        
        for dir_idx in range(8):
            # 计算方向相似度
            dir_diff = torch.abs(gradient_direction.squeeze(1) - dir_idx)
            dir_diff = torch.min(dir_diff, 8 - dir_diff)  # 考虑周期性
            
            # 基于梯度强度和方向相似度计算权重
            weight = torch.exp(-dir_diff) * (1 + gradient_magnitude.squeeze(1))
            direction_weights[:, dir_idx] = weight
        
        # 归一化权重
        direction_weights = F.softmax(direction_weights, dim=1)
        
        # 融合掩膜
        fused_mask = torch.zeros(B, K*K, H, W, device=feature_maps.device)
        
        for dir_idx in range(8):
            # 展平当前方向的掩膜
            dir_mask = directional_masks[:, dir_idx].view(B, K*K, H, W)
            
            # 加权累加
            weight = direction_weights[:, dir_idx:dir_idx+1]  # [B, 1, H, W]
            fused_mask += dir_mask * weight
        
        return fused_mask
    
    def forward(self, feature_maps):
        """
        前向传播
        
        Args:
            feature_maps: 输入特征图 [B, C, H, W]
            
        Returns:
            adaptive_mask: 自适应掩膜 [B, kernel_size^2, H, W]
            fractional_orders: 学习到的分数阶次 [B, 8, H, W] (用于可视化)
        """
        B, C, H, W = feature_maps.shape
        
        # 1. 预测每个位置8个方向的分数阶次
        raw_orders = self.order_predictor(feature_maps)  # [B, 8, H, W], 范围[-1, 1]
        
        # 2. 缩放到实际的阶次范围
        fractional_orders = raw_orders * (self.order_range[1] - self.order_range[0]) / 2 + \
                           (self.order_range[1] + self.order_range[0]) / 2
        
        # 3. 生成8个方向的分数阶掩膜
        directional_masks = self.generate_directional_masks(fractional_orders)
        
        # 4. 自适应融合掩膜
        adaptive_mask = self.adaptive_mask_fusion(directional_masks, feature_maps)
        
        # 5. 可选的注意力机制
        if self.use_attention:
            attention_weight = self.attention(feature_maps)
            adaptive_mask = adaptive_mask * attention_weight
        
        # 6. 归一化掩膜 (确保每个位置的掩膜系数和为1)
        adaptive_mask = self.normalize_mask(adaptive_mask)
        
        return adaptive_mask, fractional_orders
    
    def normalize_mask(self, mask):
        """
        归一化掩膜，确保每个空间位置的系数和为1
        
        Args:
            mask: 输入掩膜 [B, K^2, H, W]
            
        Returns:
            normalized_mask: 归一化后的掩膜
        """
        # 计算每个位置的系数和
        mask_sum = mask.sum(dim=1, keepdim=True) + 1e-8
        
        # 归一化
        normalized_mask = mask / mask_sum
        
        return normalized_mask
    
    # 在 FractionalOrderMaskGenerator 类中添加以下方法

    def compute_fractional_coefficient(self, v, m):
        """
        计算分数阶系数: (-1)^m * Γ(v+1) / (Γ(m+1) * Γ(v-m+1))
        使用递推关系避免直接计算Gamma函数
        
        Args:
            v: 分数阶次 (张量)
            m: 系数索引 (整数)
            
        Returns:
            coefficient: 分数阶系数
        """
        if m == 0:
            return torch.ones_like(v)
        
        # 使用递推关系: W_m = W_{m-1} * (v - m + 1) / m * (-1)
        coeff = torch.ones_like(v)
        for k in range(1, m + 1):
            coeff = coeff * (v - k + 1) / k * (-1)
        
        return coeff

    def generate_fractional_differential_masks(self, fractional_orders):
        """
        基于分数阶次生成真正的分数阶微积分算子掩膜
        
        Args:
            fractional_orders: 每个方向的分数阶次 [B, 8, H, W]
            
        Returns:
            masks: 分数阶微积分算子掩膜 [B, kernel_size^2, H, W]
        """
        B, num_dirs, H, W = fractional_orders.shape
        device = fractional_orders.device
        
        # 初始化最终掩膜
        final_masks = torch.zeros(B, self.kernel_size * self.kernel_size, H, W, 
                                device=device, dtype=fractional_orders.dtype)
        
        center = self.kernel_size // 2
        
        # 为每个方向生成分数阶微积分算子
        for dir_idx in range(8):
            # 获取当前方向的分数阶次
            v = fractional_orders[:, dir_idx, :, :]  # [B, H, W]
            
            # 获取方向向量
            dx = float(self.direction_vectors[dir_idx, 0])
            dy = float(self.direction_vectors[dir_idx, 1])
            
            # 生成该方向的分数阶微积分算子
            direction_mask = torch.zeros(B, self.kernel_size, self.kernel_size, H, W, device=device)
            
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    # 计算相对位置
                    rel_i, rel_j = i - center, j - center
                    
                    # 计算在当前方向上的投影距离
                    projection = rel_i * dx + rel_j * dy
                    
                    # 计算分数阶系数
                    if abs(projection) < 1e-6:  # 中心点
                        if torch.allclose(v, torch.zeros_like(v)):
                            direction_mask[:, i, j, :, :] = 1.0  # 零阶 = 恒等
                        else:
                            direction_mask[:, i, j, :, :] = 0.0  # 分数阶在中心点为0
                    else:
                        # 计算距离步长
                        step = int(round(abs(projection)))
                        if step <= self.kernel_size // 2:
                            # 计算分数阶系数
                            coeff = self.compute_fractional_coefficient(v, step)
                            
                            # 考虑方向（正向或负向）
                            if projection < 0:
                                coeff = -coeff
                            
                            direction_mask[:, i, j, :, :] = coeff
            
            # 将3D掩膜展平并累加到最终掩膜
            flattened_mask = direction_mask.view(B, self.kernel_size * self.kernel_size, H, W)
            final_masks += flattened_mask / 8.0  # 平均8个方向
        
        return final_masks

    def apply_fractional_operator(self, image, fractional_orders):
        """
        应用分数阶微积分算子到图像
        
        Args:
            image: 输入图像 [B, C, H, W]
            fractional_orders: 分数阶次 [B, 8, H, W]
            
        Returns:
            processed_image: 处理后的图像
        """
        B, C, H, W = image.shape
        
        # 生成分数阶算子掩膜
        fractional_masks = self.generate_fractional_differential_masks(fractional_orders)
        
        # 对图像应用分数阶算子（类似CARAFE操作）
        pad = self.kernel_size // 2
        padded_image = F.pad(image, [pad] * 4, mode='reflect')
        
        # 使用unfold提取滑动窗口
        unfolded = F.unfold(padded_image, kernel_size=self.kernel_size, stride=1, padding=0)
        unfolded = unfolded.view(B, C, self.kernel_size * self.kernel_size, H, W)
        
        # 应用分数阶掩膜
        processed = unfolded * fractional_masks.unsqueeze(1)  # 广播到所有通道
        processed = processed.sum(dim=2)  # 在kernel维度求和
        
        return processed
    # 在 FractionalMaskEvaluator 类中添加这个方法
    def visualize_mask_effect(self, image_tensor, mask_tensor):
        """
        可视化掩膜效果
        
        Args:
            image_tensor: 原始图像 [1, 3, H, W]
            mask_tensor: 掩膜张量 [1, K^2, H, W]
        
        Returns:
            visualization: 掩膜效果可视化 [H, W]
        """
        # 计算掩膜的空间变化
        mask_variance = mask_tensor.var(dim=1, keepdim=True)  # [1, 1, H, W]
        return mask_variance.squeeze().cpu().numpy()

class FractionalCarafe(nn.Module):
    """
    基于分数阶微积分的CARAFE实现
    
    结合分数阶自适应掩膜和CARAFE操作，实现更好的细节保持。
    """
    
    def __init__(self, 
                 in_channels,
                 kernel_size=5,
                 up_factor=2,
                 order_range=(-1.5, 1.5),
                 encoder_channels=64):
        """
        初始化分数阶CARAFE
        
        Args:
            in_channels: 输入通道数
            kernel_size: 卷积核大小
            up_factor: 上采样倍数
            order_range: 分数阶次范围
            encoder_channels: 编码器通道数
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        
        # 特征压缩器（如果需要）
        self.feature_compressor = nn.Conv2d(in_channels, encoder_channels, 1) if in_channels != encoder_channels else nn.Identity()
        
        # 分数阶掩膜生成器
        self.mask_generator = FractionalOrderMaskGenerator(
            in_channels=encoder_channels,  # 使用压缩后的通道数
            kernel_size=kernel_size,
            order_range=order_range,
            encoder_channels=encoder_channels
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            output: 处理后的特征 [B, C, H*up_factor, W*up_factor]
            fractional_orders: 学习到的分数阶次（用于分析）
        """
        # 1. 压缩特征（如果需要）
        compressed_feat = self.feature_compressor(x)
        
        # 2. 生成自适应掩膜
        adaptive_mask, fractional_orders = self.mask_generator(compressed_feat)
        
        # 3. 应用CARAFE操作
        output = self.fractional_carafe_op(x, adaptive_mask)
        
        return output, fractional_orders
    
    def fractional_carafe_op(self, x, mask):
        """
        执行分数阶CARAFE操作
        
        Args:
            x: 输入特征 [B, C, H, W]
            mask: 自适应掩膜 [B, K^2, H, W]
            
        Returns:
            output: 处理后的特征
        """
        B, C, H, W = x.shape
        
        # 1. 填充
        pad = self.kernel_size // 2
        padded_x = F.pad(x, [pad] * 4, mode='reflect')
        
        # 2. Unfold操作
        unfolded_x = F.unfold(padded_x, kernel_size=self.kernel_size, stride=1, padding=0)
        unfolded_x = unfolded_x.reshape(B, C * self.kernel_size * self.kernel_size, H, W)
        
        # 3. 上采样
        if self.up_factor > 1:
            unfolded_x = F.interpolate(unfolded_x, scale_factor=self.up_factor, mode='nearest')
            mask = F.interpolate(mask, scale_factor=self.up_factor, mode='nearest')
        
        # 4. 重新组织维度
        output_h, output_w = H * self.up_factor, W * self.up_factor
        unfolded_x = unfolded_x.reshape(B, C, self.kernel_size * self.kernel_size, output_h, output_w)
        mask = mask.reshape(B, 1, self.kernel_size * self.kernel_size, output_h, output_w)
        
        # 5. 应用自适应掩膜
        output = unfolded_x * mask
        output = output.sum(dim=2)
        
        return output


# 另外，如果您想要使用真正的分数阶微积分算子，可以添加一个新的类：
class FractionalDifferentialCarafe(nn.Module):
    """
    基于真正分数阶微积分算子的CARAFE实现
    """
    
    def __init__(self, 
                 in_channels,
                 kernel_size=5,
                 up_factor=2,
                 order_range=(-1.5, 1.5),
                 encoder_channels=64):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        
        # 特征压缩器
        self.feature_compressor = nn.Conv2d(in_channels, encoder_channels, 1) if in_channels != encoder_channels else nn.Identity()
        
        # 分数阶掩膜生成器
        self.mask_generator = FractionalOrderMaskGenerator(
            in_channels=encoder_channels,
            kernel_size=kernel_size,
            order_range=order_range,
            encoder_channels=encoder_channels
        )
    
    def forward(self, x):
        """
        使用真正的分数阶微积分算子处理
        """
        # 1. 压缩特征
        compressed_feat = self.feature_compressor(x)
        
        # 2. 预测分数阶次
        _, fractional_orders = self.mask_generator(compressed_feat)
        
        # 3. 直接应用分数阶微积分算子
        processed_features = self.mask_generator.apply_fractional_operator(x, fractional_orders)
        
        # 4. 上采样（如果需要）
        if self.up_factor > 1:
            processed_features = F.interpolate(processed_features, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        
        return processed_features, fractional_orders

# 使用示例和测试代码
if __name__ == "__main__":
    # 测试分数阶掩膜生成器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    B, C, H, W = 2, 64, 32, 32
    test_features = torch.randn(B, C, H, W).to(device)
    
    print("=" * 60)
    print("测试分数阶掩膜生成器")
    print("=" * 60)
    
    # 初始化模型
    mask_generator = FractionalOrderMaskGenerator(
        in_channels=C,
        kernel_size=5,
        order_range=(-1.5, 1.5),
        encoder_channels=32
    ).to(device)
    
    # 前向传播
    adaptive_mask, fractional_orders = mask_generator(test_features)
    
    print(f"输入特征形状: {test_features.shape}")
    print(f"输出掩膜形状: {adaptive_mask.shape}")
    print(f"分数阶次形状: {fractional_orders.shape}")
    print(f"分数阶次范围: [{fractional_orders.min().item():.3f}, {fractional_orders.max().item():.3f}]")
    
    # 验证掩膜归一化
    mask_sum = adaptive_mask.sum(dim=1)
    print(f"掩膜归一化检查 - 最小和: {mask_sum.min().item():.6f}, 最大和: {mask_sum.max().item():.6f}")
    
    print("\n" + "=" * 60)
    print("测试分数阶CARAFE")
    print("=" * 60)
    
    # 测试分数阶CARAFE
    fractional_carafe = FractionalCarafe(
        in_channels=C,
        kernel_size=5,
        up_factor=2,
        order_range=(-1.5, 1.5),
        encoder_channels=32
    ).to(device)
    
    output, orders = fractional_carafe(test_features)
    
    print(f"CARAFE输入形状: {test_features.shape}")
    print(f"CARAFE输出形状: {output.shape}")
    print(f"上采样倍数验证: {output.shape[2] // test_features.shape[2]}x")
    
    # 计算参数量
    total_params = sum(p.numel() for p in fractional_carafe.parameters())
    print(f"总参数量: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("测试分数阶微积分算子")
    print("=" * 60)
    
    # 测试真正的分数阶微积分算子
    fractional_diff_carafe = FractionalDifferentialCarafe(
        in_channels=C,
        kernel_size=5,
        up_factor=1,  # 不上采样，只进行分数阶处理
        order_range=(-1.5, 1.5),
        encoder_channels=32
    ).to(device)
    
    diff_output, diff_orders = fractional_diff_carafe(test_features)
    
    print(f"分数阶微积分输入形状: {test_features.shape}")
    print(f"分数阶微积分输出形状: {diff_output.shape}")
    
    # 分析处理效果
    input_std = test_features.std().item()
    output_std = diff_output.std().item()
    print(f"输入标准差: {input_std:.4f}")
    print(f"输出标准差: {output_std:.4f}")
    print(f"标准差变化: {(output_std - input_std) / input_std * 100:.2f}%")
    
    print("\n模型测试完成！")