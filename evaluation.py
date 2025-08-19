import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 导入分数阶融合上采样模块
from fractional_fusion import FractionalFusionUp

class FractionalMaskEvaluator:
    """分数阶掩膜效果评估器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        
    def load_test_image(self, image_path, target_size=(256, 256)):
        """
        加载并预处理测试图像
        Args:
            image_path: 图像路径
            target_size: 目标尺寸
                - None: 不缩放，保持原图尺寸
                - int: 以长边为 target_size 的等比缩放
                - (w, h): 缩放到指定宽高
        Returns:
            tensor: 预处理后的张量 [1, 3, H, W]
            original: 原始RGB图像（numpy）
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 保存原始RGB图（仅用于展示）
        original = cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2RGB)

        # 转 RGB 作为网络输入
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 按 target_size 处理尺寸
        if target_size is None:
            resized = image  # 保持原图
        elif isinstance(target_size, int):
            h, w = image.shape[:2]
            if h == 0 or w == 0:
                raise ValueError(f"异常尺寸: {image.shape} @ {image_path}")
            # 以长边为 target_size 等比缩放
            if h >= w:
                new_h, new_w = target_size, max(1, int(w * (target_size / h)))
            else:
                new_w, new_h = target_size, max(1, int(h * (target_size / w)))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
            # 注意 cv2.resize 的 dsize=(width,height)
            resized = cv2.resize(image, (int(target_size[0]), int(target_size[1])), interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError(f"非法 target_size: {target_size}")

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor, original
    
    def create_synthetic_test_images(self):
        """创建合成测试图像，用于验证不同区域的处理效果"""
        device = self.device
        size = 128  # 减小尺寸便于测试
        
        # 1. 边缘测试图像 - 强对比度边缘
        edge_image = torch.zeros(1, 3, size, size, device=device)
        # 大方块
        edge_image[:, :, 30:98, 30:98] = 1.0
        # 中方块
        edge_image[:, :, 45:83, 45:83] = 0.0
        # 小方块
        edge_image[:, :, 55:73, 55:73] = 1.0
        # 添加一些线条
        edge_image[:, :, 20, :] = 0.8
        edge_image[:, :, :, 20] = 0.8
        
        # 2. 纹理测试图像 - 棋盘纹理
        texture_image = torch.zeros(1, 3, size, size, device=device)
        for i in range(0, size, 8):
            for j in range(0, size, 8):
                if (i//8 + j//8) % 2 == 0:
                    texture_image[:, :, i:min(i+8, size), j:min(j+8, size)] = 1.0
        
        # 3. 平滑区域测试图像 - 径向渐变
        smooth_image = torch.ones(1, 3, size, size, device=device) * 0.5
        center_x, center_y = size // 2, size // 2
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(size, device=device, dtype=torch.float32),
            torch.arange(size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 计算到中心的距离
        dist = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # 创建径向渐变
        gradient = 0.5 + 0.3 * torch.sin(dist / 10)
        smooth_image[0, :, :, :] = gradient.unsqueeze(0).expand(3, -1, -1)
        
        # 4. 噪声测试图像
        noise_image = torch.ones(1, 3, size, size, device=device) * 0.5
        noise = torch.randn(1, 3, size, size, device=device) * 0.2
        noise_image = torch.clamp(noise_image + noise, 0, 1)
        
        # 5. 混合特征图像 - 包含所有类型
        mixed_image = torch.zeros(1, 3, size, size, device=device)
        
        # 左上：边缘
        h_half, w_half = size // 2, size // 2
        mixed_image[:, :, :h_half, :w_half] = edge_image[:, :, :h_half, :w_half]
        
        # 右上：纹理
        mixed_image[:, :, :h_half, w_half:] = texture_image[:, :, :h_half, w_half:]
        
        # 左下：平滑
        mixed_image[:, :, h_half:, :w_half] = smooth_image[:, :, h_half:, :w_half]
        
        # 右下：噪声
        mixed_image[:, :, h_half:, w_half:] = noise_image[:, :, h_half:, w_half:]
        
        return {
            'edge': edge_image,
            'texture': texture_image, 
            'smooth': smooth_image,
            'noise': noise_image,
            'mixed': mixed_image
        }
    
    def analyze_fractional_orders(self, image_tensor, image_name="test"):
        """
        分析分数阶次分布
        
        Args:
            image_tensor: 输入图像张量
            image_name: 图像名称
        """
        with torch.no_grad():
            # 获取分数阶次
            outputs = self.model(image_tensor)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                fractional_orders = outputs[1]
            else:
                raise RuntimeError("模型输出格式不符合预期")
            
            # 转换为numpy
            orders_np = fractional_orders.squeeze(0).cpu().numpy()  # [8, H, W]
            
            # 统计信息
            print(f"\n=== {image_name} 分数阶次分析 ===")
            print(f"整体范围: [{orders_np.min():.3f}, {orders_np.max():.3f}]")
            print(f"平均值: {orders_np.mean():.3f}")
            print(f"标准差: {orders_np.std():.3f}")
            
            # 各方向统计
            direction_names = ['右', '右下', '下', '左下', '左', '左上', '上', '右上']
            for i, name in enumerate(direction_names):
                mean_val = orders_np[i].mean()
                std_val = orders_np[i].std()
                print(f"{name}方向: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
            
            return orders_np
    
    def visualize_fractional_orders(self, image_tensor, save_path=None):
        """
        可视化分数阶次分布
        
        Args:
            image_tensor: 输入图像张量
            save_path: 保存路径
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            fractional_orders = outputs[1]
            
            # 转换为numpy
            orders_np = fractional_orders.squeeze(0).cpu().numpy()
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 增强可视化效果
            orders_np_enhanced = orders_np.copy()
            
            # 计算更有意义的显示范围
            vmin = np.percentile(orders_np, 5)
            vmax = np.percentile(orders_np, 95)
            vrange = max(abs(vmin), abs(vmax))
            
            # 创建子图 - 调整布局
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
            # 原始图像
            axes[0, 0].imshow(image_np, cmap='gray')
            axes[0, 0].set_title('原始图像', fontsize=14)
            axes[0, 0].axis('off')
            
            # 显示平均分数阶次
            avg_orders = orders_np.mean(axis=0)
            im_avg = axes[0, 1].imshow(avg_orders, cmap='RdBu_r', vmin=-vrange, vmax=vrange)
            axes[0, 1].set_title('平均分数阶次', fontsize=14)
            axes[0, 1].axis('off')
            plt.colorbar(im_avg, ax=axes[0, 1], fraction=0.046)
            
            # 显示阶次标准差（表示方向性差异）
            std_orders = orders_np.std(axis=0)
            im_std = axes[0, 2].imshow(std_orders, cmap='viridis')
            axes[0, 2].set_title('方向差异性', fontsize=14)
            axes[0, 2].axis('off')
            plt.colorbar(im_std, ax=axes[0, 2], fraction=0.046)
            
            # 显示主导方向
            dominant_direction = np.argmax(np.abs(orders_np), axis=0)
            direction_names = ['右', '右下', '下', '左下', '左', '左上', '上', '右上']
            im_dom = axes[0, 3].imshow(dominant_direction, cmap='tab10', vmin=0, vmax=7)
            axes[0, 3].set_title('主导方向', fontsize=14)
            axes[0, 3].axis('off')
            plt.colorbar(im_dom, ax=axes[0, 3], fraction=0.046)
            
            # 8个方向的分数阶次
            for i in range(8):
                row = (i // 4) + 1
                col = i % 4
                
                im = axes[row, col].imshow(orders_np[i], cmap='RdBu_r', vmin=-vrange, vmax=vrange)
                axes[row, col].set_title(f'{direction_names[i]}方向 (μ={orders_np[i].mean():.3f})', fontsize=12)
                axes[row, col].axis('off')
                
                # 添加颜色条
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果保存至: {save_path}")
            
            plt.show()
            
            # 打印统计信息
            print(f"分数阶次统计:")
            print(f"  整体范围: [{orders_np.min():.4f}, {orders_np.max():.4f}]")
            print(f"  平均值: {orders_np.mean():.4f}")
            print(f"  标准差: {orders_np.std():.4f}")
            print(f"  方向间差异平均值: {std_orders.mean():.4f}")
    
    def evaluate_edge_preservation(self, image_tensor):
        """
        评估边缘保持能力
        
        Args:
            image_tensor: 输入图像张量
            
        Returns:
            edge_scores: 边缘保持评分
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            fractional_orders = outputs[1]
            
            # 计算图像梯度
            gray_image = image_tensor.mean(dim=1, keepdim=True)
            
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                 device=self.device, dtype=torch.float32)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                 device=self.device, dtype=torch.float32)
            
            grad_x = F.conv2d(gray_image, sobel_x, padding=1)
            grad_y = F.conv2d(gray_image, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            # 分析边缘区域的分数阶次
            edge_threshold = edge_magnitude.quantile(0.8)  # 前20%强边缘
            edge_mask = edge_magnitude > edge_threshold
            
            edge_orders = fractional_orders[edge_mask.expand_as(fractional_orders)]
            smooth_orders = fractional_orders[~edge_mask.expand_as(fractional_orders)]
            
            # 计算评分
            edge_order_mean = edge_orders.mean().item()
            smooth_order_mean = smooth_orders.mean().item()
            
            # 边缘区域应该有更大的阶次
            edge_enhancement_score = edge_order_mean - smooth_order_mean
            
            print(f"\n=== 边缘保持评估 ===")
            print(f"边缘区域平均阶次: {edge_order_mean:.3f}")
            print(f"平滑区域平均阶次: {smooth_order_mean:.3f}")
            print(f"边缘增强评分: {edge_enhancement_score:.3f}")
            
            return edge_enhancement_score
    
    def compare_with_baseline(self, image_tensor, baseline_method='bilinear'):
        """
        与基线方法比较
        
        Args:
            image_tensor: 输入图像张量
            baseline_method: 基线方法 ('bilinear', 'bicubic', 'nearest')
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            fractional_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # 基线方法处理
            baseline_output = F.interpolate(image_tensor, scale_factor=2, mode=baseline_method)
            
            # 计算PSNR和SSIM（需要真实的高分辨率参考图像）
            # 这里我们计算处理后图像的统计特性
            
            print(f"\n=== 与{baseline_method}方法比较 ===")
            print(f"分数阶方法输出形状: {fractional_output.shape}")
            print(f"基线方法输出形状: {baseline_output.shape}")
            
            # 计算图像质量指标（基于自参考）
            fractional_var = fractional_output.var().item()
            baseline_var = baseline_output.var().item()
            
            print(f"分数阶方法方差: {fractional_var:.6f}")
            print(f"基线方法方差: {baseline_var:.6f}")
            
            return fractional_output, baseline_output
    

    def simple_adaptation(self, test_images, num_iterations=50):
        """
        简单的自适应调整，让模型对测试图像有更好的响应
        """
        print(f"\n执行简单自适应调整 ({num_iterations} 次迭代)...")
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for iteration in range(num_iterations):
            total_loss = 0.0
            
            for name, image in test_images.items():
                optimizer.zero_grad()
                
                # 前向传播（兼容返回 (up, orders[, gate])）
                outputs = self.model(image)
                if isinstance(outputs, (tuple, list)):
                    fractional_orders = outputs[1]
                else:
                    raise RuntimeError("模型输出格式不符合预期")
                
                # 计算图像梯度
                gray_image = image.mean(dim=1, keepdim=True)
                sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]] , 
                                       device=self.device, dtype=torch.float32)
                sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]] , 
                                       device=self.device, dtype=torch.float32)
                
                grad_x = F.conv2d(gray_image, sobel_x, padding=1)
                grad_y = F.conv2d(gray_image, sobel_y, padding=1)
                edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
                
                # 边缘区域应该有更大的分数阶次
                edge_threshold = edge_magnitude.quantile(0.7)
                edge_mask = edge_magnitude > edge_threshold
                
                # 边缘/平滑区域阶次
                edge_orders = fractional_orders[edge_mask.expand_as(fractional_orders)]
                smooth_orders = fractional_orders[~edge_mask.expand_as(fractional_orders)]
                
                # 构造损失并反传
                if edge_orders.numel() > 0 and smooth_orders.numel() > 0:
                    edge_loss = -edge_orders.mean()               # 边缘更正
                    smooth_loss = smooth_orders.abs().mean()      # 内部更负/更小
                    diversity_loss = -fractional_orders.std()     # 多样性
                    loss = edge_loss + smooth_loss + 0.1 * diversity_loss
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            if iteration % 10 == 0:
                print(f"  迭代 {iteration}: 损失 = {total_loss:.4f}")

    def comprehensive_evaluation(self, test_images_dir=None):
            """综合评估 - 包含完整的图像处理效果"""
            print("=" * 60)
            print("分数阶微积分算子综合评估")
            print("=" * 60)
            
            # # 1. 合成图像测试
            # print("\n1. 合成图像测试")
            # synthetic_images = self.create_synthetic_test_images()
            
            # # 执行简单的自适应调整
            # self.simple_adaptation(synthetic_images, num_iterations=30)
            # self.model.eval()
            
            # for name, image in synthetic_images.items():
            #     print(f"\n--- {name.upper()} 图像测试 ---")
                
            #     # 分析分数阶次
            #     orders = self.analyze_fractional_orders(image, name)
                
            #     # 可视化分数阶次分布
            #     self.visualize_fractional_orders(image, f"{name}_fractional_orders.png")
                
            #     # 可视化完整处理效果
            #     stats = self.visualize_complete_processing_effect(image, f"{name}_complete_effect.png")
                
            #     print(f"处理效果: 边缘增强{stats['edge_enhancement_ratio']:.3f}倍, "
            #         f"标准差变化{(stats['output_std']-stats['input_std'])/stats['input_std']*100:.1f}%")
            
            # 2. PASCAL数据集测试
            if test_images_dir and os.path.exists(test_images_dir):
                print(f"\n2. PASCAL数据集测试 (目录: {test_images_dir})")
                image_files = [f for f in os.listdir(test_images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if len(image_files) == 0:
                    print("未找到图像文件!")
                else:
                    print(f"找到 {len(image_files)} 张图像，处理前3张...")
                    
                    for i, img_file in enumerate(image_files[:3]):  # 处理前3张
                        img_path = os.path.join(test_images_dir, img_file)
                        try:
                            print(f"\n--- 处理第{i+1}/3张PASCAL图像: {img_file} ---")
                            image_tensor, _ = self.load_test_image(img_path, target_size=None)
                            
                            # 分析分数阶次
                            self.analyze_fractional_orders(image_tensor, img_file)
                            
                            # 可视化分数阶次分布
                            save_name = f"pascal_{img_file.split('.')[0]}_fractional_orders.png"
                            self.visualize_fractional_orders(image_tensor, save_name)
                            
                            # 可视化完整处理效果
                            effect_save_name = f"pascal_{img_file.split('.')[0]}_complete_effect.png"
                            stats = self.visualize_complete_processing_effect(image_tensor, effect_save_name)
                            
                            print(f"PASCAL图像处理统计:")
                            print(f"  边缘增强倍数: {stats['edge_enhancement_ratio']:.3f}")
                            print(f"  标准差变化: {(stats['output_std']-stats['input_std'])/stats['input_std']*100:.1f}%")
                            print(f"  平均分数阶次: {stats['mean_order']:.3f}")
                            
                        except Exception as e:
                            print(f"处理 {img_file} 时出错: {e}")
                            continue
            else:
                print(f"\n2. PASCAL数据集路径不存在或未提供: {test_images_dir}")
            
            # 3. 参数敏感性分析
            print("\n3. 参数敏感性分析")
            self.parameter_sensitivity_analysis()
    
    def parameter_sensitivity_analysis(self):
        """参数敏感性分析"""
        test_image = self.create_synthetic_test_images()['edge']
        
        print("分析不同阶次范围的影响...")
        order_ranges = [(-1.0, 1.0), (-1.5, 1.5), (-2.0, 2.0)]
        
        for order_range in order_ranges:
            print(f"阶次范围 {order_range}: ", end="")
            with torch.no_grad():
                outputs = self.model(test_image)
                orders = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
                scaled_orders = orders * (order_range[1] - order_range[0]) / 3.0  # 简单缩放
                edge_regions = (scaled_orders > 0.5).sum().item()
                total_regions = scaled_orders.numel()
                print(f"高阶次区域占比: {edge_regions/total_regions:.2%}")
    
    

    def visualize_processing_effect(self, image_tensor, save_path=None):
        """
        可视化分数阶微积分处理效果（自动对齐不同分辨率）
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            # 兼容 (up, orders[, gate])
            if isinstance(outputs, (tuple, list)):
                processed_features, fractional_orders = outputs[0], outputs[1]
            else:
                raise RuntimeError("模型输出格式不符合预期")
            
            # 若输出通道与输入一致，可直接可视化；否则尝试用分数阶算子生成 effect
            if processed_features.shape[1] == image_tensor.shape[1]:
                output_for_viz = processed_features
            else:
                if hasattr(self.model, 'apply_fractional_operator'):
                    ap_out = self.model.apply_fractional_operator(image_tensor, fractional_orders)
                    output_for_viz = ap_out[0] if isinstance(ap_out, (tuple, list)) else ap_out
                else:
                    # 回退：取前3通道或通道均值
                    if processed_features.shape[1] >= 3:
                        output_for_viz = processed_features[:, :3]
                    else:
                        output_for_viz = processed_features.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                        
    def visualize_mask_effect(self, image_tensor, mask_tensor):
        """
        可视化掩膜效果
        
        Args:
            image_tensor: 原始图像 [1, 3, H, W]
            mask_tensor: 掩膜张量 [1, K^2, H, W]
        
        Returns:
            visualization: 掩膜效果可视化 [H, W]
        """
        # 计算掩膜的加权平均，权重基于图像内容
        gray_image = image_tensor.mean(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # 将掩膜重新整形为核形状
        B, K_sq, H, W = mask_tensor.shape
        K = int(np.sqrt(K_sq))
        
        # 简化：计算掩膜的方差，显示空间变化
        mask_variance = mask_tensor.var(dim=1, keepdim=True)  # [1, 1, H, W]
        
        return mask_variance.squeeze().cpu().numpy()

    def evaluate_fractional_differential_effect(self, image_tensor, image_name="test"):
        """
        评估分数阶微积分算子的实际效果
        
        Args:
            image_tensor: 输入图像张量
            image_name: 图像名称
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            processed_features, fractional_orders = outputs[0], outputs[1]
            
            print(f"\n=== {image_name} 分数阶微积分效果评估 ===")
            
            # 检查是否为掩膜输出
            if processed_features.shape[1] != image_tensor.shape[1]:
                print(f"注意: 模型输出为掩膜形式 {processed_features.shape}，而非直接的图像处理结果")
                
                # 如果有apply_fractional_operator方法，使用它
                if hasattr(self.model, 'apply_fractional_operator'):
                    actual_output = self.model.apply_fractional_operator(image_tensor, fractional_orders)
                    print("使用分数阶算子处理得到实际输出")
                else:
                    print("模型输出为自适应掩膜，无法直接比较图像处理效果")
                    return {
                        'std_change_rate': 0.0,
                        'edge_enhancement_ratio': 1.0,
                        'smoothing_ratio': 1.0,
                        'mask_diversity': processed_features.std().item()
                    }
            else:
                actual_output = processed_features
            
            # 1. 分析输入输出差异
            input_std = image_tensor.std().item()
            output_std = actual_output.std().item()
            
            print(f"输入标准差: {input_std:.4f}")
            print(f"输出标准差: {output_std:.4f}")
            print(f"标准差变化率: {(output_std - input_std) / input_std * 100:.2f}%")
            
            # 2. 分析边缘增强效果
            gray_input = image_tensor.mean(dim=1, keepdim=True)
            gray_output = actual_output.mean(dim=1, keepdim=True)
            
            # 计算边缘强度
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                device=self.device, dtype=torch.float32)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                device=self.device, dtype=torch.float32)
            
            input_edges = torch.sqrt(
                F.conv2d(gray_input, sobel_x, padding=1)**2 + 
                F.conv2d(gray_input, sobel_y, padding=1)**2 + 1e-8
            )
            
            output_edges = torch.sqrt(
                F.conv2d(gray_output, sobel_x, padding=1)**2 + 
                F.conv2d(gray_output, sobel_y, padding=1)**2 + 1e-8
            )
            
            edge_enhancement_ratio = output_edges.mean() / input_edges.mean()
            print(f"边缘增强倍数: {edge_enhancement_ratio:.4f}")
            
            # 3. 分析平滑区域效果
            smooth_threshold = input_edges.quantile(0.3)
            smooth_mask = input_edges < smooth_threshold
            
            smoothing_ratio = 1.0
            if smooth_mask.sum() > 0:
                input_smooth_var = gray_input[smooth_mask].var().item()
                output_smooth_var = gray_output[smooth_mask].var().item()
                smoothing_ratio = output_smooth_var / (input_smooth_var + 1e-8)
                print(f"平滑区域方差比率: {smoothing_ratio:.4f} (< 1.0 表示更平滑)")
            
            return {
                'std_change_rate': (output_std - input_std) / input_std,
                'edge_enhancement_ratio': edge_enhancement_ratio.item(),
                'smoothing_ratio': smoothing_ratio
            }
    
    def apply_fractional_processing(self, image_tensor, fractional_orders):
        """
        实际应用分数阶微积分算子处理图像
        
        Args:
            image_tensor: 输入图像 [B, C, H, W]
            fractional_orders: 分数阶次 [B, 8, H, W]
        
        Returns:
            processed_image: 处理后的图像 [B, C, H, W]
        """
        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        
        # 使用模型的分数阶算子处理图像
        if hasattr(self.model, 'apply_fractional_operator'):
            processed_image = self.model.apply_fractional_operator(image_tensor, fractional_orders)
        else:
            # 手动实现分数阶图像处理
            processed_image = self.manual_fractional_processing(image_tensor, fractional_orders)
        
        return processed_image

    def manual_fractional_processing(self, image_tensor, fractional_orders):
        """
        手动实现分数阶图像处理
        """
        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        
        # 创建处理后的图像
        processed = torch.zeros_like(image_tensor)
        
        # 对每个通道分别处理
        for c in range(C):
            channel_data = image_tensor[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # 应用8个方向的分数阶处理
            channel_processed = torch.zeros_like(channel_data)
            
            for dir_idx in range(8):
                v = fractional_orders[:, dir_idx, :, :]  # [B, H, W]
                
                # 简化的分数阶处理：基于阶次调整像素值
                # 正阶次：增强，负阶次：平滑
                direction_weight = torch.tanh(v).unsqueeze(1)  # [B, 1, H, W], 范围[-1,1]
                
                # 计算梯度
                sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                    device=device, dtype=torch.float32)
                sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                    device=device, dtype=torch.float32)
                
                grad_x = F.conv2d(channel_data, sobel_x, padding=1)
                grad_y = F.conv2d(channel_data, sobel_y, padding=1)
                
                # 根据方向选择梯度分量
                direction_vectors = [
                    (1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1)
                ]
                dx, dy = direction_vectors[dir_idx]
                
                directional_grad = dx * grad_x + dy * grad_y
                
                # 应用分数阶处理
                # 正阶次：增强梯度，负阶次：抑制梯度
                enhancement = channel_data + direction_weight * directional_grad * 0.1
                
                channel_processed += enhancement / 8.0  # 平均8个方向
            
            processed[:, c, :, :] = channel_processed.squeeze(1)
        
        return torch.clamp(processed, 0, 1)

    def visualize_complete_processing_effect(self, image_tensor, save_path=None):
        """
        完整的处理效果可视化，包含真正的图像处理结果
        """
        with torch.no_grad():
            # 获取分数阶次
            outputs = self.model(image_tensor)
            if isinstance(outputs, (tuple, list)):
                # up: [B,C,H*s,W*s]（此处仅用来展示形状，不直接可视化）
                up = outputs[0]
                fractional_orders = outputs[1]
                gate = outputs[2] if len(outputs) > 2 else None
            else:
                raise RuntimeError("模型输出格式不符合预期: 需返回 (up, orders[, gate])")
            
            # 应用分数阶处理（同分辨率）
            if hasattr(self.model, 'apply_fractional_operator'):
                proc = self.model.apply_fractional_operator(image_tensor, fractional_orders)
                processed_image = proc[0] if isinstance(proc, (tuple, list)) else proc
            else:
                processed_image = self.manual_fractional_processing(image_tensor, fractional_orders)
            
            # 转换为numpy
            input_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 确保都在[0,1]范围
            input_np = np.clip(input_np, 0, 1)
            output_np = np.clip(output_np, 0, 1)
            
            # 计算差异图
            diff_np = np.abs(output_np - input_np)
            
            # 创建子图
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # 第一行：图像对比
            axes[0, 0].imshow(input_np)
            axes[0, 0].set_title('原始图像', fontsize=14)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(output_np)
            axes[0, 1].set_title('分数阶处理后', fontsize=14)
            axes[0, 1].axis('off')
            
            im_diff = axes[0, 2].imshow(diff_np.mean(axis=2), cmap='hot', vmin=0, vmax=0.3)
            axes[0, 2].set_title('处理差异', fontsize=14)
            axes[0, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)
            
            # 边缘检测对比
            gray_input = image_tensor.mean(dim=1, keepdim=True)
            gray_output = torch.from_numpy(output_np).permute(2,0,1).unsqueeze(0).to(image_tensor.device).float()
            gray_output = gray_output.mean(dim=1, keepdim=True)
            
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                device=self.device, dtype=torch.float32)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                device=self.device, dtype=torch.float32)
            
            input_edges = torch.sqrt(
                F.conv2d(gray_input, sobel_x, padding=1)**2 + 
                F.conv2d(gray_input, sobel_y, padding=1)**2 + 1e-8
            ).squeeze().cpu().numpy()
            
            output_edges = torch.sqrt(
                F.conv2d(gray_output, sobel_x, padding=1)**2 + 
                F.conv2d(gray_output, sobel_y, padding=1)**2 + 1e-8
            ).squeeze().cpu().numpy()
            
            axes[0, 3].imshow(np.concatenate([input_edges, output_edges], axis=1), cmap='gray')
            axes[0, 3].set_title('边缘对比 (左:原始 右:处理后)', fontsize=14)
            axes[0, 3].axis('off')
            
            # 第二行：分数阶次分析
            orders_mean = fractional_orders.squeeze(0).mean(dim=0).cpu().numpy()
            im_orders = axes[1, 0].imshow(orders_mean, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
            axes[1, 0].set_title('平均分数阶次', fontsize=14)
            axes[1, 0].axis('off')
            plt.colorbar(im_orders, ax=axes[1, 0], fraction=0.046)
            
            # 阶次标准差
            orders_std = fractional_orders.squeeze(0).std(dim=0).cpu().numpy()
            im_std = axes[1, 1].imshow(orders_std, cmap='viridis')
            axes[1, 1].set_title('方向差异性', fontsize=14)
            axes[1, 1].axis('off')
            plt.colorbar(im_std, ax=axes[1, 1], fraction=0.046)
            
            # 分数阶次分布直方图
            orders_flat = fractional_orders.cpu().numpy().flatten()
            axes[1, 2].hist(orders_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 2].set_xlabel('分数阶次值')
            axes[1, 2].set_ylabel('频次')
            axes[1, 2].set_title('分数阶次分布')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 处理效果统计
            input_std = input_np.std()
            output_std = output_np.std()
            edge_enhancement = output_edges.mean() / (input_edges.mean() + 1e-8)
            
            stats_text = f"""处理效果统计:
输入标准差: {input_std:.4f}
输出标准差: {output_std:.4f}
标准差变化: {(output_std-input_std)/input_std*100:.1f}%
边缘增强倍数: {edge_enhancement:.3f}
平均分数阶次: {orders_mean.mean():.3f}
阶次标准差: {orders_mean.std():.3f}"""
            
            axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes, 
                            fontsize=10, verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 3].set_title('处理统计', fontsize=14)
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"完整处理效果可视化保存至: {save_path}")
            
            plt.show()
            
            return {
                'input_std': input_std,
                'output_std': output_std,
                'edge_enhancement_ratio': edge_enhancement,
                'mean_order': orders_mean.mean(),
                'order_std': orders_mean.std()
            }
    


def create_demo_images():
    """创建演示图像用于测试"""
    demo_dir = "demo_images"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 创建测试图像
    size = 256
    
    # 1. 棋盘图案（强边缘）
    checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, 32):
        for j in range(0, size, 32):
            if (i//32 + j//32) % 2 == 0:
                checkerboard[i:i+32, j:j+32] = 255
    cv2.imwrite(os.path.join(demo_dir, "checkerboard.png"), checkerboard)
    
    # 2. 圆形图案（边缘+平滑）
    circle_img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(circle_img, (size//2, size//2), size//3, (255, 255, 255), -1)
    cv2.circle(circle_img, (size//2, size//2), size//4, (128, 128, 128), -1)
    cv2.imwrite(os.path.join(demo_dir, "circles.png"), circle_img)
    
    # 3. 噪声图像
    noise_img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(demo_dir, "noise.png"), noise_img)
    
    print(f"演示图像已创建在 {demo_dir} 目录中")
    return demo_dir


def main_evaluation():
    """主评估流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 若你只是演示 orders 与 effect，可把 up_factor 设为 1，避免可视化中反复 resize
    mask_generator = FractionalFusionUp(
        in_channels=3,
        up_factor=1,          # 如需避免可视化对齐，可改为 1
        kernel_size=3,
        vmax=1.0,
        hidden=48,
        beta=1.4,
        mode='both'
    ).to(device)

    evaluator = FractionalMaskEvaluator(mask_generator, device)

    # 如果 pascal 路径不存在，自动仅跑合成图
    pascal_dir = r"D:\pythonRepo\PASCAL_AUG\VOC2012\JPEGImages"
    if not os.path.isdir(pascal_dir):
        print(f"警告: 未找到 PASCAL 目录 {pascal_dir}，仅运行合成图评估。")
        pascal_dir = None

    evaluator.comprehensive_evaluation(pascal_dir)

if __name__ == "__main__":
    main_evaluation()