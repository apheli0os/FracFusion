#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# 克隆 GitHub 仓库
get_ipython().system('git clone https://github.com/LiheYoung/UniMatch.git')

# 切换工作目录到项目根目录
# 这对于脚本正确找到配置文件和模块非常重要
project_dir = '/kaggle/working/UniMatch'
os.chdir(project_dir)

print(f"当前工作目录: {os.getcwd()}")


# In[ ]:


# 安装项目所需的依赖包
# -r requirements.txt 会自动安装文件里列出的所有库
get_ipython().system('pip install -r requirements.txt')




# In[ ]:


# 新增 Cell: 复制预训练模型到项目目录

import os
import shutil

# 确保在 UniMatch 项目目录中
project_dir = '/kaggle/working/UniMatch'
os.chdir(project_dir)

# 创建项目的 pretrained 目录
pretrained_dir = 'pretrained'
os.makedirs(pretrained_dir, exist_ok=True)

# 源文件路径（您上传的模型）
source_model_path = '/kaggle/input/pretrained/resnet101.pth'

# 目标文件路径（项目期望的位置）
target_model_path = os.path.join(pretrained_dir, 'resnet101.pth')

# 检查源文件是否存在
if os.path.exists(source_model_path):
    print(f"✅ 找到预训练模型: {source_model_path}")

    # 复制文件到项目目录
    shutil.copy2(source_model_path, target_model_path)

    print(f"✅ 预训练模型已复制到: {target_model_path}")
    print(f"文件大小: {os.path.getsize(target_model_path) / (1024*1024):.1f} MB")

    # 验证复制是否成功
    if os.path.exists(target_model_path):
        print("✅ 预训练模型准备完成，可以开始训练了！")
    else:
        print("❌ 文件复制失败！")
else:
    print(f"❌ 未找到预训练模型文件: {source_model_path}")
    print("请确认您的 Kaggle 数据集名称和文件路径是否正确。")

# 显示 pretrained 目录的内容
print(f"\npretrained 目录内容:")
if os.path.exists(pretrained_dir):
    for file in os.listdir(pretrained_dir):
        file_path = os.path.join(pretrained_dir, file)
        file_size = os.path.getsize(file_path) / (1024*1024)
        print(f"  - {file} ({file_size:.1f} MB)")
else:
    print("  (目录不存在)")


# In[ ]:


# 新增 Cell: 优化训练配置以提高速度

import yaml
import os

# 确保在正确目录
os.chdir('/kaggle/working/UniMatch')

# 读取配置文件
config_file_path = 'configs/pascal.yaml'
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

print("=== 当前训练配置 ===")
print(f"批次大小 (batch_size): {config_data.get('batch_size', '未设置')}")
print(f"学习率 (lr): {config_data.get('lr', '未设置')}")
print(f"裁剪尺寸 (crop_size): {config_data.get('crop_size', '未设置')}")

# === 优化配置 ===
print("\n=== 优化配置 ===")

# 1. 增加批次大小（原来可能是2，我们可以增加到6-8）
original_batch_size = config_data.get('batch_size', 2)
optimized_batch_size = 8  # 根据15GB显存，可以尝试8
config_data['batch_size'] = optimized_batch_size
config_data['backbone'] = 'resnet101'
# 2. 相应调整学习率（批次大小增加时，通常需要线性增加学习率）
original_lr = config_data.get('lr', 0.001)
lr_scale_factor = optimized_batch_size / original_batch_size
optimized_lr = original_lr * lr_scale_factor
config_data['lr'] = optimized_lr

# 3. 可选：增加图像尺寸（如果显存允许）
# config_data['crop_size'] = 513  # 从321增加到513，但这会增加显存使用

print(f"批次大小: {original_batch_size} → {optimized_batch_size}")
print(f"学习率: {original_lr} → {optimized_lr:.6f}")
print(f"学习率缩放因子: {lr_scale_factor}")
#print(f"学习率缩放因子: {lr_scale_factor}")

# 保存优化后的配置
with open(config_file_path, 'w') as f:
    yaml.dump(config_data, f)

print("✅ 配置优化完成！")


# In[ ]:


# 修复 Cell: 正确设置配置参数

import yaml
import os

# 确保在正确目录
os.chdir('/kaggle/working/UniMatch')

# 读取配置文件
config_file_path = 'configs/pascal.yaml'
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

print("=== 当前训练配置（修复前）===")
print(f"批次大小 (batch_size): {config_data.get('batch_size', '未设置')} (类型: {type(config_data.get('batch_size', '未设置'))})")
print(f"骨干网络 (backbone): {config_data.get('backbone', '未设置')}")
print(f"学习率 (lr): {config_data.get('lr', '未设置')}")
print(f"裁剪尺寸 (crop_size): {config_data.get('crop_size', '未设置')}")

# === 修复和优化配置 ===
print("\n=== 修复和优化配置 ===")

# 1. 正确设置批次大小（修复之前的错误）
config_data['batch_size'] = 6  # 设置为正确的数字类型

# 2. 正确设置骨干网络
config_data['backbone'] = 'resnet101'

# 3. 重新计算学习率
original_batch_size = 2  # 原始的基础批次大小
current_batch_size = 8   # 当前设置的批次大小
base_lr = 0.001          # 基础学习率

# 根据批次大小线性调整学习率
lr_scale_factor = current_batch_size / original_batch_size
optimized_lr = base_lr * lr_scale_factor
config_data['lr'] = optimized_lr

print(f"批次大小: 修复为 {config_data['batch_size']} (数字类型)")
print(f"骨干网络: {config_data['backbone']}")
print(f"学习率: {base_lr} → {optimized_lr:.6f}")
print(f"学习率缩放因子: {lr_scale_factor}")

# 4. 确保其他重要参数正确
config_data['crop_size'] = 321  # 确保crop_size是数字
config_data['nclass'] = 21      # PASCAL VOC的类别数

# 保存优化后的配置
with open(config_file_path, 'w') as f:
    yaml.dump(config_data, f)

print("\n=== 修复后的配置 ===")
print(f"批次大小 (batch_size): {config_data['batch_size']} (类型: {type(config_data['batch_size'])})")
print(f"骨干网络 (backbone): {config_data['backbone']}")
print(f"学习率 (lr): {config_data['lr']}")
print(f"裁剪尺寸 (crop_size): {config_data['crop_size']}")

print("✅ 配置修复和优化完成！")


# In[ ]:


# 新增 Cell: 数据加载和训练优化

import yaml
import os

os.chdir('/kaggle/working/UniMatch')

config_file_path = 'configs/pascal.yaml'
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

# 数据加载优化
data_loading_optimizations = {
    'num_workers': 6,  # 增加数据加载工作进程
    'pin_memory': True,  # 启用固定内存
    'persistent_workers': True,  # 保持工作进程活跃
    'prefetch_factor': 4,  # 预取因子
}

print("=== 数据加载优化 ===")
for key, value in data_loading_optimizations.items():
    old_value = config_data.get(key, '未设置')
    config_data[key] = value
    print(f"{key}: {old_value} → {value}")

# 保存配置
with open(config_file_path, 'w') as f:
    yaml.dump(config_data, f)

print("✅ 数据加载优化已应用")


# In[ ]:


# 新增 Cell: 调整验证频率

import yaml
import os

os.chdir('/kaggle/working/UniMatch')

config_file_path = 'configs/pascal.yaml'
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

# 调整验证频率
# 原来可能每个epoch都验证，现在改为每2-3个epoch验证一次
config_data['eval_freq'] = 5  # 每2个epoch验证一次
config_data['epochs'] = 80  # 减少总轮数
print(f"验证频率: 每个epoch → 每{config_data['eval_freq']}个epoch")
print("预期加速: 减少验证时间约50%")

# 保存配置
with open(config_file_path, 'w') as f:
    yaml.dump(config_data, f)

print("✅ 验证频率已调整")


# In[ ]:


# 最终修复脚本 - 确保所有torch.load都有weights_only=False

import os
import re

os.chdir('/kaggle/working/UniMatch')

def comprehensive_torch_load_fix():
    """全面修复torch.load兼容性问题"""

    # 首先删除有问题的checkpoint文件，重新开始训练
    problematic_checkpoint = '/kaggle/working/exp/pascal/unimatch/r101/732/latest.pth'
    if os.path.exists(problematic_checkpoint):
        os.remove(problematic_checkpoint)
        print(f"✅ 已删除有问题的checkpoint: {problematic_checkpoint}")

    # 修复所有Python文件中的torch.load
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"找到 {len(python_files)} 个Python文件")

    fixed_files = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复torch.load调用 - 更精确的正则表达式
            # 1. 没有weights_only参数的torch.load
            pattern1 = r'torch\.load\(([^)]*)\)(?![^,]*weights_only)'

            def replace_torch_load(match):
                args = match.group(1).strip()
                if not args:
                    return 'torch.load(weights_only=False)'
                elif args.endswith(','):
                    return f'torch.load({args} weights_only=False)'
                else:
                    return f'torch.load({args}, weights_only=False)'

            content = re.sub(pattern1, replace_torch_load, content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(file_path)

        except Exception as e:
            print(f"⚠️  无法处理文件 {file_path}: {e}")

    print(f"✅ 已修复 {len(fixed_files)} 个文件:")
    for file_path in fixed_files:
        print(f"  - {file_path}")

    return fixed_files

# 运行修复
fixed_files = comprehensive_torch_load_fix()

# 特别检查关键文件
key_files = ['unimatch.py', 'train_unimatch_ftv.py']
print("\n=== 关键文件检查 ===")

for file_path in key_files:
    if os.path.exists(file_path):
        print(f"\n{file_path}:")
        with open(file_path, 'r') as f:
            lines = f.readlines()

        torch_load_lines = []
        for i, line in enumerate(lines):
            if 'torch.load(' in line:
                torch_load_lines.append((i+1, line.strip()))

        if torch_load_lines:
            for line_num, line_content in torch_load_lines:
                print(f"  第{line_num}行: {line_content}")
        else:
            print("  未找到torch.load调用")

print("\n🎉 PyTorch 2.6兼容性修复完成！")
print("现在可以重新运行您的训练命令。")


# In[ ]:


# Cell 引入分数阶增强
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import cv2
from PIL import Image
import requests
from io import BytesIO
import os


plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
# 使用您已有的分数阶系数计算模块
class FractionalCoefficients:
    """基于您提供公式(18)的分数阶系数计算类"""

    @staticmethod
    def compute_coefficients(v, max_order=10):
        """计算分数阶系数 C_k"""
        coeffs = {}

        # 基于您提供的公式计算
        coeffs[-1] = v/4 + (v**2)/8                    # C_{-1}
        coeffs[0] = 1 - (v**2)/8 - (v**3)/8            # C_0  
        coeffs[1] = -5*v/4 + 5*(v**2)/16 + (v**4)/16   # C_1

        # 通用公式计算更高阶项
        for k in range(2, max_order + 1):
            try:
                c_k = (gamma(k-v+1) / (gamma(k+1) * gamma(-v))) * (v/4 + v**2/8)
                c_k += (gamma(k-v) / gamma(k+1)) * (1 - v**2/4)
                c_k += (gamma(k-v-1) / (gamma(k-1) * gamma(-v))) * (-v/4 + v**2/8)
                coeffs[k] = c_k
            except:
                # Gamma函数计算失败时使用递推关系
                coeffs[k] = coeffs[k-1] * (k-v) / k

        return coeffs

class FractionalMaskGenerator:
    """8方向分数阶掩膜生成器"""

    def __init__(self, v=0.5, mask_size=7):
        self.v = v
        self.mask_size = mask_size
        self.coeffs = FractionalCoefficients.compute_coefficients(v, mask_size//2)

    def generate_8_direction_masks(self):
        """生成8个方向的分数阶掩膜"""
        masks = {}
        center = self.mask_size // 2

        # 水平方向
        mask_h = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                mask_h[center, center + i] = coeff
        masks['horizontal'] = mask_h

        # 垂直方向
        mask_v = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                mask_v[center + i, center] = coeff
        masks['vertical'] = mask_v

        # 主对角线方向
        mask_d1 = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                if 0 <= center + i < self.mask_size and 0 <= center + i < self.mask_size:
                    mask_d1[center + i, center + i] = coeff
        masks['diagonal_main'] = mask_d1

        # 反对角线方向
        mask_d2 = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                if 0 <= center + i < self.mask_size and 0 <= center - i < self.mask_size:
                    mask_d2[center + i, center - i] = coeff
        masks['diagonal_anti'] = mask_d2

        # 额外的4个方向用于8方向增强
        # 东北方向 (45度)
        mask_ne = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                row, col = center - i, center + i
                if 0 <= row < self.mask_size and 0 <= col < self.mask_size:
                    mask_ne[row, col] = coeff
        masks['northeast'] = mask_ne

        # 西北方向 (135度)
        mask_nw = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                row, col = center - i, center - i
                if 0 <= row < self.mask_size and 0 <= col < self.mask_size:
                    mask_nw[row, col] = coeff
        masks['northwest'] = mask_nw

        # 东南方向 (-45度)
        mask_se = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                row, col = center + i, center + i
                if 0 <= row < self.mask_size and 0 <= col < self.mask_size:
                    mask_se[row, col] = coeff
        masks['southeast'] = mask_se

        # 西南方向 (-135度)
        mask_sw = torch.zeros(self.mask_size, self.mask_size)
        for i, coeff in self.coeffs.items():
            if abs(i) <= center:
                row, col = center + i, center - i
                if 0 <= row < self.mask_size and 0 <= col < self.mask_size:
                    mask_sw[row, col] = coeff
        masks['southwest'] = mask_sw

        return masks

class FractionalImageEnhancer:
    """
    模块化分数阶图像增强类

    此类提供了使用8方向卷积掩膜进行分数阶图像增强的完整解决方案。

    参数:
        fractional_order (float): 分数阶参数 v (0.1 到 2.0)
        mask_size (int): 卷积掩膜大小 (默认: 7)
        enhancement_factor (float): 增强强度 (默认: 0.3)
        enhancement_mode (str): 增强策略 ('average', 'weighted', 'selective')
    """

    def __init__(self, fractional_order=0.5, mask_size=7, enhancement_factor=0.3, enhancement_mode='weighted'):
        self.fractional_order = fractional_order
        self.mask_size = mask_size
        self.enhancement_factor = enhancement_factor
        self.enhancement_mode = enhancement_mode

        # 生成分数阶掩膜
        self.mask_generator = FractionalMaskGenerator(fractional_order, mask_size)
        self.masks = self.mask_generator.generate_8_direction_masks()

        # 加权增强的方向权重
        self.direction_weights = {
            'horizontal': 1.0,
            'vertical': 1.0,
            'diagonal_main': 0.8,
            'diagonal_anti': 0.8,
            'northeast': 0.6,
            'northwest': 0.6,
            'southeast': 0.6,
            'southwest': 0.6
        }

    def _preprocess_image(self, image):
        """将输入图像预处理为标准格式"""
        # 处理不同的输入格式
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)

        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0

        # 确保正确的张量格式 [C, H, W]
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # HWC 格式
                image = image.permute(2, 0, 1)  # 转换为 CHW
        elif len(image.shape) == 2:  # 灰度图
            image = image.unsqueeze(0)  # 添加通道维度

        return image

    def _extract_directional_features(self, image):
        """提取所有8个方向的纹理特征"""
        # 转换为灰度图进行处理
        if image.shape[0] == 3:  # RGB
            gray_image = torch.mean(image, dim=0, keepdim=True)
        else:
            gray_image = image

        # 为卷积添加批次维度
        gray_image = gray_image.unsqueeze(0)  # [1, 1, H, W]

        directional_features = {}

        # 应用每个掩膜
        for direction, mask in self.masks.items():
            conv_kernel = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, mask_size, mask_size]
            padding = self.mask_size // 2

            # 与分数阶掩膜卷积
            feature = F.conv2d(gray_image, conv_kernel, padding=padding)
            directional_features[direction] = feature.squeeze()

        return directional_features

    def _combine_features(self, directional_features):
        """根据增强模式组合方向特征"""
        feature_list = list(directional_features.values())

        if self.enhancement_mode == 'average':
            # 简单平均所有方向
            combined = torch.stack(feature_list).mean(dim=0)

        elif self.enhancement_mode == 'weighted':
            # 基于方向重要性的加权组合
            combined = torch.zeros_like(feature_list[0])
            total_weight = 0

            for direction, feature in directional_features.items():
                weight = self.direction_weights.get(direction, 1.0)
                combined += weight * feature
                total_weight += weight

            combined = combined / total_weight

        elif self.enhancement_mode == 'selective':
            # 自适应选择最强响应
            feature_stack = torch.stack(feature_list)  # [8, H, W]

            # 计算每个方向的局部方差
            variances = torch.var(feature_stack, dim=(1, 2))

            # 根据方差加权（纹理更强的方向权重更高）
            weights = F.softmax(variances * 10, dim=0)  # 温度缩放

            combined = torch.zeros_like(feature_list[0])
            for i, feature in enumerate(feature_list):
                combined += weights[i] * feature

        else:
            raise ValueError(f"未知的增强模式: {self.enhancement_mode}")

        return combined

    def enhance_image(self, image):
        """
        主要增强函数

        参数:
            image: 输入图像 (PIL.Image, numpy.ndarray, 或 torch.Tensor)
                  - 对于numpy: 形状应为 (H, W) 或 (H, W, C)
                  - 对于tensor: 形状应为 (C, H, W) 或 (H, W)

        返回:
            enhanced_image: 增强后的图像 torch.Tensor [C, H, W]
            enhancement_map: 增强细节图
            directional_features: 方向特征字典
        """
        # 预处理输入
        original_image = self._preprocess_image(image)
        original_shape = original_image.shape

        # 提取方向特征
        directional_features = self._extract_directional_features(original_image)

        # 组合特征
        enhancement_map = self._combine_features(directional_features)

        # 应用增强
        if len(original_shape) == 3:  # 彩色图像
            enhanced_image = original_image.clone()
            for c in range(original_image.shape[0]):
                enhanced_image[c] += self.enhancement_factor * enhancement_map
        else:  # 灰度图
            enhanced_image = original_image + self.enhancement_factor * enhancement_map

        # 将值限制在有效范围内
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        return enhanced_image, enhancement_map, directional_features

    def set_parameters(self, fractional_order=None, enhancement_factor=None, enhancement_mode=None):
        """更新增强参数"""
        if fractional_order is not None:
            self.fractional_order = fractional_order
            # 用新的阶数重新生成掩膜
            self.mask_generator = FractionalMaskGenerator(fractional_order, self.mask_size)
            self.masks = self.mask_generator.generate_8_direction_masks()

        if enhancement_factor is not None:
            self.enhancement_factor = enhancement_factor

        if enhancement_mode is not None:
            self.enhancement_mode = enhancement_mode


# In[ ]:


# Cell: 数据集分数阶增强预处理
import os
import yaml
from PIL import Image
import torch
from tqdm import tqdm
import shutil

def preprocess_dataset_with_fractional_enhancement():
    """
    对整个数据集进行分数阶增强预处理
    """
    print("🚀 开始对数据集进行分数阶增强预处理...")

    # 1. 设置路径
    original_data_root = '/kaggle/input/pascal1-3/VOC2012'
    enhanced_data_root = '/kaggle/working/enhanced_VOC2012'

    original_img_dir = os.path.join(original_data_root, 'JPEGImages')
    enhanced_img_dir = os.path.join(enhanced_data_root, 'JPEGImages')

    # 2. 创建增强后的数据目录
    os.makedirs(enhanced_img_dir, exist_ok=True)

    # 3. 复制其他必要目录（标签等）
    for dir_name in ['SegmentationClass', 'SegmentationObject', 'ImageSets']:
        src_dir = os.path.join(original_data_root, dir_name)
        dst_dir = os.path.join(enhanced_data_root, dir_name)
        if os.path.exists(src_dir):
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"✅ 复制目录: {dir_name}")

    # 4. 初始化分数阶增强器
    enhancer = FractionalImageEnhancer(
        fractional_order=0.6,          # 分数阶参数
        enhancement_factor=0.2,        # 适中的增强强度
        enhancement_mode='weighted'    # 加权模式
    )

    # 5. 获取所有图像文件
    img_files = [f for f in os.listdir(original_img_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"找到 {len(img_files)} 张图像需要处理")

    # 6. 批量处理图像
    processed_count = 0
    failed_count = 0

    for img_file in tqdm(img_files, desc="处理图像"):
        try:
            # 加载原始图像
            img_path = os.path.join(original_img_dir, img_file)
            original_img = Image.open(img_path).convert('RGB')

            # 应用分数阶增强
            enhanced_tensor, _, _ = enhancer.enhance_image(original_img)

            # 转换回PIL图像
            enhanced_array = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            enhanced_img = Image.fromarray(enhanced_array)

            # 保存增强后的图像
            enhanced_path = os.path.join(enhanced_img_dir, img_file)
            enhanced_img.save(enhanced_path, quality=95)

            processed_count += 1

        except Exception as e:
            print(f"❌ 处理图像失败 {img_file}: {str(e)}")
            failed_count += 1

            # 如果增强失败，复制原图像
            try:
                shutil.copy2(img_path, os.path.join(enhanced_img_dir, img_file))
                processed_count += 1
            except:
                pass

    print(f"\n✅ 预处理完成:")
    print(f"   成功处理: {processed_count} 张图像")
    print(f"   失败: {failed_count} 张图像")
    print(f"   增强后数据集位置: {enhanced_data_root}")

    return enhanced_data_root

def update_config_for_enhanced_data(enhanced_data_root):
    """
    更新配置文件以使用增强后的数据集
    """
    print("\n🔧 更新配置文件...")

    # 确保在正确目录
    project_dir = '/kaggle/working/UniMatch'
    os.chdir(project_dir)

    config_file_path = 'configs/pascal.yaml'
    with open(config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # 更新数据根路径
    print(f"原数据路径: {config_data.get('data_root', '未设置')}")
    config_data['data_root'] = enhanced_data_root
    print(f"新数据路径: {enhanced_data_root}")

    # 保存配置
    with open(config_file_path, 'w') as f:
        yaml.dump(config_data, f)

    print("✅ 配置文件更新完成")
    return config_data

def create_enhanced_splits():
    """
    创建增强后数据集的splits文件
    """
    print("\n📝 创建增强后数据集的splits...")

    project_dir = '/kaggle/working/UniMatch'
    os.chdir(project_dir)

    # 读取原始splits
    dataset = 'pascal'
    split = '732'

    labeled_id_path = f'splits/{dataset}/{split}/labeled.txt'
    unlabeled_id_path = f'splits/{dataset}/{split}/unlabeled.txt'

    # 验证splits文件存在
    if os.path.exists(labeled_id_path) and os.path.exists(unlabeled_id_path):
        print(f"✅ 使用现有的splits文件:")
        print(f"   标记数据: {labeled_id_path}")
        print(f"   无标记数据: {unlabeled_id_path}")

        # 统计数量
        with open(labeled_id_path, 'r') as f:
            labeled_count = len(f.readlines())
        with open(unlabeled_id_path, 'r') as f:
            unlabeled_count = len(f.readlines())

        print(f"   标记样本数: {labeled_count}")
        print(f"   无标记样本数: {unlabeled_count}")

        return True
    else:
        print("❌ splits文件不存在")
        return False

# 主预处理流程
def main_preprocessing():
    """
    主要的预处理流程
    """
    print("=" * 60)
    print("🎯 开始分数阶增强数据预处理流程")
    print("=" * 60)

    # Step 1: 对数据集进行分数阶增强
    enhanced_data_root = preprocess_dataset_with_fractional_enhancement()

    # Step 2: 更新配置文件
    config = update_config_for_enhanced_data(enhanced_data_root)

    # Step 3: 验证splits文件
    splits_ok = create_enhanced_splits()

    if splits_ok:
        print("\n🎉 预处理流程完成！现在可以开始训练了。")
        print(f"增强后的数据集路径: {enhanced_data_root}")
        return True
    else:
        print("\n❌ 预处理失败，请检查数据和配置")
        return False

# 运行预处理
# if main_preprocessing():
#     print("\n" + "=" * 60)
#     print("🚀 准备开始 UniMatch 训练...")
#     print("=" * 60)
# else:
#     print("❌ 预处理失败，无法继续训练")


# In[ ]:


# Cell: 修复 torch.load 语法错误

import os
import re

os.chdir('/kaggle/working/UniMatch')

def fix_torch_load_syntax_error():
    """修复torch.load中重复的weights_only参数"""

    # 查找所有Python文件
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"检查 {len(python_files)} 个Python文件...")

    fixed_files = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复重复的weights_only参数
            # 找到所有torch.load调用
            pattern = r'torch\.load\(([^)]*)\)'

            def fix_weights_only(match):
                args_str = match.group(1)

                # 移除重复的weights_only参数
                # 首先移除所有weights_only参数
                args_str = re.sub(r',?\s*weights_only\s*=\s*False', '', args_str)
                args_str = re.sub(r'weights_only\s*=\s*False\s*,?', '', args_str)

                # 清理多余的逗号
                args_str = re.sub(r',\s*,', ',', args_str)
                args_str = re.sub(r'^\s*,', '', args_str)
                args_str = re.sub(r',\s*$', '', args_str)

                # 添加一个weights_only=False参数
                if args_str.strip():
                    if not args_str.strip().endswith(','):
                        args_str += ', '
                    args_str += 'weights_only=False'
                else:
                    args_str = 'weights_only=False'

                return f'torch.load({args_str})'

            # 应用修复
            content = re.sub(pattern, fix_weights_only, content)

            # 检查是否有变化
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(file_path)
                print(f"✅ 修复文件: {file_path}")

        except Exception as e:
            print(f"❌ 处理文件失败 {file_path}: {e}")

    return fixed_files

# 运行修复
fixed_files = fix_torch_load_syntax_error()

# 特别检查unimatch.py文件
unimatch_file = 'unimatch.py'
if os.path.exists(unimatch_file):
    print(f"\n=== 检查 {unimatch_file} ===")
    with open(unimatch_file, 'r') as f:
        lines = f.readlines()

    # 查找包含torch.load的行
    for i, line in enumerate(lines):
        if 'torch.load(' in line:
            print(f"第{i+1}行: {line.strip()}")

            # 检查是否有语法错误
            if line.count('weights_only') > 1:
                print(f"❌ 第{i+1}行有重复的weights_only参数")
            elif 'weights_only=False' in line:
                print(f"✅ 第{i+1}行语法正确")

print(f"\n✅ 语法错误修复完成，共修复 {len(fixed_files)} 个文件")


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class FractionalFusionUp(nn.Module):
    """
    分数阶自适应上采样：替代最终 logits 的双线性插值
    - 仅放大空间尺寸，不改变通道（类别）数
    - 预测8方向分数阶次 v ∈ [-vmax,vmax] 与门控 g ∈ [0,1]
    - 输出: y [B,C,H,W], orders [B,8,h,w], gate [B,1,H,W]
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

        dirs = torch.tensor([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]], dtype=torch.float32)
        self.register_buffer('dirs', dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8))

        self.order_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 8, 3, 1, 1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1), nn.Sigmoid()
        )

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
        if v.dim() == 3: v = v.unsqueeze(1)
        B, _, H, W = v.shape
        coeffs = [torch.ones(B,1,H,W, device=v.device, dtype=v.dtype)]
        w = coeffs[0]
        for m in range(1, M+1):
            w = w * (v - m + 1.0)/m * (-1.0)
            coeffs.append(w)
        return torch.cat(coeffs, dim=1)

    def _direction_kernel(self, v_dir, d):
        M = self.center
        coeff = self.gl_coeffs(v_dir, M)  # [B,M+1,H,W]
        base_pos = self.dir_step_base[d,0]  # [M+1,K,K]
        base_neg = self.dir_step_base[d,1]
        ker = (coeff[:, :, None, None, ...] *
               (base_pos[None, :, :, :, None, None] - base_neg[None, :, :, :, None, None])).sum(1)
        return ker  # [B,K,K,H,W]

    def _build_kernels(self, v):
        ks = [self._direction_kernel(v[:, d:d+1], d) for d in range(8)]
        return torch.stack(ks, 0).mean(0)  # [B,K,K,H,W]

    def forward(self, x):
        B, C, h, w = x.shape
        raw_v = self.order_head(x)
        g = self.gate(x)                         # [B,1,h,w]
        v = self.vmax * torch.tanh(raw_v)        # [B,8,h,w]
        ker = self._build_kernels(v)             # [B,K,K,h,w]
        K, pad = self.k, self.center

        # 低通权重（中心先验 + 温度 softmax）
        mask_pre = ker.view(B, K*K, h, w)
        center_idx = self.center * self.k + self.center
        mask_pre[:, center_idx:center_idx+1] += self.center_bias
        mask_lp = torch.softmax(mask_pre / self.tau, dim=1)

        # 邻域展开
        xp = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        patches = F.unfold(xp, kernel_size=K, stride=1, padding=0).view(B, C, K*K, h, w)

        # 放大空间维
        if self.s > 1:
            neigh   = patches.repeat_interleave(self.s, dim=3).repeat_interleave(self.s, dim=4)  # [B,C,K^2,H,W]
            mask_lp = mask_lp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)  # [B,K^2,H,W]
            g_up    = g.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)        # [B,1,H,W]
            x_up    = F.interpolate(x, scale_factor=self.s, mode='bilinear', align_corners=True) # [B,C,H,W]
        else:
            neigh, g_up, x_up = patches, g, x

        # 低通输出 + 边缘自适应残差
        y_lp_out = (neigh * mask_lp[:, None]).sum(dim=2)
        lambda_eff = self.smooth_residual * (1.0 - g_up)
        y_lp = x_up + lambda_eff * (y_lp_out - x_up)

        if self.mode == 'lowpass':
            return y_lp, v, g_up

        # 高通分支（零直流 + L1归一）
        ker_hp = ker - ker.mean(dim=(1, 2), keepdim=True)
        ker_hp = ker_hp / (ker_hp.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        mask_hp = ker_hp.view(B, K*K, h, w)
        if self.s > 1:
            mask_hp = mask_hp.repeat_interleave(self.s, dim=2).repeat_interleave(self.s, dim=3)
        y_hp = (neigh * mask_hp[:, None]).sum(dim=2)

        if self.mode == 'highpass':
            return y_hp, v, g_up

        y = y_lp + torch.tanh(self.beta) * g_up * y_hp
        return y, v, g_up


# In[ ]:


# 生成分数阶上采样模块：model/modules/fractional_fusion.py
import os, pathlib

repo = '/kaggle/working/UniMatch'
dst  = os.path.join(repo, 'model', 'modules')
os.makedirs(dst, exist_ok=True)
path = os.path.join(dst, 'fractional_fusion.py')

content = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FractionalFusionUp(nn.Module):
    '''
    分数阶自适应上采样（替代最终 logits 的双线性上采样）
    - 仅放大空间维度，不改变通道数
    - 8方向分数阶核 + 门控高通增强 + 低通残差融合
    返回: y [B,C,H,W], orders [B,8,h,w], gate [B,1,H,W]
    '''
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
"""

pathlib.Path(path).write_text(content, encoding='utf-8')
print(f'✅ 已生成: {path}')

# 简单校验
try:
    from model.modules.fractional_fusion import FractionalFusionUp
    m = FractionalFusionUp(in_channels=21, up_factor=4)
    import torch
    y, _, _ = m(torch.randn(1,21,64,64))
    print('✅ 模块导入与前向通过，输出形状:', y.shape)
except Exception as e:
    print('❌ 校验失败:', e)


# In[ ]:


import os, re, pathlib, sys

repo = '/kaggle/working/UniMatch'
os.chdir(repo)

# 查找 deeplabv3plus.py
candidate = None
for root, _, files in os.walk(os.path.join(repo, 'model')):
    for f in files:
        if f.lower() == 'deeplabv3plus.py' and 'semseg' in root.replace('\\','/'):
            candidate = os.path.join(root, f)
            break
    if candidate: break

assert candidate and os.path.exists(candidate), f'未找到 deeplabv3plus.py，当前 repo: {repo}'
print('目标文件:', candidate)

src = pathlib.Path(candidate).read_text(encoding='utf-8')
orig = src

# 1) 确保有 F 和我们的模块导入
if 'import torch.nn.functional as F' not in src:
    src = src.replace('from torch import nn', 'from torch import nn\nimport torch.nn.functional as F')

if 'from model.modules.fractional_fusion import FractionalFusionUp' not in src:
    # 尝试在 F.import 后面插入
    if 'import torch.nn.functional as F' in src:
        src = src.replace('import torch.nn.functional as F',
                          'import torch.nn.functional as F\nfrom model.modules.fractional_fusion import FractionalFusionUp')
    else:
        # 兜底插到文件顶部
        src = 'from model.modules.fractional_fusion import FractionalFusionUp\n' + src

# 2) 在 class DeepLabV3Plus 内注入 _up 方法（若不存在）
cls_pat = re.compile(r'(class\s+DeepLabV3Plus\s*\(.*?\)\s*:\s*)', re.S)
m = cls_pat.search(src)
assert m, '未找到类 DeepLabV3Plus 定义'

if 'def _up(self,' not in src:
    inject = """
    def _up(self, t, size=None, scale_factor=None, mode="bilinear", align_corners=True):
        \"\"\"安全上采样：仅在通道数等于类别数时启用分数阶，否则回退双线性\"\"\"
        try:
            nclass = getattr(self, "nclass", None)
            if nclass is None and hasattr(self, "classifier"):
                nclass = getattr(self.classifier, "out_channels", None)
            if nclass is None:
                nclass = t.shape[1]
            is_logits = (t.shape[1] == nclass)
            if is_logits:
                # 懒初始化分数阶上采样（根据当前倍率推断 up_factor）
                if not hasattr(self, "frac_up") or self.frac_up is None:
                    if scale_factor is not None:
                        s = int(scale_factor)
                    elif size is not None:
                        s = max(1, int(round((size[0] if isinstance(size, (list, tuple)) else size) / t.shape[-2])))
                    else:
                        s = 4
                    s = max(1, s)
                    self.frac_up = FractionalFusionUp(
                        in_channels=nclass, up_factor=s, kernel_size=3,
                        vmax=1.0, hidden=48, mode="both", beta=1.6,
                        tau=0.5, center_bias=2.5, smooth_residual=0.15
                    )
                y, _, _ = self.frac_up(t)
                if size is not None and tuple(y.shape[-2:]) != tuple(size):
                    y = F.interpolate(y, size=size, mode=mode, align_corners=align_corners)
                return y
        except Exception:
            pass
        return F.interpolate(t, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    """
    # 注入到类定义后
    insert_at = m.end()
    src = src[:insert_at] + '\n' + inject + '\n' + src[insert_at:]

# 3) 全局替换 F.interpolate 为 self._up
src = re.sub(r'\bF\.interpolate\s*\(', 'self._up(', src)

# 写入备份与新文件
if src != orig:
    backup = candidate + '.bak'
    pathlib.Path(backup).write_text(orig, encoding='utf-8')
    pathlib.Path(candidate).write_text(src, encoding='utf-8')
    print('✅ 已打补丁，备份文件:', backup)
else:
    print('ℹ️ 无变化（可能之前已打过补丁）')


# In[ ]:


# 快速写入配置（如已有字段则覆盖）
import yaml, os
cfg_path = 'configs/pascal.yaml'
with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
cfg.update({
    'use_fractional_up': True,
    'frac_up_factor': 4,            # logits 的下采样倍率，DeepLabV3+ 常见为4
    'frac_kernel_size': 3,
    'frac_vmax': 1.0,
    'frac_beta': 1.6,
    'frac_hidden': 48,
    'frac_tau': 0.5,
    'frac_center_bias': 2.5,
    'frac_smooth_residual': 0.15
})
with open(cfg_path, 'w') as f: yaml.dump(cfg, f)
print('✅ 已更新 configs/pascal.yaml')


# In[ ]:


import os, re, pathlib, textwrap

repo = '/kaggle/working/UniMatch'
os.chdir(repo)

# 1) 找到 deeplabv3plus.py
target = None
for root, _, files in os.walk(os.path.join(repo, 'model')):
    for f in files:
        if f.lower() == 'deeplabv3plus.py' and 'semseg' in root.replace('\\','/'):
            target = os.path.join(root, f)
            break
    if target: break

assert target and os.path.exists(target), f'未找到 deeplabv3plus.py；当前 repo: {repo}'
print('目标文件:', target)

# 2) 若有 .bak，回滚到 .bak 再重打补丁（避免上一次打坏缩进）
bak = target + '.bak'
if os.path.exists(bak):
    print('发现备份，先回滚 .bak 再补丁')
    pathlib.Path(target).write_text(pathlib.Path(bak).read_text(encoding='utf-8'), encoding='utf-8')

src = pathlib.Path(target).read_text(encoding='utf-8')

# 3) 统一缩进（制表符 → 4空格），统一行尾
src = src.replace('\r\n', '\n').replace('\r', '\n')
src = src.replace('\t', '    ')

# 4) 确保 import F 与 FractionalFusionUp
if 'import torch.nn.functional as F' not in src:
    # 插在 from torch import nn 之后（若存在），否则顶端插入
    if 'from torch import nn' in src:
        src = src.replace('from torch import nn', 'from torch import nn\nimport torch.nn.functional as F')
    else:
        src = 'import torch.nn.functional as F\n' + src

if 'from model.modules.fractional_fusion import FractionalFusionUp' not in src:
    src = src.replace(
        'import torch.nn.functional as F',
        'import torch.nn.functional as F\nfrom model.modules.fractional_fusion import FractionalFusionUp'
    )

# 5) 找到 DeepLabV3Plus 类体范围
class_pat = re.compile(r'\nclass\s+DeepLabV3Plus\s*\(.*?\)\s*:\s*\n', re.S)
m = class_pat.search(src)
assert m, '未找到类 DeepLabV3Plus 定义'
cls_start = m.end()

# 查找类结束（下一个 class/def 在列首，或文件结尾）
tail_pat = re.compile(r'\n(class\s+|def\s+)', re.S)
m_end = tail_pat.search(src, cls_start)
cls_end = m_end.start()+1 if m_end else len(src)

cls_block = src[cls_start:cls_end]

# 6) 注入 _up（若不存在），用与类体一致的缩进
if 'def _up(' not in cls_block:
    # 估计类体缩进：找第一条非空行的前导空格；默认 4 空格
    lines_after = cls_block.splitlines()
    indent_cls = '    '
    for ln in lines_after:
        if ln.strip():
            indent_cls = ln[:len(ln)-len(ln.lstrip(' '))] or '    '
            break

    up_code = textwrap.dedent("""
    def _up(self, t, size=None, scale_factor=None, mode="bilinear", align_corners=False):
        \"\"\"仅当通道数等于类别数时启用分数阶上采样；否则回退双线性\"\"\"
        try:
            nclass = getattr(self, "nclass", None)
            if nclass is None and hasattr(self, "classifier"):
                nclass = getattr(self.classifier, "out_channels", None)
            if nclass is None:
                nclass = t.shape[1]
            is_logits = (t.shape[1] == nclass)

            if is_logits:
                # 懒初始化分数阶上采样
                if not hasattr(self, "frac_up") or self.frac_up is None:
                    if scale_factor is not None:
                        s = int(scale_factor)
                    elif size is not None:
                        tgt_h = size[0] if isinstance(size, (list, tuple)) else size
                        s = max(1, int(round(tgt_h / t.shape[-2])))
                    else:
                        s = 4
                    s = max(1, s)
                    self.frac_up = FractionalFusionUp(
                        in_channels=nclass, up_factor=s, kernel_size=3,
                        vmax=1.0, hidden=48, mode="both", beta=1.6,
                        tau=0.5, center_bias=2.5, smooth_residual=0.15
                    )
                y, _, _ = self.frac_up(t)
                if size is not None and tuple(y.shape[-2:]) != tuple(size):
                    y = F.interpolate(y, size=size, mode=mode, align_corners=align_corners)
                return y
        except Exception:
            # 任意异常都回退双线性，保证训练不被打断
            pass
        return F.interpolate(t, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    """).strip('\n')

    # 缩进到类体
    up_code_indented = '\n'.join(
        (indent_cls + ln) if ln.strip() else ln
        for ln in up_code.splitlines()
    )
    # 插到类体开头
    cls_block = up_code_indented + '\n\n' + cls_block

# 7) 仅在类体内将 F.interpolate( 替换为 self._up(
cls_block = re.sub(r'\bF\.interpolate\s*\(', 'self._up(', cls_block)

# 8) 回写文件（备份一次）
orig = pathlib.Path(target).read_text(encoding='utf-8')
backup = target + '.bak'
pathlib.Path(backup).write_text(orig, encoding='utf-8')
new_src = src[:cls_start] + cls_block + src[cls_start+len(src[cls_start:cls_end]):]
pathlib.Path(target).write_text(new_src, encoding='utf-8')
print('✅ 修复完成，已写入并备份到:', backup)

# 9) 简要提示
if not os.path.exists('model/modules/fractional_fusion.py'):
    print('⚠️ 未检测到 model/modules/fractional_fusion.py，请先创建该模块文件后再 dry-run。')


# In[ ]:


# 修复 _up 方法内被误替换成 self._up(...) 的调用，统一还原为 F.interpolate(...)
import os, re, pathlib

repo = '/kaggle/working/UniMatch'
os.chdir(repo)

# 定位 deeplabv3plus.py
target = None
for root, _, files in os.walk(os.path.join(repo, 'model')):
    for f in files:
        if f.lower() == 'deeplabv3plus.py' and 'semseg' in root.replace('\\','/'):
            target = os.path.join(root, f)
            break
    if target: break
assert target and os.path.exists(target), '未找到 deeplabv3plus.py'

src = pathlib.Path(target).read_text(encoding='utf-8')

# 提取并修复 _up 方法体（把 self._up( 还原成 F.interpolate( ）
m = re.search(r'(def\s+_up\s*\(.*?\):\s*)([\s\S]*?)(?=\n\s{4}def\s+|\n\s*class\s+|\Z)', src)
assert m, '未在文件内找到 _up 方法（请先执行“自修复补丁”注入 _up）'

head, body = m.group(1), m.group(2)
fixed_body, cnt = re.subn(r'\bself\._up\s*\(', 'F.interpolate(', body)
if cnt == 0:
    print('ℹ️ _up 内未发现 self._up 调用，无需修改')
else:
    print(f'✅ 已修复 _up 内 {cnt} 处递归调用')

# 写回文件并备份
backup = target + '.bak2'
pathlib.Path(backup).write_text(src, encoding='utf-8')
new_src = src[:m.start(2)] + fixed_body + src[m.end(2):]
pathlib.Path(target).write_text(new_src, encoding='utf-8')
print('写入完成，备份到:', backup)


# In[ ]:


import os, yaml, torch, importlib
os.chdir('/kaggle/working/UniMatch')

import model.semseg.deeplabv3plus as dlv3p
importlib.reload(dlv3p)  # 重新加载，应用刚刚的修复
from model.semseg.deeplabv3plus import DeepLabV3Plus

with open('configs/pascal.yaml','r') as f:
    cfg = yaml.safe_load(f)

m = DeepLabV3Plus(cfg).eval()
x = torch.randn(1,3,321,321)
with torch.no_grad():
    y = m(x)
print('input:', x.shape, 'output:', y.shape)


# In[ ]:


# Cell 3: 修正版 - 确保在正确目录下使用 torchrun

import yaml
import os

# --- 确保我们在正确的工作目录 ---
project_dir = '/kaggle/working/UniMatch'
os.chdir(project_dir)
print(f"当前工作目录: {os.getcwd()}")

# 读取已经更新的配置（包含增强后的数据路径）
config_file_path = 'configs/pascal.yaml'
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

print(f"使用增强后的数据集: {config_data['data_root']}")


# --- 第 1 步: 复现 train.sh 中的变量设置 ---
dataset = 'pascal'
method = 'unimatch'
exp = 'r101'
split = '732'
num_gpus = 1

# --- 第 2 步: 定义和修正所有路径 ---
config_file_path = f'configs/{dataset}.yaml'

labeled_id_path = f'splits/{dataset}/{split}/labeled.txt'  # 使用相对路径
unlabeled_id_path = f'splits/{dataset}/{split}/unlabeled.txt'  # 使用相对路径
save_path = f'/kaggle/working/exp/{dataset}/{method}/{exp}/{split}'
os.makedirs(save_path, exist_ok=True)

port = 12345

# --- 第 3 步: 动态修改配置文件中的 data_root ---
print(f"准备修改配置文件: {config_file_path}")
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)

print(f"原 data_root: {config_data.get('data_root', '未找到')}")
print(f"新 data_root: {config_data['data_root']}")
config_data['data_root'] = '/kaggle/input/pascal1-3/VOC2012'
with open(config_file_path, 'w') as f:
    yaml.dump(config_data, f)
print("配置文件修改成功！")

# --- 第 4 步: 验证关键文件存在 ---
print("\n验证关键文件:")
files_to_check = [
    f'{method}.py',
    config_file_path,
    labeled_id_path,
    unlabeled_id_path
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - 未找到!")

# --- 第 5 步: 使用 torchrun 启动双GPU训练 ---
print(f"\n将使用 {num_gpus} 个 GPU 进行分布式训练 (使用 torchrun)...")

# 使用相对路径调用 unimatch.py，因为我们已经在项目目录中
get_ipython().system('torchrun      --nproc_per_node={num_gpus}      --master_port={port}      {method}.py      --config {config_file_path}      --labeled-id-path {labeled_id_path}      --unlabeled-id-path {unlabeled_id_path}      --save-path {save_path}      --port {port}')

print("\n分布式训练命令已执行。")


# In[ ]:


# 修复 Cell: 训练结果可视化（PyTorch 2.6兼容 + 数据路径修复）

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import random
from torch.utils.data import DataLoader
import sys

# 确保在正确目录
os.chdir('/kaggle/working/UniMatch')

# 导入必要的模块
sys.path.append('/kaggle/working/UniMatch')

def load_trained_model():
    """加载训练好的模型"""

    # 读取配置
    config_file_path = 'configs/pascal.yaml'
    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"配置文件加载成功")
    print(f"Backbone: {cfg.get('backbone', 'unknown')}")
    print(f"Model: {cfg.get('model', 'unknown')}")
    print(f"Data root: {cfg.get('data_root', 'unknown')}")

    # 查找最新的模型checkpoint
    save_path = '/kaggle/working/exp/pascal/unimatch/r101/732'

    # 查找最新的.pth文件
    pth_files = []
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            if file.endswith('.pth'):
                pth_files.append(os.path.join(save_path, file))

    if not pth_files:
        print("❌ 未找到训练好的模型文件！")
        return None, None

    # 选择最新的模型文件
    latest_model = max(pth_files, key=os.path.getctime)
    print(f"加载模型: {latest_model}")

    # 修复PyTorch 2.6兼容性问题
    try:
        # 首先尝试使用 weights_only=False
        checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
        print("✅ 使用 weights_only=False 成功加载")
    except Exception as e:
        print(f"❌ weights_only=False 失败: {e}")
        try:
            # 备用方案：使用 pickle 模块直接加载
            import pickle
            with open(latest_model, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✅ 使用 pickle 成功加载")
        except Exception as e2:
            print(f"❌ 所有加载方式都失败: {e2}")
            return None, None

    # 检查checkpoint的结构
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("找到 'model' 键")
    else:
        state_dict = checkpoint
        print("直接使用checkpoint作为state_dict")

    # 动态创建模型
    model = None

    try:
        from model.semseg.deeplabv3plus import DeepLabV3Plus
        model = DeepLabV3Plus(cfg)
        print("✅ 使用 DeepLabV3Plus 创建模型")
    except Exception as e:
        print(f"❌ DeepLabV3Plus 创建失败: {e}")
        return None, None

    # 尝试加载权重
    try:
        # 检查是否是分布式训练保存的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("检测到分布式训练模型，移除 'module.' 前缀")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 移除 'module.' 前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✅ 模型权重加载完成")

        if missing_keys:
            print(f"⚠️  缺少 {len(missing_keys)} 个键")
        if unexpected_keys:
            print(f"⚠️  多余 {len(unexpected_keys)} 个键")

        model.eval()
        if torch.cuda.is_available():
            model.cuda()
            print("✅ 模型已移至GPU")

        return model, cfg

    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return None, None

def create_simple_test_dataset(cfg):
    """创建简单的测试数据集"""

    # 检查数据根目录
    data_root = cfg['data_root']
    print(f"数据根目录: {data_root}")

    # 尝试不同的路径组合
    possible_img_dirs = [
        os.path.join(data_root, 'JPEGImages'),
        os.path.join(data_root, 'images'),
        os.path.join(data_root, 'img'),
    ]

    possible_mask_dirs = [
        os.path.join(data_root, 'SegmentationClass'),
        os.path.join(data_root, 'annotations'),
        os.path.join(data_root, 'masks'),
    ]

    val_img_dir = None
    val_mask_dir = None

    # 查找图片目录
    for img_dir in possible_img_dirs:
        if os.path.exists(img_dir):
            val_img_dir = img_dir
            print(f"✅ 找到图片目录: {val_img_dir}")
            break

    # 查找标签目录  
    for mask_dir in possible_mask_dirs:
        if os.path.exists(mask_dir):
            val_mask_dir = mask_dir
            print(f"✅ 找到标签目录: {val_mask_dir}")
            break

    if not val_img_dir:
        print(f"❌ 未找到图片目录，尝试的路径:")
        for dir_path in possible_img_dirs:
            print(f"   - {dir_path}")
        return []

    if not val_mask_dir:
        print(f"❌ 未找到标签目录，尝试的路径:")
        for dir_path in possible_mask_dirs:
            print(f"   - {dir_path}")
        return []

    # 获取图片列表
    try:
        all_img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"找到 {len(all_img_files)} 张图片")

        if len(all_img_files) == 0:
            print("❌ 图片目录为空")
            return []

        # 随机选择几张图片
        random.seed(42)
        selected_files = random.sample(all_img_files, min(5, len(all_img_files)))

        test_data = []
        for img_file in selected_files:
            img_path = os.path.join(val_img_dir, img_file)

            # 尝试不同的标签文件扩展名
            base_name = os.path.splitext(img_file)[0]
            possible_mask_files = [
                base_name + '.png',
                base_name + '.jpg', 
                img_file.replace('.jpg', '.png'),
                img_file.replace('.jpeg', '.png')
            ]

            mask_path = None
            for mask_file in possible_mask_files:
                potential_path = os.path.join(val_mask_dir, mask_file)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break

            if mask_path and os.path.exists(img_path):
                test_data.append((img_path, mask_path))
                print(f"✅ 添加测试样本: {img_file}")
            else:
                print(f"⚠️  跳过样本（缺少标签）: {img_file}")

        print(f"最终测试数据集大小: {len(test_data)}")
        return test_data

    except Exception as e:
        print(f"❌ 创建测试数据集失败: {e}")
        return []

def simple_preprocess(img_path, target_size=321):
    """简单的图像预处理"""

    try:
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # 转换为tensor并归一化
        img_array = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

        # ImageNet归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor.unsqueeze(0)  # 添加batch维度

    except Exception as e:
        print(f"❌ 预处理图片失败 {img_path}: {e}")
        return None

def visualize_simple_predictions(model, cfg, num_samples=3):
    """简化的可视化预测结果"""

    # 创建测试数据
    test_data = create_simple_test_dataset(cfg)

    if not test_data:
        print("❌ 未找到测试数据，无法进行可视化")
        return

    # PASCAL VOC 类别名称
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    # 设置图片布局
    num_samples = min(num_samples, len(test_data))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"开始预测 {num_samples} 张图片...")

    successful_predictions = 0

    with torch.no_grad():
        for i, (img_path, mask_path) in enumerate(test_data[:num_samples]):
            try:
                print(f"\n处理图片 {i+1}: {os.path.basename(img_path)}")

                # 预处理图像
                img_tensor = simple_preprocess(img_path, cfg['crop_size'])
                if img_tensor is None:
                    continue

                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()

                # 模型预测
                try:
                    pred = model(img_tensor)
                    print(f"  预测输出形状: {pred.shape}")
                    pred = torch.argmax(pred, dim=1)
                except Exception as pred_error:
                    print(f"  ❌ 模型预测失败: {pred_error}")
                    continue

                # 加载原始图像和标签
                original_img = Image.open(img_path).convert('RGB')
                original_img = original_img.resize((cfg['crop_size'], cfg['crop_size']), Image.Resampling.LANCZOS)

                # 尝试加载标签图像
                try:
                    mask_img = Image.open(mask_path)
                    if mask_img.mode != 'L':  # 如果不是灰度图，转换为灰度
                        mask_img = mask_img.convert('L')
                    mask_img = mask_img.resize((cfg['crop_size'], cfg['crop_size']), Image.Resampling.NEAREST)
                    mask_array = np.array(mask_img)
                except Exception as mask_error:
                    print(f"  ❌ 加载标签失败: {mask_error}")
                    # 创建虚拟标签
                    mask_array = np.zeros((cfg['crop_size'], cfg['crop_size']))

                # 转换预测结果
                pred_array = pred[0].cpu().numpy()

                print(f"  预测数组形状: {pred_array.shape}")
                print(f"  标签数组形状: {mask_array.shape}")
                print(f"  预测类别范围: {pred_array.min()} - {pred_array.max()}")
                print(f"  标签类别范围: {mask_array.min()} - {mask_array.max()}")

                # 创建彩色分割图
                def colorize_mask(mask, colors):
                    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                    unique_classes = np.unique(mask)
                    for class_id in unique_classes:
                        if class_id < len(colors):
                            color_mask[mask == class_id] = colors[class_id][:3]
                    return color_mask

                colored_gt = colorize_mask(mask_array, colors)
                colored_pred = colorize_mask(pred_array, colors)

                # 绘制结果
                axes[i, 0].imshow(original_img)
                axes[i, 0].set_title(f'Original Image {i+1}')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(colored_gt)
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(colored_pred)
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')

                successful_predictions += 1
                print(f"✅ 图片 {i+1} 预测完成")

            except Exception as e:
                print(f"❌ 图片 {i+1} 预测失败: {e}")
                # 显示错误信息在图片上
                if i < len(axes):
                    for j in range(3):
                        axes[i, j].text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                                       ha='center', va='center', transform=axes[i, j].transAxes,
                                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                        axes[i, j].axis('off')
                continue

    plt.tight_layout()
    plt.show()

    print(f"\n=== 预测结果总结 ===")
    print(f"成功预测: {successful_predictions}/{num_samples}")

    if successful_predictions > 0:
        # 显示类别颜色映射
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, (name, color) in enumerate(zip(class_names, colors)):
            ax.barh(i, 1, color=color[:3])
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Class Color Mapping')
        ax.set_title('PASCAL VOC Class Colors')
        plt.tight_layout()
        plt.show()
    else:
        print("❌ 所有预测都失败了，请检查模型和数据")

# 主执行函数
def main():
    print("=== 开始可视化训练结果 ===")

    # 加载模型
    model, cfg = load_trained_model()

    if model is None:
        print("❌ 无法加载模型")
        return

    # 可视化预测结果
    print("\n开始可视化预测结果...")
    try:
        visualize_simple_predictions(model, cfg, num_samples=3)
        print("\n✅ 可视化完成")
    except Exception as e:
        print(f"\n❌ 可视化过程出错: {e}")

# 运行可视化
main()


# In[ ]:


# Cell: 备用方案 - 手动解析和可视化训练数据
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def manual_parse_and_visualize():
    """手动解析事件文件并可视化"""

    log_dir = Path(r"/kaggle/input/origin-event")
    events_files = list(log_dir.glob("events.out.tfevents.*"))

    if not events_files:
        print("❌ 未找到事件文件")
        return

    print(f"📊 手动解析 {len(events_files)} 个事件文件...")

    all_data = {}

    for events_file in events_files:
        print(f"\n解析文件: {events_file.name}")

        try:
            # 尝试使用 TensorFlow 解析
            import tensorflow as tf

            for event in tf.compat.v1.train.summary_iterator(str(events_file)):
                if event.summary:
                    for value in event.summary.value:
                        tag = value.tag
                        scalar_value = value.simple_value
                        step = event.step

                        if tag not in all_data:
                            all_data[tag] = {'steps': [], 'values': []}

                        all_data[tag]['steps'].append(step)
                        all_data[tag]['values'].append(scalar_value)

            print(f"✅ 成功解析")

        except ImportError:
            print("❌ TensorFlow 未安装，跳过此文件")
            continue
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            continue

    if not all_data:
        print("❌ 未找到任何可用数据")
        return

    # 可视化数据
    print(f"\n📈 找到 {len(all_data)} 个指标:")
    for tag in all_data.keys():
        print(f"   - {tag}: {len(all_data[tag]['values'])} 个数据点")

    # 创建图表
    num_metrics = len(all_data)
    cols = 2
    rows = (num_metrics + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))

    if num_metrics == 1:
        axes = [axes]
    elif num_metrics <= 2:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()

    for i, (tag, data) in enumerate(all_data.items()):
        if i < len(axes):
            steps = np.array(data['steps'])
            values = np.array(data['values'])

            axes[i].plot(steps, values, 'b-', linewidth=2, marker='o', markersize=4)
            axes[i].set_title(f'{tag}', fontsize=12)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)

            # 显示统计信息
            if len(values) > 0:
                final_value = values[-1]
                max_value = np.max(values)
                min_value = np.min(values)

                info_text = f'Final: {final_value:.4f}\nMax: {max_value:.4f}\nMin: {min_value:.4f}'
                axes[i].text(0.02, 0.98, info_text, 
                           transform=axes[i].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                           verticalalignment='top',
                           fontsize=8)

    # 隐藏多余的子图
    for i in range(len(all_data), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    return all_data

# 运行手动解析
training_data = manual_parse_and_visualize()

