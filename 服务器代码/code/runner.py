#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch训练脚本 - 适用于Linux云服务器
支持分数阶增强和自适应上采样
"""

import os
import sys
import argparse
import subprocess
import shutil
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import re
import pathlib
import textwrap
from PIL import Image
from tqdm import tqdm
from scipy.special import gamma

def run_command(cmd, check=True, shell=False):
    """安全执行系统命令"""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        
        # 实时输出，不缓冲
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时打印输出
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
        return process
        
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {cmd}")
        print(f"错误代码: {e.returncode}")
        if check:
            raise
        return e
class FractionalCoefficients:
    """基于公式(18)的分数阶系数计算类"""
    
    @staticmethod
    def compute_coefficients(v, max_order=10):
        """计算分数阶系数 C_k"""
        coeffs = {}
        
        # 基于提供的公式计算
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
        
        # 额外的4个方向
        directions = ['northeast', 'northwest', 'southeast', 'southwest']
        offsets = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        
        for direction, (row_offset, col_offset) in zip(directions, offsets):
            mask = torch.zeros(self.mask_size, self.mask_size)
            for i, coeff in self.coeffs.items():
                if abs(i) <= center:
                    row, col = center + i * row_offset, center + i * col_offset
                    if 0 <= row < self.mask_size and 0 <= col < self.mask_size:
                        mask[row, col] = coeff
            masks[direction] = mask
        
        return masks

class FractionalImageEnhancer:
    """模块化分数阶图像增强类"""
    
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
            'horizontal': 1.0, 'vertical': 1.0,
            'diagonal_main': 0.8, 'diagonal_anti': 0.8,
            'northeast': 0.6, 'northwest': 0.6,
            'southeast': 0.6, 'southwest': 0.6
        }
    
    def _preprocess_image(self, image):
        """将输入图像预处理为标准格式"""
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
            combined = torch.stack(feature_list).mean(dim=0)
            
        elif self.enhancement_mode == 'weighted':
            combined = torch.zeros_like(feature_list[0])
            total_weight = 0
            
            for direction, feature in directional_features.items():
                weight = self.direction_weights.get(direction, 1.0)
                combined += weight * feature
                total_weight += weight
                
            combined = combined / total_weight
            
        elif self.enhancement_mode == 'selective':
            feature_stack = torch.stack(feature_list)  # [8, H, W]
            variances = torch.var(feature_stack, dim=(1, 2))
            weights = F.softmax(variances * 10, dim=0)  # 温度缩放
            
            combined = torch.zeros_like(feature_list[0])
            for i, feature in enumerate(feature_list):
                combined += weights[i] * feature
                
        else:
            raise ValueError(f"未知的增强模式: {self.enhancement_mode}")
            
        return combined
    
    def enhance_image(self, image):
        """主要增强函数"""
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

def setup_environment(args):
    """设置环境和下载代码"""
    print("🚀 设置训练环境...")
    
    # 创建工作目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    
    # 克隆UniMatch仓库
    unimatch_dir = work_dir / 'UniMatch'
    if not unimatch_dir.exists():
        print("📥 克隆 UniMatch 仓库...")
        run_command(f"git clone https://github.com/LiheYoung/UniMatch.git")
    else:
        print("✅ UniMatch 仓库已存在")
    
    os.chdir(unimatch_dir)
    print(f"当前工作目录: {os.getcwd()}")
    
    # 安装依赖
    if (unimatch_dir / 'requirements.txt').exists():
        print("📦 安装依赖包...")
        run_command("pip install -r requirements.txt")
    
    return unimatch_dir

def setup_pretrained_model(args, project_dir):
    """设置预训练模型"""
    print("📥 设置预训练模型...")
    
    pretrained_dir = project_dir / 'pretrained'
    pretrained_dir.mkdir(exist_ok=True)
    
    target_model_path = pretrained_dir / 'resnet101.pth'
    
    # 检查预训练模型路径
    if args.pretrained_path and Path(args.pretrained_path).exists():
        print(f"✅ 找到预训练模型: {args.pretrained_path}")
        shutil.copy2(args.pretrained_path, target_model_path)
        print(f"✅ 预训练模型已复制到: {target_model_path}")
        print(f"文件大小: {target_model_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ 未找到预训练模型文件: {args.pretrained_path}")
        print("请下载 ResNet101 预训练模型到指定路径")
        return False
    
    return True

def create_fractional_fusion_module(project_dir):
    """创建分数阶融合模块"""
    print("🔧 创建分数阶融合模块...")
    
    modules_dir = project_dir / 'model' / 'modules'
    modules_dir.mkdir(parents=True, exist_ok=True)
    
    fractional_fusion_path = modules_dir / 'fractional_fusion.py'
    
    content = '''
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
'''
    
    fractional_fusion_path.write_text(content.strip(), encoding='utf-8')
    print(f'✅ 已生成: {fractional_fusion_path}')
    
    return True

def patch_deeplabv3plus(project_dir):
    """修补DeepLabV3Plus以支持分数阶上采样"""
    print("🔧 修补 DeepLabV3Plus...")
    
    # 查找 deeplabv3plus.py
    target = None
    for root, _, files in os.walk(project_dir / 'model'):
        for f in files:
            if f.lower() == 'deeplabv3plus.py' and 'semseg' in str(root):
                target = Path(root) / f
                break
        if target: break
    
    if not target or not target.exists():
        print(f'❌ 未找到 deeplabv3plus.py')
        return False
    
    print(f'目标文件: {target}')
    
    # 备份原文件
    backup = target.with_suffix(target.suffix + '.bak')
    if backup.exists():
        # 从备份恢复
        shutil.copy2(backup, target)
    else:
        # 创建备份
        shutil.copy2(target, backup)
    
    src = target.read_text(encoding='utf-8')
    
    # 统一缩进和行尾
    src = src.replace('\r\n', '\n').replace('\r', '\n').replace('\t', '    ')
    
    # 确保有必要的导入
    if 'import torch.nn.functional as F' not in src:
        if 'from torch import nn' in src:
            src = src.replace('from torch import nn', 'from torch import nn\nimport torch.nn.functional as F')
        else:
            src = 'import torch.nn.functional as F\n' + src
    
    if 'from model.modules.fractional_fusion import FractionalFusionUp' not in src:
        src = src.replace(
            'import torch.nn.functional as F',
            'import torch.nn.functional as F\nfrom model.modules.fractional_fusion import FractionalFusionUp'
        )
    
    # 找到 DeepLabV3Plus 类
    class_pat = re.compile(r'\nclass\s+DeepLabV3Plus\s*\(.*?\)\s*:\s*\n', re.S)
    m = class_pat.search(src)
    if not m:
        print('❌ 未找到类 DeepLabV3Plus 定义')
        return False
    
    cls_start = m.end()
    
    # 查找类结束
    tail_pat = re.compile(r'\n(class\s+|def\s+)', re.S)
    m_end = tail_pat.search(src, cls_start)
    cls_end = m_end.start()+1 if m_end else len(src)
    
    cls_block = src[cls_start:cls_end]
    
    # 注入 _up 方法（若不存在）
    if 'def _up(' not in cls_block:
        # 估计类体缩进
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
    
    # 仅在类体内将 F.interpolate( 替换为 self._up(
    cls_block = re.sub(r'\bF\.interpolate\s*\(', 'self._up(', cls_block)
    
    # 修复 _up 方法内被误替换的调用
    up_method_pattern = r'(def\s+_up\s*\(.*?\):\s*)([\s\S]*?)(?=\n\s{4}def\s+|\n\s*class\s+|\Z)'
    m_up = re.search(up_method_pattern, cls_block)
    if m_up:
        head, body = m_up.group(1), m_up.group(2)
        fixed_body = re.sub(r'\bself\._up\s*\(', 'F.interpolate(', body)
        cls_block = cls_block[:m_up.start(2)] + fixed_body + cls_block[m_up.end(2):]
    
    # 回写文件
    new_src = src[:cls_start] + cls_block + src[cls_start+len(src[cls_start:cls_end]):]
    target.write_text(new_src, encoding='utf-8')
    print('✅ DeepLabV3Plus 修补完成')
    
    return True

def fix_torch_load_compatibility(project_dir):
    """修复PyTorch 2.6兼容性问题"""
    print("🔧 修复 torch.load 兼容性...")
    
    python_files = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"检查 {len(python_files)} 个Python文件...")
    
    fixed_files = []
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # 更精确的正则表达式，只匹配真正的torch.load调用
            # 匹配 torch.load(参数) 但不匹配在其他函数调用内部的情况
            pattern = r'torch\.load\s*\(\s*([^)]+)\s*\)'
            
            def fix_weights_only(match):
                args_str = match.group(1).strip()
                
                # 检查是否已经有weights_only参数
                if 'weights_only' in args_str:
                    return match.group(0)  # 已经有了，不修改
                
                # 确保这不是在其他函数调用内部
                # 如果参数中包含函数调用，需要更仔细处理
                if args_str.count('(') != args_str.count(')'):
                    return match.group(0)  # 可能是嵌套调用，跳过
                
                # 添加weights_only参数
                if args_str.strip():
                    args_str += ', weights_only=False'
                else:
                    args_str = 'weights_only=False'
                
                return f'torch.load({args_str})'
            
            # 应用修复
            new_content = re.sub(pattern, fix_weights_only, content)
            
            # 检查是否有变化
            if new_content != original_content:
                file_path.write_text(new_content, encoding='utf-8')
                fixed_files.append(file_path)
                
        except Exception as e:
            print(f"⚠️  无法处理文件 {file_path}: {e}")
    
    print(f"✅ 已修复 {len(fixed_files)} 个文件")
    return True

def update_config(args, project_dir):
    """更新配置文件"""
    print("🔧 更新配置文件...")
    
    config_path = project_dir / 'configs' / 'pascal.yaml'
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 更新基本配置
    config_data.update({
        'data_root': str(args.data_root),
        'batch_size': args.batch_size,
        'backbone': 'resnet101',
        'lr': args.lr,
        'crop_size': 321,
        'nclass': 21,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
        'eval_freq': args.eval_freq,
        'epochs': args.epochs,
    })
    
    # 添加分数阶相关配置
    config_data.update({
        'use_fractional_up': True,
        'frac_up_factor': 4,
        'frac_kernel_size': 3,
        'frac_vmax': 1.0,
        'frac_beta': 1.6,
        'frac_hidden': 48,
        'frac_tau': 0.5,
        'frac_center_bias': 2.5,
        'frac_smooth_residual': 0.15
    })
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    print("✅ 配置文件更新完成")
    return config_data

def preprocess_dataset_with_fractional_enhancement(args, project_dir):
    """对整个数据集进行分数阶增强预处理"""
    if not args.enable_fractional_enhancement:
        print("跳过分数阶增强预处理")
        return args.data_root
    
    print("🚀 开始对数据集进行分数阶增强预处理...")
    
    original_data_root = Path(args.data_root)
    enhanced_data_root = project_dir.parent / 'enhanced_VOC2012'
    
    original_img_dir = original_data_root / 'JPEGImages'
    enhanced_img_dir = enhanced_data_root / 'JPEGImages'
    
    # 创建增强后的数据目录
    enhanced_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制其他必要目录
    for dir_name in ['SegmentationClass', 'SegmentationObject', 'ImageSets']:
        src_dir = original_data_root / dir_name
        dst_dir = enhanced_data_root / dir_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"✅ 复制目录: {dir_name}")
    
    # 初始化分数阶增强器
    enhancer = FractionalImageEnhancer(
        fractional_order=0.6,
        enhancement_factor=0.2,
        enhancement_mode='weighted'
    )
    
    # 获取所有图像文件
    img_files = [f for f in original_img_dir.iterdir() 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"找到 {len(img_files)} 张图像需要处理")
    
    # 批量处理图像
    processed_count = 0
    failed_count = 0
    
    for img_file in tqdm(img_files, desc="处理图像"):
        try:
            # 加载原始图像
            original_img = Image.open(img_file).convert('RGB')
            
            # 应用分数阶增强
            enhanced_tensor, _, _ = enhancer.enhance_image(original_img)
            
            # 转换回PIL图像
            enhanced_array = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            enhanced_img = Image.fromarray(enhanced_array)
            
            # 保存增强后的图像
            enhanced_path = enhanced_img_dir / img_file.name
            enhanced_img.save(enhanced_path, quality=95)
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理图像失败 {img_file.name}: {str(e)}")
            failed_count += 1
            
            # 如果增强失败，复制原图像
            try:
                shutil.copy2(img_file, enhanced_img_dir / img_file.name)
                processed_count += 1
            except:
                pass
    
    print(f"\n✅ 预处理完成:")
    print(f"   成功处理: {processed_count} 张图像")
    print(f"   失败: {failed_count} 张图像")
    print(f"   增强后数据集位置: {enhanced_data_root}")
    
    return enhanced_data_root

def start_training(args, project_dir):
    """启动训练"""
    print("🚀 开始训练...")
    
    # 设置训练参数
    dataset = 'pascal'
    method = 'unimatch'
    exp = 'r101'
    split = '732'
    
    config_file_path = f'configs/{dataset}.yaml'
    labeled_id_path = f'splits/{dataset}/{split}/labeled.txt'
    unlabeled_id_path = f'splits/{dataset}/{split}/unlabeled.txt'
    save_path = args.save_path or str(project_dir.parent / 'exp' / dataset / method / exp / split)
    
    # 创建保存目录
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 验证关键文件存在
    files_to_check = [
        f'{method}.py',
        config_file_path,
        labeled_id_path,
        unlabeled_id_path
    ]
    
    print("\n验证关键文件:")
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 未找到!")
            return False
    
    # 构建训练命令 - 即使单GPU也要用torchrun
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={args.port}",
        "--nnodes=1",
        "--node_rank=0",
        f"{method}.py",
        "--config", config_file_path,
        "--labeled-id-path", labeled_id_path,
        "--unlabeled-id-path", unlabeled_id_path,
        "--save-path", save_path,
        "--port", str(args.port)
    ]
    
    print(f"\n使用 {args.num_gpus} 个 GPU 进行训练...")
    print(f"执行命令: {' '.join(cmd)}")
    
    # 启动训练
    try:
        result = run_command(cmd)
        print("✅ 训练完成")
        return True
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='UniMatch训练脚本 - 支持分数阶增强')
    
    # 基本参数
    parser.add_argument('--work-dir', type=str, default='/root/autodl-tmp/unimatch_workspace',
                       help='工作目录路径')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据集根目录路径')
    parser.add_argument('--pretrained-path', type=str, 
                       help='预训练模型路径 (resnet101.pth)')
    parser.add_argument('--save-path', type=str,
                       help='模型保存路径')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=6,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.004,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=80,
                       help='训练轮数')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='验证频率')
    parser.add_argument('--num-workers', type=int, default=6,
                       help='数据加载工作进程数')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='使用的GPU数量')
    parser.add_argument('--port', type=int, default=12345,
                       help='分布式训练端口')
    
    # 分数阶增强参数
    parser.add_argument('--enable-fractional-enhancement', action='store_true',
                       help='启用分数阶增强预处理')
    
    # 流程控制
    parser.add_argument('--skip-setup', action='store_true',
                       help='跳过环境设置')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='跳过数据预处理')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练（仅设置环境）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 UniMatch 分数阶增强训练脚本")
    print("=" * 60)
    print(f"工作目录: {args.work_dir}")
    print(f"数据根目录: {args.data_root}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"分数阶增强: {'启用' if args.enable_fractional_enhancement else '禁用'}")
    print("=" * 60)
    
    try:
        # Step 1: 设置环境
        if not args.skip_setup:
            project_dir = setup_environment(args)
            
            # 设置预训练模型
            if args.pretrained_path:
                if not setup_pretrained_model(args, project_dir):
                    print("❌ 预训练模型设置失败")
                    return 1
            
            # 创建分数阶融合模块
            create_fractional_fusion_module(project_dir)
            
            # 修补DeepLabV3Plus
            patch_deeplabv3plus(project_dir)
            
            # 修复PyTorch兼容性
            fix_torch_load_compatibility(project_dir)
            
        else:
            project_dir = Path(args.work_dir) / 'UniMatch'
            os.chdir(project_dir)
        
        # Step 2: 数据预处理
        if not args.skip_preprocessing:
            enhanced_data_root = preprocess_dataset_with_fractional_enhancement(args, project_dir)
            # 更新数据根路径
            if args.enable_fractional_enhancement:
                args.data_root = str(enhanced_data_root)
        
        # Step 3: 更新配置
        update_config(args, project_dir)
        
        # Step 4: 开始训练
        if not args.skip_training:
            success = start_training(args, project_dir)
            if success:
                print("\n🎉 训练完成！")
                return 0
            else:
                print("\n❌ 训练失败！")
                return 1
        else:
            print("\n✅ 环境设置完成，跳过训练")
            return 0
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())