#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版训练结果分析脚本
"""

import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def enhanced_log_analysis(result_dir):
    """增强的日志分析"""
    result_dir = Path(result_dir)
    print(f"\n🔍 详细日志分析: {result_dir}")
    
    # 查找所有可能的日志文件
    log_patterns = ["*.log", "*.out", "train*", "*.txt"]
    log_files = []
    for pattern in log_patterns:
        log_files.extend(result_dir.glob(pattern))
    
    if not log_files:
        print("❌ 未找到任何日志文件")
        return
    
    all_metrics = {
        'epochs': [],
        'losses': [],
        'loss_x': [],
        'loss_s': [],
        'mious': [],
        'timestamps': []
    }
    
    for log_file in log_files:
        print(f"\n📖 分析文件: {log_file.name}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 多种日志格式的解析
            parse_unimatch_logs(content, all_metrics)
            
        except Exception as e:
            print(f"⚠️ 读取文件失败: {e}")
    
    # 生成可视化
    if all_metrics['epochs']:
        create_enhanced_plots(all_metrics, result_dir)
        print_detailed_stats(all_metrics)
    else:
        print("❌ 未找到训练指标，尝试手动检查日志格式...")
        manual_log_inspection(result_dir)

def parse_unimatch_logs(content, metrics):
    """解析UniMatch的多种日志格式"""
    
    # 格式1: Iters: X, Total loss: Y.YYY
    pattern1 = r'Iters:\s*(\d+),\s*Total loss:\s*([\d.]+)'
    matches1 = re.findall(pattern1, content)
    for iter_num, loss in matches1:
        metrics['epochs'].append(int(iter_num))
        metrics['losses'].append(float(loss))
    
    # 格式2: Loss x: X.XXX, Loss s: Y.YYY
    pattern2 = r'Loss x:\s*([\d.]+),\s*Loss s:\s*([\d.]+)'
    matches2 = re.findall(pattern2, content)
    for loss_x, loss_s in matches2:
        metrics['loss_x'].append(float(loss_x))
        metrics['loss_s'].append(float(loss_s))
    
    # 格式3: Epoch: X, ... mIoU: Y.YYYY
    pattern3 = r'Epoch:\s*(\d+).*?mIoU:\s*([\d.]+)'
    matches3 = re.findall(pattern3, content, re.DOTALL)
    epoch_miou = {}
    for epoch, miou in matches3:
        epoch_miou[int(epoch)] = float(miou)
    
    # 对齐mIoU到epoch
    for epoch in metrics['epochs']:
        if epoch in epoch_miou:
            metrics['mious'].append(epoch_miou[epoch])
        else:
            metrics['mious'].append(0)
    
    # 格式4: 时间戳格式
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
    timestamps = re.findall(timestamp_pattern, content)
    metrics['timestamps'].extend(timestamps)

def create_enhanced_plots(metrics, result_dir):
    """创建增强的可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('UniMatch Training Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss曲线
    if metrics['losses']:
        axes[0, 0].plot(metrics['epochs'], metrics['losses'], 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(metrics['losses']) > 10:
            z = np.polyfit(metrics['epochs'], metrics['losses'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(metrics['epochs'], p(metrics['epochs']), "r--", alpha=0.8, label='Trend')
            axes[0, 0].legend()
    
    # 2. Loss组件对比
    if metrics['loss_x'] and metrics['loss_s']:
        min_len = min(len(metrics['loss_x']), len(metrics['loss_s']), len(metrics['epochs']))
        x_range = metrics['epochs'][:min_len]
        axes[0, 1].plot(x_range, metrics['loss_x'][:min_len], 'r-', label='Loss X', linewidth=2)
        axes[0, 1].plot(x_range, metrics['loss_s'][:min_len], 'g-', label='Loss S', linewidth=2)
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Iterations')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. mIoU曲线
    if any(m > 0 for m in metrics['mious']):
        valid_indices = [i for i, m in enumerate(metrics['mious']) if m > 0]
        valid_epochs = [metrics['epochs'][i] for i in valid_indices]
        valid_mious = [metrics['mious'][i] for i in valid_indices]
        
        axes[1, 0].plot(valid_epochs, valid_mious, 'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_title('Validation mIoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 标注最佳点
        if valid_mious:
            best_idx = np.argmax(valid_mious)
            best_epoch = valid_epochs[best_idx]
            best_miou = valid_mious[best_idx]
            axes[1, 0].annotate(f'Best: {best_miou:.4f}', 
                              xy=(best_epoch, best_miou), 
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 4. 训练统计
    axes[1, 1].axis('off')
    stats_text = generate_stats_text(metrics, result_dir)
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    save_path = result_dir / 'enhanced_training_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 增强分析图表已保存到: {save_path}")

def generate_stats_text(metrics, result_dir):
    """生成统计文本"""
    stats = []
    stats.append("📊 训练统计信息")
    stats.append("=" * 20)
    
    if metrics['losses']:
        stats.append(f"总迭代次数: {len(metrics['losses'])}")
        stats.append(f"初始Loss: {metrics['losses'][0]:.4f}")
        stats.append(f"最终Loss: {metrics['losses'][-1]:.4f}")
        loss_reduction = (metrics['losses'][0] - metrics['losses'][-1]) / metrics['losses'][0] * 100
        stats.append(f"Loss下降: {loss_reduction:.1f}%")
    
    if any(m > 0 for m in metrics['mious']):
        valid_mious = [m for m in metrics['mious'] if m > 0]
        stats.append(f"最佳mIoU: {max(valid_mious):.4f}")
        stats.append(f"验证次数: {len(valid_mious)}")
    
    # 检查分数阶融合
    best_model = result_dir / 'best.pth'
    if best_model.exists():
        try:
            checkpoint = torch.load(best_model, map_location='cpu', weights_only=False)
            frac_params = [k for k in checkpoint['model'].keys() if 'frac_up' in k]
            if frac_params:
                stats.append("✅ 分数阶融合: 已启用")
                stats.append(f"   参数数量: {len(frac_params)}")
            else:
                stats.append("❌ 分数阶融合: 未启用")
        except:
            pass
    
    return '\n'.join(stats)

def print_detailed_stats(metrics):
    """打印详细统计"""
    print("\n📈 详细训练统计:")
    
    if metrics['losses']:
        print(f"   训练迭代: {len(metrics['losses'])} 次")
        print(f"   Loss范围: {min(metrics['losses']):.4f} - {max(metrics['losses']):.4f}")
        
        # Loss趋势
        if len(metrics['losses']) > 1:
            recent_avg = np.mean(metrics['losses'][-10:])
            early_avg = np.mean(metrics['losses'][:10])
            print(f"   Loss趋势: {early_avg:.4f} → {recent_avg:.4f}")
    
    if any(m > 0 for m in metrics['mious']):
        valid_mious = [m for m in metrics['mious'] if m > 0]
        print(f"   验证次数: {len(valid_mious)}")
        print(f"   mIoU范围: {min(valid_mious):.4f} - {max(valid_mious):.4f}")

def manual_log_inspection(result_dir):
    """手动检查日志格式"""
    print("\n🔍 手动日志检查:")
    
    for log_file in result_dir.glob("*"):
        if log_file.is_file() and log_file.suffix in ['.log', '.out', '.txt']:
            print(f"\n📄 {log_file.name} 内容示例:")
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # 显示前10行和后10行
                    print("前10行:")
                    for i, line in enumerate(lines[:10], 1):
                        print(f"{i:2d}: {line.strip()}")
                    
                    if len(lines) > 20:
                        print("...")
                        print("后10行:")
                        for i, line in enumerate(lines[-10:], len(lines)-9):
                            print(f"{i:2d}: {line.strip()}")
            except Exception as e:
                print(f"读取失败: {e}")

def main():
    """主函数"""
    result_dir = "/root/autodl-tmp/unimatch_workspace/exp/pascal/unimatch/r101/732"
    
    print("🎯 UniMatch 增强训练结果分析")
    print("=" * 50)
    
    enhanced_log_analysis(result_dir)
    
    print("\n✅ 增强分析完成！")

if __name__ == "__main__":
    main()