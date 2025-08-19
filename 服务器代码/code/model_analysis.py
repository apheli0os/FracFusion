#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从模型文件分析训练结果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import os

def direct_model_analysis():
    """直接从模型文件分析训练结果"""
    result_dir = Path("/root/autodl-tmp/unimatch_workspace/exp/pascal/unimatch/r101/732")
    
    print("🔍 直接模型文件分析")
    print("=" * 50)
    
    # 1. 分析模型文件
    analyze_model_checkpoints(result_dir)
    
    # 2. 查找所有文件
    find_all_files(result_dir)
    
    # 3. 检查上级目录的日志
    check_parent_logs()
    
    # 4. 生成基于模型的可视化
    create_model_based_visualization(result_dir)

def analyze_model_checkpoints(result_dir):
    """分析模型checkpoint"""
    print("\n📊 模型文件详细分析:")
    
    model_files = ['best.pth', 'latest.pth']
    
    for filename in model_files:
        filepath = result_dir / filename
        if not filepath.exists():
            print(f"❌ {filename} 不存在")
            continue
            
        try:
            print(f"\n🔍 分析 {filename}:")
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # 基本信息
            print(f"   文件大小: {filepath.stat().st_size / (1024*1024):.2f} MB")
            
            # 检查所有键
            print(f"   Checkpoint包含的键: {list(checkpoint.keys())}")
            
            # 详细分析每个键
            for key, value in checkpoint.items():
                if key == 'model':
                    model_params = list(value.keys())
                    print(f"   模型参数数量: {len(model_params)}")
                    
                    # 分数阶融合参数
                    frac_params = [k for k in model_params if 'frac' in k.lower()]
                    if frac_params:
                        print(f"   ✅ 分数阶融合参数 ({len(frac_params)}个):")
                        for param in frac_params[:5]:  # 显示前5个
                            print(f"      - {param}")
                        if len(frac_params) > 5:
                            print(f"      - ... 还有 {len(frac_params)-5} 个")
                    else:
                        print(f"   ❌ 未找到分数阶融合参数")
                    
                    # backbone参数
                    backbone_params = [k for k in model_params if 'backbone' in k]
                    print(f"   backbone参数: {len(backbone_params)} 个")
                    
                    # 分类器参数
                    classifier_params = [k for k in model_params if 'classifier' in k]
                    print(f"   分类器参数: {len(classifier_params)} 个")
                    
                elif key == 'epoch':
                    print(f"   训练轮数: {value}")
                elif key == 'best_miou':
                    if isinstance(value, (int, float)):
                        print(f"   最佳mIoU: {value:.4f}")
                    else:
                        print(f"   最佳mIoU: {value}")
                elif key == 'optimizer':
                    print(f"   优化器状态: 已保存")
                else:
                    print(f"   {key}: {type(value).__name__}")
                    
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")

def find_all_files(result_dir):
    """查找所有相关文件"""
    print(f"\n📁 目录文件列表 ({result_dir}):")
    
    if not result_dir.exists():
        print(f"❌ 目录不存在: {result_dir}")
        return
    
    all_files = []
    for item in result_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            all_files.append((item.name, size_mb, item.suffix))
    
    # 按大小排序
    all_files.sort(key=lambda x: x[1], reverse=True)
    
    print("   文件名                          大小(MB)   类型")
    print("   " + "-" * 50)
    for name, size, ext in all_files:
        print(f"   {name:<30} {size:>8.2f}   {ext}")

def check_parent_logs():
    """检查上级目录的日志文件"""
    print(f"\n🔍 搜索训练日志:")
    
    # 搜索可能的日志位置
    search_paths = [
        "/root/autodl-tmp/",
        "/root/autodl-tmp/unimatch_workspace/",
        "/root/autodl-tmp/unimatch_workspace/UniMatch/",
    ]
    
    log_patterns = ["*.log", "*.out", "train*", "nohup*", "*.txt"]
    
    found_logs = []
    for search_path in search_paths:
        search_dir = Path(search_path)
        if search_dir.exists():
            for pattern in log_patterns:
                found_logs.extend(search_dir.glob(pattern))
    
    if found_logs:
        print("   找到的日志文件:")
        for log_file in found_logs:
            size_mb = log_file.stat().st_size / (1024*1024)
            print(f"   📄 {log_file} ({size_mb:.2f} MB)")
            
            # 显示文件内容示例
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"      首行: {lines[0].strip()}")
                        if len(lines) > 1:
                            print(f"      末行: {lines[-1].strip()}")
            except:
                pass
    else:
        print("   ❌ 未找到任何日志文件")

def create_model_based_visualization(result_dir):
    """基于模型信息创建可视化"""
    print(f"\n📊 创建模型分析图表:")
    
    # 尝试从两个模型文件提取信息
    model_info = {}
    
    for filename in ['best.pth', 'latest.pth']:
        filepath = result_dir / filename
        if filepath.exists():
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                model_info[filename] = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'miou': checkpoint.get('best_miou', 'unknown'),
                    'file_size': filepath.stat().st_size / (1024*1024),
                    'has_frac': any('frac' in k.lower() for k in checkpoint.get('model', {}).keys())
                }
            except:
                model_info[filename] = None
    
    if not any(model_info.values()):
        print("   ❌ 无法读取模型信息")
        return
    
    # 创建对比图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('UniMatch Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. 模型对比
    models = []
    epochs = []
    mious = []
    sizes = []
    
    for filename, info in model_info.items():
        if info:
            models.append(filename.replace('.pth', ''))
            epochs.append(info['epoch'] if isinstance(info['epoch'], int) else 0)
            mious.append(info['miou'] if isinstance(info['miou'], (int, float)) else 0)
            sizes.append(info['file_size'])
    
    if models:
        # 轮数对比
        ax1.bar(models, epochs, color=['skyblue', 'lightcoral'][:len(models)])
        ax1.set_title('Training Epochs')
        ax1.set_ylabel('Epochs')
        
        # mIoU对比
        if any(mious):
            ax2.bar(models, mious, color=['lightgreen', 'orange'][:len(models)])
            ax2.set_title('Best mIoU')
            ax2.set_ylabel('mIoU')
            ax2.set_ylim(0, 1)
        
        # 文件大小对比
        ax3.bar(models, sizes, color=['gold', 'plum'][:len(models)])
        ax3.set_title('Model Size (MB)')
        ax3.set_ylabel('Size (MB)')
    
    # 4. 分数阶融合状态
    ax4.axis('off')
    summary_text = "🎯 训练总结\n" + "="*20 + "\n"
    
    for filename, info in model_info.items():
        if info:
            summary_text += f"\n📄 {filename}:\n"
            summary_text += f"   轮数: {info['epoch']}\n"
            summary_text += f"   mIoU: {info['miou']}\n"
            summary_text += f"   大小: {info['file_size']:.1f}MB\n"
            summary_text += f"   分数阶融合: {'✅' if info['has_frac'] else '❌'}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图表
    save_path = result_dir / 'model_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 模型分析图表已保存: {save_path}")

def extract_metrics_from_tensorboard():
    """从TensorBoard事件文件提取指标"""
    print(f"\n📈 尝试从TensorBoard提取指标:")
    
    event_files = list(Path("/root/autodl-tmp/unimatch_workspace/exp/pascal/unimatch/r101/732").glob("events.out.*"))
    
    if not event_files:
        print("   ❌ 未找到TensorBoard事件文件")
        return
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        for event_file in event_files:
            print(f"   📊 分析事件文件: {event_file.name}")
            
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # 获取标量数据
            scalar_tags = ea.Tags()['scalars']
            print(f"   可用指标: {scalar_tags}")
            
            # 提取并可视化指标
            if scalar_tags:
                fig, axes = plt.subplots(1, len(scalar_tags), figsize=(15, 5))
                if len(scalar_tags) == 1:
                    axes = [axes]
                
                for i, tag in enumerate(scalar_tags):
                    scalar_events = ea.Scalars(tag)
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    
                    axes[i].plot(steps, values, linewidth=2)
                    axes[i].set_title(tag)
                    axes[i].set_xlabel('Step')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_path = Path("/root/autodl-tmp/unimatch_workspace/exp/pascal/unimatch/r101/732") / 'tensorboard_metrics.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ TensorBoard指标图表已保存: {save_path}")
            
    except ImportError:
        print("   ⚠️ 需要安装tensorboard: pip install tensorboard")
    except Exception as e:
        print(f"   ❌ 提取失败: {e}")

if __name__ == "__main__":
    direct_model_analysis()
    extract_metrics_from_tensorboard()
    print("\n✅ 直接模型分析完成！")