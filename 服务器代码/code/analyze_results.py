#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch 训练结果分析脚本
分析训练日志、模型性能和可视化结果
"""

import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_training_results(result_dir):
    """分析训练结果的主函数"""
    result_dir = Path(result_dir)
    print(f"🔍 分析训练结果目录: {result_dir}")
    
    # 1. 分析模型文件
    analyze_model_files(result_dir)
    
    # 2. 分析训练日志
    analyze_training_logs(result_dir)
    
    # 3. 分析事件文件（如果有tensorboard日志）
    analyze_tensorboard_events(result_dir)
    
    # 4. 生成总结报告
    generate_summary_report(result_dir)

def analyze_model_files(result_dir):
    """分析模型文件"""
    print("\n📊 模型文件分析:")
    
    # 检查模型文件
    model_files = {
        'best.pth': '最佳模型',
        'latest.pth': '最新模型'
    }
    
    for filename, description in model_files.items():
        filepath = result_dir / filename
        if filepath.exists():
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                print(f"✅ {description} ({filename}):")
                print(f"   - 文件大小: {filepath.stat().st_size / (1024*1024):.1f} MB")
                
                if 'epoch' in checkpoint:
                    print(f"   - 训练轮数: {checkpoint['epoch']}")
                if 'best_miou' in checkpoint:
                    print(f"   - 最佳mIoU: {checkpoint['best_miou']:.4f}")
                if 'model' in checkpoint:
                    model_size = sum(p.numel() for p in checkpoint['model'].values())
                    print(f"   - 模型参数量: {model_size/1e6:.1f}M")
                
                # 检查是否包含分数阶融合参数
                frac_params = [k for k in checkpoint['model'].keys() if 'frac_up' in k]
                if frac_params:
                    print(f"   - 分数阶融合参数: {len(frac_params)} 个")
                    print(f"     包含: {', '.join(frac_params[:3])}{'...' if len(frac_params) > 3 else ''}")
                else:
                    print("   - 未发现分数阶融合参数")
                    
            except Exception as e:
                print(f"❌ 无法加载 {filename}: {e}")
        else:
            print(f"❌ 未找到 {description} ({filename})")

def analyze_training_logs(result_dir):
    """分析训练日志"""
    print("\n📈 训练日志分析:")
    
    # 查找日志文件
    log_files = list(result_dir.glob("*.log")) + list(result_dir.glob("train.out"))
    
    if not log_files:
        print("❌ 未找到日志文件")
        return
    
    for log_file in log_files:
        print(f"\n分析日志文件: {log_file.name}")
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取训练信息
            epochs, losses, mious = extract_training_metrics(content)
            
            if epochs:
                print(f"✅ 训练轮数: {len(epochs)} 轮")
                print(f"✅ 最终loss: {losses[-1]:.4f}")
                if mious:
                    print(f"✅ 最佳mIoU: {max(mious):.4f}")
                
                # 绘制训练曲线
                plot_training_curves(epochs, losses, mious, result_dir)
            else:
                print("⚠️ 未找到训练指标")
                
        except Exception as e:
            print(f"❌ 分析日志失败: {e}")

def extract_training_metrics(log_content):
    """从日志中提取训练指标"""
    epochs = []
    losses = []
    mious = []
    
    # 匹配训练loss的模式
    loss_pattern = r'Epoch:\s*(\d+).*?Total loss:\s*([\d.]+)'
    loss_matches = re.findall(loss_pattern, log_content)
    
    for epoch, loss in loss_matches:
        epochs.append(int(epoch))
        losses.append(float(loss))
    
    # 匹配验证mIoU的模式
    miou_pattern = r'Epoch:\s*(\d+).*?mIoU:\s*([\d.]+)'
    miou_matches = re.findall(miou_pattern, log_content)
    
    epoch_to_miou = {int(epoch): float(miou) for epoch, miou in miou_matches}
    mious = [epoch_to_miou.get(epoch, 0) for epoch in epochs]
    
    return epochs, losses, mious

def plot_training_curves(epochs, losses, mious, result_dir):
    """绘制训练曲线"""
    try:
        plt.figure(figsize=(12, 5))
        
        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # mIoU曲线
        plt.subplot(1, 2, 2)
        if any(mious):
            valid_epochs = [e for e, m in zip(epochs, mious) if m > 0]
            valid_mious = [m for m in mious if m > 0]
            if valid_epochs:
                plt.plot(valid_epochs, valid_mious, 'r-', linewidth=2, label='Validation mIoU')
                plt.xlabel('Epoch')
                plt.ylabel('mIoU')
                plt.title('Validation mIoU Curve')
                plt.grid(True, alpha=0.3)
                plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_path = result_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 训练曲线已保存到: {save_path}")
        
    except Exception as e:
        print(f"⚠️ 绘制训练曲线失败: {e}")

def analyze_tensorboard_events(result_dir):
    """分析TensorBoard事件文件"""
    print("\n📋 TensorBoard事件分析:")
    
    # 查找事件文件
    event_files = list(result_dir.glob("events.out.tfevents.*"))
    
    if not event_files:
        print("❌ 未找到TensorBoard事件文件")
        return
    
    try:
        # 需要安装tensorboard: pip install tensorboard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        for event_file in event_files:
            print(f"✅ 发现事件文件: {event_file.name}")
            
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # 获取可用的标量
            scalar_tags = ea.Tags()['scalars']
            print(f"   记录的指标: {', '.join(scalar_tags)}")
            
    except ImportError:
        print("⚠️ 需要安装tensorboard来分析事件文件: pip install tensorboard")
    except Exception as e:
        print(f"⚠️ 分析事件文件失败: {e}")

def generate_summary_report(result_dir):
    """生成总结报告"""
    print("\n📄 生成总结报告:")
    
    report = {
        'analysis_time': datetime.now().isoformat(),
        'result_directory': str(result_dir),
        'files_found': [],
        'model_info': {},
        'training_summary': {}
    }
    
    # 收集文件信息
    for file in result_dir.iterdir():
        if file.is_file():
            report['files_found'].append({
                'name': file.name,
                'size_mb': file.stat().st_size / (1024*1024),
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
    
    # 尝试加载最佳模型信息
    best_model_path = result_dir / 'best.pth'
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            report['model_info'] = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_miou': checkpoint.get('best_miou', 'unknown'),
                'has_fractional_fusion': any('frac_up' in k for k in checkpoint.get('model', {}).keys())
            }
        except:
            pass
    
    # 保存报告
    report_path = result_dir / 'analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 分析报告已保存到: {report_path}")
    
    # 打印关键信息
    print("\n🎯 关键结果:")
    if 'best_miou' in report['model_info']:
        print(f"   最佳mIoU: {report['model_info']['best_miou']}")
    if 'epoch' in report['model_info']:
        print(f"   训练轮数: {report['model_info']['epoch']}")
    if report['model_info'].get('has_fractional_fusion'):
        print("   ✅ 包含分数阶融合模块")
    else:
        print("   ❌ 未包含分数阶融合模块")

def compare_models(result_dirs):
    """比较多个训练结果"""
    print("\n🔍 模型对比分析:")
    
    results = []
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        if not result_dir.exists():
            continue
            
        best_model_path = result_dir / 'best.pth'
        if not best_model_path.exists():
            continue
            
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            results.append({
                'name': result_dir.name,
                'path': str(result_dir),
                'miou': checkpoint.get('best_miou', 0),
                'epoch': checkpoint.get('epoch', 0),
                'has_frac': any('frac_up' in k for k in checkpoint.get('model', {}).keys())
            })
        except:
            continue
    
    if results:
        # 按mIoU排序
        results.sort(key=lambda x: x['miou'], reverse=True)
        
        print("排名  | 模型名称 | mIoU   | 轮数 | 分数阶融合")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            frac_mark = "✅" if result['has_frac'] else "❌"
            print(f"{i:2d}    | {result['name'][:15]:<15} | {result['miou']:.4f} | {result['epoch']:3d}  | {frac_mark}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析UniMatch训练结果')
    parser.add_argument('--result-dir', type=str, 
                       default='/root/autodl-tmp/unimatch_workspace/exp/pascal/unimatch/r101/732',
                       help='训练结果目录')
    parser.add_argument('--compare', nargs='+', help='比较多个训练结果目录')
    
    args = parser.parse_args()
    
    print("🎯 UniMatch 训练结果分析")
    print("=" * 50)
    
    if args.compare:
        compare_models(args.compare)
    else:
        analyze_training_results(args.result_dir)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()