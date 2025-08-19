import torch
from fractional_fusion import FractionalFusionUp


def demo_fractional_fusion_up():
    """使用分数阶融合上采样的简单演示"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32, device=device)
    up = FractionalFusionUp(64, up_factor=2, kernel_size=5, vmax=1.2, hidden=64, mode='both').to(device)
    y, v, g = up(x)
    print('输入:', x.shape, '输出:', y.shape, '阶次:', v.shape, '门控:', g.shape)


if __name__ == '__main__':
    demo_fractional_fusion_up()