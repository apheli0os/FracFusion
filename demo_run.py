import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 本地导入
from fractional_fusion import FractionalFusionUp

def load_image(path, size=(256, 256)):
    img = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)/255.0
    ten = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return ten, img

def make_synthetic(size=(256,256)):
    H,W = size
    arr = np.zeros((H,W,3), np.float32)
    arr[:, :W//2, :] = 0.2
    arr[:, W//2:, :] = 0.8
    # 斜线
    for i in range(min(H,W)):
        y = i
        x = (i*3//2)%W
        arr[y:y+2, x:x+2, :] = 0.0
    ten = torch.fromnumpy(arr).permute(2,0,1).unsqueeze(0)
    return ten, Image.fromarray((arr*255).astype(np.uint8))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    if img_path and os.path.isfile(img_path):
        x, _ = load_image(img_path, size=(256,256))
    else:
        print("未提供有效图片路径，使用合成图演示。")
        x, _ = make_synthetic(size=(256,256))

    x = x.to(device)

    # 选择 up_factor=1 便于与原图同尺寸对比；如需上采样，把 up_factor=2
    net = FractionalFusionUp(in_channels=3, up_factor=1, kernel_size=5, vmax=1.2, hidden=48, mode='both').to(device)
    net.eval()

    with torch.no_grad():
        # 同分辨率处理，获得低通/高通分量
        y, y_lp, y_hp, orders, gate = net.apply_fractional_operator(x, return_parts=True)

    def to_np(t):
        t = t.squeeze(0).permute(1,2,0).clamp(0,1).detach().cpu().numpy()
        return t

    x_np   = to_np(x)
    y_np   = to_np(y)
    ylp_np = to_np(y_lp)
    yhp_np = to_np((y_hp - y_hp.min())/(y_hp.max()-y_hp.min()+1e-8))  # 归一化仅用于显示
    omap   = orders.squeeze(0).mean(0).detach().cpu().numpy()
    gmap   = gate.squeeze(0).squeeze(0).detach().cpu().numpy()

    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1); plt.imshow(x_np);  plt.title('原图'); plt.axis('off')
    plt.subplot(2,3,2); plt.imshow(ylp_np); plt.title('低通输出(类内平滑)'); plt.axis('off')
    plt.subplot(2,3,3); plt.imshow(yhp_np); plt.title('高通输出(边界增强-可视化归一)'); plt.axis('off')
    plt.subplot(2,3,4); plt.imshow(y_np);  plt.title('混合结果'); plt.axis('off')
    plt.subplot(2,3,5); im=plt.imshow(omap, cmap='RdBu_r', vmin=-1.2, vmax=1.2); plt.title('阶次均值'); plt.axis('off'); plt.colorbar(im, fraction=0.046)
    plt.subplot(2,3,6); im2=plt.imshow(gmap, cmap='viridis'); plt.title('门控g'); plt.axis('off'); plt.colorbar(im2, fraction=0.046)
    plt.tight_layout()
    out_png = 'outputs/fractional_demo.png'
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'已保存可视化: {os.path.abspath(out_png)}')
    plt.show()

if __name__ == '__main__':
    main()