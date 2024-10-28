import math
import torch
import random
import cv2
import os
import numpy as np
import torchvision.transforms as transforms

from torch.nn import functional as F

from utils import USMSharp, DiffJPEG, filter2D, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils import circular_lowpass_kernel, random_mixed_kernels, img2tensor, TinyRRDBNet



def degradation_process(x, config):

    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
    hemorrhage = TinyRRDBNet().cuda()

    pulse_tensor = torch.zeros(21, 21).float()
    pulse_tensor[10, 10] = 1
    
    
    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(config['kernel_range'])
    # if np.random.uniform() < config['sinc_prob']:
    #     # this sinc filter setting is for kernels ranging from [7, 21]
    #     if kernel_size < 13:
    #         omega_c = np.random.uniform(np.pi / 3, np.pi)
    #     else:
    #         omega_c = np.random.uniform(np.pi / 5, np.pi)
    #     kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    # else:
    kernel = random_mixed_kernels(
        config['kernel_list'],
        config['kernel_prob'],
        kernel_size,
        config['blur_sigma'],
        config['blur_sigma'], [-math.pi, math.pi],
        config['betag_range'],
        config['betap_range'],
        noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(config['kernel_range'])
    if np.random.uniform() < config['sinc_prob2']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            config['kernel_list2'],
            config['kernel_prob2'],
            kernel_size,
            config['blur_sigma2'],
            config['blur_sigma2'], [-math.pi, math.pi],
            config['betag_range2'],
            config['betap_range2'],
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < config['final_sinc_prob']:
        kernel_size = random.choice(config['kernel_range'])
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel).cuda()
    else:
        sinc_kernel = pulse_tensor.cuda()

    gt = x.astype(np.float32) / 255.0
    # BGR to RGB, HWC to CHW, numpy to tensor
    gt = img2tensor([gt], bgr2rgb=True, float32=True)[0].unsqueeze(0)

        
    gt = gt.cuda()
    gt_usm = usm_sharpener(gt)

    gt_h, gt_w = gt.size()[2:4]

    kernel = torch.FloatTensor(kernel).cuda()
    kernel2 = torch.FloatTensor(kernel2).cuda()
    
    
    
    
    
    """
    The arthroscopic degradation process
    """
    
    
    
    
    
    # ----------------------- The arthroscopic degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, kernel)
        
    b1 = out
    
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, config['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(config['resize_range'][0], 1)
    else:
        scale = 1
    out = F.interpolate(gt_usm, scale_factor=scale, mode='bicubic')

    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size = (int(gt_h / (0.5 * config['scale'])), int(gt_w / (0.5 * config['scale']))), mode=mode)    
    
    r1 = out

#     # hemorrhage imitation
#     if np.random.uniform() < config['hemorrhage_prob']:
#         hemorrhage.load_state_dict(torch.load('hemorrhage.pth', map_location='cuda:0'))
#         hemorrhage.eval()
#         with torch.no_grad():
#             out = hemorrhage(out)
    
    # add noise
    gray_noise_prob = config['gray_noise_prob']
    if np.random.uniform() < config['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=config['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    
    n1 = out
    
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)
    
    j1 = out
    
    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < config['second_blur_prob']:
        out = filter2D(out, kernel2)

    b2 = out
    
    # random resize
    # updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob2'])[0]
    # if updown_type == 'up':
    #     scale = np.random.uniform(1, config['resize_range2'][1])
    # elif updown_type == 'down':
    #     scale = np.random.uniform(config['resize_range2'][0], 1)
    # else:
    #     scale = 1
    # mode = random.choice(['area', 'bilinear', 'bicubic'])
    # out = F.interpolate(
    #     out, size=(int(gt_h / config['scale'] * scale), int(gt_w / config['scale'] * scale)), mode=mode)
    
    # add noise
    gray_noise_prob = config['gray_noise_prob2']
    if np.random.uniform() < config['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=config['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    
    n2 =out
    
#     JPEG compression + the final sinc filter
#     We also need to resize images to desired sizes. We group [resize back + sinc filter] together
#     as one operation.
#     We consider two orders:
#       1. [resize back + sinc filter] + JPEG compression
#       2. JPEG compression + [resize back + sinc filter]
#     Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    
# if np.random.uniform() < 0.5:
    # resize back + the final sinc filter
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, size=(gt_h // config['scale'], gt_w // config['scale']), mode=mode)
    out = filter2D(out, sinc_kernel)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
# else:
    # JPEG compression
    # jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
    # out = torch.clamp(out, 0, 1)
    # out = jpeger(out, quality=jpeg_p)
    # # resize back + the final sinc filter
    # mode = random.choice(['area', 'bilinear', 'bicubic'])
    # out = F.interpolate(out, size=(gt_h // config['scale'], gt_w // config['scale']), mode=mode)
    # out = filter2D(out, sinc_kernel)

    return out


def hemorrhage_test(x):
    
    hemorrhage = TinyRRDBNet().cuda()

    gt = np.array(x).astype(np.float32) / 255.0
    # BGR to RGB, HWC to CHW, numpy to tensor
    gt = img2tensor([gt], bgr2rgb=True, float32=True)[0].unsqueeze(0)

    gt = gt.cuda()
        
    hemorrhage.load_state_dict(torch.load('G_hemorrhage_epoch_038.pth', map_location='cuda:0'))
    hemorrhage.eval()
    with torch.no_grad():
        out = hemorrhage(gt)
            
    return out


config = {
    # scale
    "scale" : 4,
    # hemorrhage imitation
    "hemorrhage_prob" : 0.5,
    # the first degradation process
    "kernel_range": [2 * v + 1 for v in range(3, 11)],  # kernel size ranges from 7 to 21
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob": 0.1,
    "blur_sigma": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "blur_kernel_size": 21,
    "resize_prob": [0.2, 0.7, 0.1],  # up, down, keep
    "resize_range": [0.15, 1.5],
    # hemorrhage prob
    "hemorrhage_prob": 0.5,
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 30],
    "poisson_scale_range": [0.05, 3],
    "gray_noise_prob": 0.4,
    "jpeg_range": [50, 100],
    # second hemorrhage imitation
    "second_hemorrhage_prob": 0.5,
    # the second degradation process
    "second_blur_prob": 0.8,
    "blur_kernel_size2": 21,
    "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob2": 0.1,
    "blur_sigma2": [0.2, 1.5],
    "betag_range2": [0.5, 4],
    "betap_range2": [1, 2],
    "final_sinc_prob": 0.8,
    "resize_prob2": [0.3, 0.4, 0.3],  # up, down, keep,
    "resize_range2": [0.3, 1.2],
    "gaussian_noise_prob2": 0.5,
    "noise_range2": [1, 25],
    "poisson_scale_range2": [0.05, 2.5],
    "gray_noise_prob2": 0.4,
    "jpeg_range2": [50, 100],
}


"""
input_file_path = r'Z:\Hemorrhage\pythonProject1\dataset\clear_frame\0002.png'
x = cv2.imread(input_file_path)

if x is None:
    raise FileNotFoundError(f"图像文件未能读取，请检查路径是否正确: {input_file_path}")

out = degradation_process(x=x, config=config)

out_np = out.detach().cpu().numpy()

img = np.transpose(out_np, (0, 2, 3, 1))  # 从 (B, C, H, W) 转为 (B, H, W, C)

file_name = os.path.basename(input_file_path)  # 提取原始文件名，如 '002.png'
# 构造新文件名，加上前缀 'h+'
new_file_name = 'a' + file_name

# 指定保存路径，可以是原目录或新的目录
save_path = r'Z:\1'

# 保存图像为 .png
cv2.imwrite(save_path, (img[0] * 255).astype(np.uint8))  # 假设保存第0张图像

print(f'图像已保存到: {save_path}')
"""