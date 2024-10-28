import os
import cv2
import torchvision.transforms as transforms
from degradation import degradation_process, config, hemorrhage_test
from utils import save_images_tensor
from PIL import Image

input_file_dir = r'DIV2K_train_LR_bicubic_X4'
save_dir_test = r'DIV2K_train_LR_bicubic_X4_sub'

input_files = [f for f in os.listdir(input_file_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for input_file in input_files:
    
    print(f"Processing file: {input_file}")
    
    input_file_path = os.path.join(input_file_dir, input_file)
    x = cv2.imread(input_file_path)

    out = degradation_process(x, config)    
    
    name, ext = os.path.splitext(input_file)
    
    # save_images_tensor(b1, save_dir=save_dir_test, filename=f"{name}_b1{ext}")
    # save_images_tensor(r1, save_dir=save_dir_test, filename=f"{name}_r1{ext}")
    # save_images_tensor(n1, save_dir=save_dir_test, filename=f"{name}_n1{ext}")
    # save_images_tensor(j1, save_dir=save_dir_test, filename=f"{name}_j1{ext}")
    # save_images_tensor(b2, save_dir=save_dir_test, filename=f"{name}_b2{ext}")
    # save_images_tensor(n2, save_dir=save_dir_test, filename=f"{name}_n2{ext}")
    save_images_tensor(out, save_dir=save_dir_test, filename=input_file)