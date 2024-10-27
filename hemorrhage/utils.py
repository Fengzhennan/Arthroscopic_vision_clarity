from pprint import pprint
import numpy as np
import random
import torch
import json
import os
import datetime

# def tensor2image(tensor):
#     image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
#     if image.shape[0] == 1:
#         image = np.tile(image, (3, 1, 1))
#     return image.astype(np.uint8)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class replaybuffer():
    def __init__(self, max_size=5):
        assert max_size > 0, "empty buffer or trying to create a buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class lambdalr():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2D') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def checkpoint(net, epoch, config):

    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    save_model_path = os.path.join('./ckpt', start_time)
    os.makedirs(save_model_path, exist_ok=True)

    model_name = 'epoch_{:03d}.pth'.format(epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))

    # Save the config
    config_save_path = os.path.join(save_model_path, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print('Config saved to {}'.format(config_save_path))

def tensor_to_image(tensor):
    # 假设 batch_size = 1，只取第一个样本
    tensor = tensor[0]  # 去掉 batch 维度，变为 CHW
    tensor = tensor.permute(1, 2, 0)  # 从 CHW 转换为 HWC
    image = tensor.cpu().detach().numpy()  # 转为 NumPy 数组
    # 如果需要将范围从 [0, 1] 转为 [0, 255]，可以加这一步
    image = (image * 255).astype(np.uint8)
    return image

def get_available_gpu_ids():
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    return gpu_ids

def save_model(net, path, epoch):
    model_name = 'G_hemorrhage_epoch_{:03d}.pth'.format(epoch)
    save_model_path = os.path.join(path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Generator hemorrhage-stimulation saved to {}'.format(save_model_path))
    
def save_config_json(config, path):
    path = os.path.join(path, 'configuration.json')

    # 将config转换为JSON格式的字符串  
    json_str = json.dumps(config, indent=4, sort_keys=True)  
      
    # 使用pprint打印（主要用于调试或查看）  
    pprint(json.loads(json_str))  # 注意：这里先将json_str转换为dict，因为pprint不能直接打印str  
      
    # 将JSON字符串保存到指定的文件路径  
    with open(path, 'w') as f:  
        f.write(json_str)