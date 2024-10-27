import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import itertools
import datetime
import torch
import wandb
import tqdm
import os

from pprint import pprint
from backbone import generator, discriminator, TinyRRDBNet, UNetDiscriminatorSN, TinyUNetDiscriminatorSN
from utils import replaybuffer, lambdalr, checkpoint, tensor_to_image, get_available_gpu_ids, set_seed, save_model, save_config_json
from dataset import HemorrhageDataset
from loss import TV_Loss, Perceptual_loss, Style_loss


def train():

    wandb.init()

    # get configuration from wandb.config
    config = wandb.config

    gpu_ids = get_available_gpu_ids()

    # detect gpu number
    if len(gpu_ids) < 2:
        raise ValueError("需要至少两个GPU来运行该脚本。")
    
    device_0 = torch.device(f'cuda:{gpu_ids[0]}')
    device_1 = torch.device(f'cuda:{gpu_ids[1]}')

    set_seed(42)

    # gpu 0
    netG_A2B = TinyRRDBNet().to(device_0)
    netD_B = TinyUNetDiscriminatorSN().to(device_0)

    # gpu 1
    netG_B2A = TinyRRDBNet().to(device_1)
    netD_A = TinyUNetDiscriminatorSN().to(device_1)

    # mixed precision training
    scaler_G_A2B = torch.cuda.amp.GradScaler()
    scaler_G_B2A = torch.cuda.amp.GradScaler()
    scaler_D_A = torch.cuda.amp.GradScaler()
    scaler_D_B = torch.cuda.amp.GradScaler()

    # loss_A2B
    loss_gan_A2B = torch.nn.BCEWithLogitsLoss().to(device_0)      # loss_GAN = torch.nn.MSELoss()
    loss_cycle_A2B = torch.nn.L1Loss().to(device_0)
    loss_identity_A2B = torch.nn.L1Loss().to(device_0)
    tv_loss_A2B = TV_Loss(config.tv_weight).to(device_0)
    perceptual_loss_A2B = Perceptual_loss().to(device_0)
    style_loss_A2B = Style_loss().to(device_0)

    # loss_B2A
    loss_gan_B2A = torch.nn.BCEWithLogitsLoss().to(device_1)  # loss_GAN = torch.nn.MSELoss()
    loss_cycle_B2A = torch.nn.L1Loss().to(device_1)
    loss_identity_B2A = torch.nn.L1Loss().to(device_1)
    tv_loss_B2A = TV_Loss(config.tv_weight).to(device_1)
    perceptual_loss_B2A = Perceptual_loss().to(device_1)
    style_loss_B2A = Style_loss().to(device_1)

    opt_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_D_A = torch.optim.Adam(netD_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(netD_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

    lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(opt_G_A2B,
                                                       lr_lambda=lambdalr(config.n_epoch,
                                                                          config.epoch,
                                                                          config.decay_epoch).step
                                                       )

    lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(opt_G_B2A,
                                                       lr_lambda=lambdalr(config.n_epoch,
                                                                          config.epoch,
                                                                          config.decay_epoch).step
                                                       )

    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A,
                                                       lr_lambda=lambdalr(config.n_epoch,
                                                                          config.epoch,
                                                                          config.decay_epoch).step
                                                       )

    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B,
                                                       lr_lambda=lambdalr(config.n_epoch,
                                                                          config.epoch,
                                                                          config.decay_epoch).step
                                                       )

    fake_A_buffer = replaybuffer()
    fake_B_buffer = replaybuffer()

    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = [
        transforms.CenterCrop((1080, 1080)),
        transforms.Resize((540, 540)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]

    dataloader = DataLoader(HemorrhageDataset(transform=transform),
                            batch_size=1,
                            shuffle=True,
                            num_workers=8,
                            )

    input_A_0 = torch.ones([1, 3, 540, 540],
                         dtype=torch.float).to(device_0)
    input_A_1 = torch.ones([1, 3, 540, 540],
                           dtype=torch.float).to(device_1)
    input_B_0 = torch.zeros([1, 3, 540, 540],
                          dtype=torch.float).to(device_0)
    input_B_1 = torch.zeros([1, 3, 540, 540],
                            dtype=torch.float).to(device_1)

    label_real_0 = torch.ones([1, 1, 540, 540], dtype=torch.float,
                            requires_grad=False).to(device_0)
    label_real_1 = torch.ones([1, 1, 540, 540], dtype=torch.float,
                              requires_grad=False).to(device_1)
    label_fake_0 = torch.zeros([1, 1, 540, 540], dtype=torch.float,
                             requires_grad=False).to(device_0)
    label_fake_1 = torch.zeros([1, 1, 540, 540], dtype=torch.float,
                               requires_grad=False).to(device_1)

    step = 0

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    exp_path = os.path.join('./ckpt', "exp_time : {}".format(now))
    os.makedirs(exp_path, exist_ok=True)

    JSON_serializable_config = {k: v for k, v in wandb.config.items()
                                if isinstance(v, (int, float, str, list, dict, type(None)))
                                and (not isinstance(v, dict)
                                     or all(
                    isinstance(x, (int, float, str, list, dict, type(None))) for x in v.values()))}

    # save configuration
    save_config_json(config=JSON_serializable_config, path=exp_path)

    # real-time supervise
    data = []
    columns = ["now", "Epoch", "Ground truth", "Hemorrhage stimulation"]

    for epoch in tqdm.tqdm(range(config.n_epoch)):

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print("Now it's {} and Epoch is {:03d}".format(nowtime, epoch))
        print("——————————————————————————————————————————————————————————")

        for i, batch in enumerate(dataloader):

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("\n" + "——————————————————————" * 8 + "%s" % nowtime)

            real_A_0 = torch.tensor(input_A_0.copy_(batch['A']), dtype=torch.float)
            real_A_1 = torch.tensor(input_A_1.copy_(batch['A']), dtype=torch.float)

            real_B_0 = torch.tensor(input_B_0.copy_(batch['B']), dtype=torch.float)
            real_B_1 = torch.tensor(input_B_1.copy_(batch['B']), dtype=torch.float)

            opt_G_B2A.zero_grad()
            opt_G_A2B.zero_grad()

            with torch.cuda.amp.autocast():
                same_B = netG_A2B(real_B_0)
                loss_identity_A2B_ = loss_identity_A2B(same_B, real_B_0) * 0.5
                same_A = netG_B2A(real_A_1)
                loss_identity_B2A_ = loss_identity_B2A(same_A, real_A_1) * 0.5

                fake_B = netG_A2B(real_A_0)

                # perceptual loss / style loss / tv loss
                loss_perceptual_A2B_ = perceptual_loss_A2B(real_A_0, fake_B)
                loss_style_A2B_ = 0 #style_loss_A2B(real_B_0, fake_B) # 0 #
                tv_loss_A2B_ = tv_loss_A2B(fake_B)

                pred_fake = netD_B(fake_B)
                loss_gan_A2B_ = loss_gan_A2B(pred_fake, label_real_0)
                fake_A = netG_B2A(real_B_1)
                
                # perceptual loss / style loss / tv loss
                loss_perceptual_B2A_ = perceptual_loss_B2A(real_B_1, fake_A)
                loss_style_B2A_ = 0 #style_loss_B2A(real_A_1, fake_A) # 0 #
                tv_loss_B2A_ = tv_loss_B2A(fake_A)

                # check generator
                if (epoch % 10 == 0 or epoch == 200) and i == 10:
                    with torch.no_grad():
                        gt = tensor_to_image(real_B_1.detach())
                        he = tensor_to_image(fake_A.detach())

                    # Real-time supervision
                    data.append([now, epoch, wandb.Image(gt), wandb.Image(he)])
                    visualization = wandb.Table(data=data, columns=columns)

                    wandb.log({"Real-time supervision": visualization, "epoch": epoch, "iter": step})

                    # save_model
                    save_model(netG_B2A, path=exp_path, epoch=epoch)
                                      
                # transfer device
                fake_B = fake_B.clone().detach().to(device_1)
                pred_fake = netD_A(fake_B)
                loss_GAN_B2A_ = loss_gan_B2A(pred_fake, label_real_1)

                # cycle loss and style loss
                # B2A
                recovered_A = netG_B2A(fake_B)
                loss_cycle_B2A_ = loss_cycle_B2A(recovered_A, real_A_1) * 1.0

                # cycle loss and style loss
                # A2B
                fake_A = fake_A.clone().detach().to(device_0)
                recovered_B = netG_A2B(fake_A)
                loss_cycle_A2B_ = loss_cycle_A2B(recovered_B, real_B_0) * 1.0

                # G_B2A let's go
                loss_G_B2A = loss_identity_B2A_ * config.lambda_i + \
                             loss_cycle_B2A_ * config.lambda_c + \
                             loss_GAN_B2A_ * config.lambda_g + \
                             loss_perceptual_B2A_ * config.lambda_p + \
                             loss_style_B2A_ * config.lambda_s + \
                             tv_loss_B2A_

                loss_G_A2B = loss_identity_A2B_ * config.lambda_i + \
                             loss_cycle_A2B_ * config.lambda_c + \
                             loss_gan_A2B_ * config.lambda_g + \
                             loss_perceptual_A2B_ * config.lambda_p + \
                             loss_style_A2B_ * config.lambda_s + \
                             tv_loss_A2B_

                print("Total_loss_net_G_B2A : {:.6f}".format(loss_G_B2A))
                print("CycleGAN_loss: identity_loss : {:.6f}, cycle_loss : {:.6f}, gan_loss : {:.6f}".format(loss_identity_B2A_, loss_cycle_B2A_, loss_GAN_B2A_))
                print("perceptual_loss : {:.6f},  tv_loss:{:.6f}".format(loss_perceptual_B2A_, tv_loss_B2A_))
                print("Style_loss_:{}".format(loss_style_B2A_))
                print("\n" + "——————————————————————" * 8 + "%s" % nowtime)

                wandb.log({"Total_loss_net_G_B2A :": loss_G_B2A,
                           "identity_loss": loss_identity_B2A_, "cycle_loss": loss_cycle_B2A_, "gan_loss": loss_GAN_B2A_,
                           "perceptual_loss": loss_perceptual_B2A_, "style_loss": loss_style_B2A_, "tv_loss_B2A": tv_loss_B2A_
                           })

            scaler_G_B2A.scale(loss_G_B2A).backward()
            scaler_G_B2A.step(opt_G_B2A)
            scaler_G_B2A.update()

            scaler_G_A2B.scale(loss_G_A2B).backward()
            scaler_G_A2B.step(opt_G_A2B)
            scaler_G_A2B.update()

            # D A

            opt_D_A.zero_grad()

            with torch.cuda.amp.autocast():
                fake_A = fake_A.clone().detach().to(device_1)
                pred_real = netD_A(real_A_1)
                loss_D_real = loss_gan_B2A(pred_real, label_real_1)

                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = loss_gan_B2A(pred_fake, label_fake_1)

                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                print("Discriminator_loss_D_A:{:.6f}".format(loss_D_A))
                wandb.log({"loss_D_A": loss_D_A})

            scaler_D_A.scale(loss_D_A).backward()
            scaler_D_A.step(opt_D_A)
            scaler_D_A.update()

            # D B
            opt_D_B.zero_grad()

            with torch.cuda.amp.autocast():
                fake_B = fake_B.clone().detach().to(device_0)
                pred_real = netD_B(real_B_0)
                loss_D_real = loss_gan_A2B(pred_real, label_real_0)

                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = loss_gan_A2B(pred_fake, label_fake_0)

                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                print("Discriminator_loss_D_B:{:.6f}".format(loss_D_B))
                wandb.log({"loss_D_B": loss_D_B})

            scaler_D_B.scale(loss_D_B).backward()
            scaler_D_B.step(opt_D_B)
            scaler_D_B.update()

            torch.cuda.empty_cache()
            
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("\n" + "——————————————————————" * 8 + "%s" % nowtime)
            print("Now iter {} ends ~!".format(step))
            
            step += 1

        lr_scheduler_G_A2B.step()
        lr_scheduler_G_B2A.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    # Mark the run as finished
    wandb.finish()

if __name__ == '__main__':

    sweep_config = {
        'method': 'grid',
        'parameters': {
            'lr': {'values': [2e-4, 2e-5]},
            'n_epoch': {'value': 100},
            'epoch': {'value': 0},
            'decay_epoch': {'values': [50, 75]},
            'tv_weight': {'values': [1]},
            'lambda_i': {'values': [0.5, 1]},
            'lambda_c': {'values': [0.5, 1]},
            'lambda_g': {'values': [1]},
            'lambda_p': {'values': [0]},
            'lambda_s': {'values': [0]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='hemorrhage')

    wandb.agent(sweep_id, function=train)