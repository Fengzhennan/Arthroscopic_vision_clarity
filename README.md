<div align="center">

# Deep learning presents better arthroscopic field clarity

<div align="center">
    
📑[**Abstract**](#-abstract) **|** 🔧[**Install**](#-dependencies-and-installation)  **|** 💻[**Train**](#-train) **|** ⚡[**Usage**](#-inference)  **|** 👀[**Demos**](#-demo-videos) **|** 📧[**Contact**](#-contact)

<div align="left">

<!---------------------------------- Abstract --------------------------->
## 📑 Abstract

<!---------------------------------- Install ---------------------------->
## 🔧 Install
1. Clone repo
    ```bash
    git clone https://github.com/Fengzhennan/Deep_learning_presents_better_arthroscopic_vision_clarity.git
    cd Deep_learning_presents_better_arthroscopic_vision_clarity
    ```
2. Install dependent packages
    ```bash
    pip install -r requirements.txt
    
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    ```
<!----------------------------------  Train  ---------------------------->
## 💻 Train
### Overview
The training has been divided into three stages. These two stages have the same data synthesis process and training pipeline, except for the loss functions. We also recommend using the official 4-GPU training setup. Specifically,

1. We first train Real-ESRNet from the pre-trained ESRGAN model, using the L1 loss function with a combination of the clear image dataset and high-order degradation model.
2. Next, we use the trained Real-ESRNet model to initialize the generator, training Real-ESRGAN with a combination of L1 loss, perceptual loss, and GAN loss, again using the clear image dataset combined with the high-order degradation model.
3. Finally, we apply the model, which has demonstrated generalizability and robustness in arthroscopic fields, to the clear image dataset combined with the arthroscopic degradation model.

### Homorrhage and Clear Dataset Preparation
#### Step 1: [Optional] Generate multi-scale images
We placed the clear image dataset, i.e., the Ground-Truth images, in `superresolution/datasets/DF2K/DF2K_HR`.
```bash
python superresolution/scripts/generate_multiscale_DF2K.py --input superresolution/datasets/DF2K/DF2K_HR --output superresolution/datasets/DF2K/DF2K_multiscale
```
#### Step 2: Prepare a txt for meta information in 1st&2nd stages
You can use the [superresolution/scripts/generate_meta_info.py](superresolution/scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:
```bash
python superresolution/scripts/generate_meta_info.py --input superresolution/datasets/DF2K/DF2K_HR superresolution/datasets/DF2K/DF2K_multiscale --root superresolution/datasets/DF2K datasets/DF2K --meta_info superresolution/datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```
#### Step 3: Generation of degraded arthroscopic images
```bash
python superresolution/scripts/datasets/DF2K/test.py
```
#### Step 4: Prepare a txt for meta information in 3rd stages
```bash
python superresolution/scripts/generate_meta_info_pairdata.py --input superresolution/datasets/DF2K/DIV2K_train_HR_sub superresolution/datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub --meta_info superresolution/datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt
```

### Train Hemorrhage Generator
### Stage 1 training
We initiated the first stage of training using a pre-trained ESRGAN model on a publicly available dataset.
### Synthesis Data (Arthroscopic Degradation)
### Stage 2&3 training
<!----------------------------------  Usage  ---------------------------->
## ⚡ Usage
### Usage of Python script

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

Results are in the `results` folder

<!----------------------------------  Usage  ---------------------------->
## 👀 Demos videos

<!---------------------------------- Contact ---------------------------->
## 📧 Contact
Any question refer to email `fengzhennan@hotmail.com`.

<!------------------------------ Acknowledgement ------------------------>
## ❤️‍🩹 Acknowledgement
We extend our gratitude to all the orthopedic surgeons from the Department of Joint Surgery and Sports Medicine at the Third Xiangya Hospital of Central South University for their evaluation of our model. Specifically, we thank the [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) project for its open-source spirit and for inspiring this research.
