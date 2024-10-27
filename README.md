# Deep learning presents better arthroscopic field clarity

<div align="center">

📑[**Abstract**](#-abstract) **|** 🔧[**Install**](#-dependencies-and-installation)  **|** 💻[**Train**](#-train) **|** ⚡[**Usage**](#-inference)  **|** 👀[**Demos**](#-demo-videos) **|** 📧[**Contact**](#-contact)

</div>

---

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
### Homorrhage and Clear Dataset Preparation
### Train Hemorrhage Generator
### Stage 1 training
### Synthesis Data(Arthroscopic Degradation)
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
