# Back Projection Generative Strategy for Low and Normal Light Image Pairs with Enhanced Statistical Fidelity and Diversity (TCE'24)
By Cheuk-Yiu Chan, Wan-Chi Siu, Yuk-Hee Chan and H. Anthony Chan

<img width="1174" alt="image" src="https://github.com/user-attachments/assets/07607520-7e92-407f-82e0-f0a96956e471">

Paper can be found in [Here](https://ieeexplore.ieee.org/document/10794693)

# BibTex
```
@ARTICLE{10794693,
  author={Chan, Cheuk-Yiu and Siu, Wan-Chi and Chan, Yuk-Hee and Chan, H. Anthony},
  journal={IEEE Transactions on Consumer Electronics}, 
  title={Back Projection Generative Strategy for Low and Normal Light Image Pairs With Enhanced Statistical Fidelity and Diversity}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Diffusion models;Image enhancement;Data models;Deep learning;Image color analysis;Lighting;Colored noise;Training data;Mathematical models;Image synthesis;Low Light Image Enhancement;Image Synthesis;Generative Model;Diffusion;Data Augmentation},
  doi={10.1109/TCE.2024.3516366}}
```

# Abstract
  Low light image enhancement (LLIE) using supervised deep learning is limited by the scarcity of matched low/normal light image pairs. We propose Back Projection Normal-to-Low Diffusion Model (N2LDiff-BP), a novel diffusion-based generative model that realistically transforms normal-light images into diverse low-light counterparts. By injecting noise perturbations over multiple timesteps, our model synthesizes low-light images with authentic noise, blur, and color distortions. We introduce innovative architectural components - Back Projection Attention, BP$`^2`$ Feedforward, and BP Transformer Blocks - that integrate back projection to model the narrow dynamic range and nuanced noise of real low-light images. Experiment and results show N2LDiff-BP significantly outperforms prior augmentation techniques, enabling effective data augmentation for robust LLIE. We also introduce LOL-Diff, a large-scale synthetic low-light dataset. Our novel framework, architectural innovations, and dataset advance deep learning for low-light vision tasks by addressing data scarcity. N2LDiff-BP establishes a new state-of-the-art in realistic low-light image synthesis for LLIE.

# Installation 
The code is implemented in PyTorch. To install the required packages, run:
```
conda env create -f environment.yml
MAX_JOBS=4 pip install flash-attn==2.2.5 --no-build-isolation
```

After installing the required packages, activate the environment by running:

```conda activate N2L```

# Data Preparation 
We use LOL, VELOL and LOL-v2 datasets for training and testing. You can download the datasets from the following links:

1. [LOL](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

2. [VELOL](https://www.dropbox.com/s/vfft7a8d370gnh7/VE-LOL-L.zip?dl=0)

3. [LOL-v2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing)

and put them in the folder `data/` with the following hararchy (note that some of the folder may need renaming):

```
data
    - LOL
        - eval15
        - our485

    - LOLv2
        - Real_captured
            - Test
            - Train

    - VE-LOL-L
        - VE-LOL-L-Cap-Full
            - test
            - train 
```
# Inference
## Pre-trained model
Pre-trained model can be downloaded from [here](https://1drv.ms/f/s!AvJJYu8Th24UjNAW5lfkuKZvA9vI6Q?e=g3RGi9). 
The pre-trained model is required to be placed in the folder `model/` with the following hararchy:
```
model
    - final.ckpt
```

## Inference for unpaired data
To generate low-light images using the pre-trained model, configure the `cfg/test/test_unpaired.yaml` file:
```
test_folder_unpaired: "WHERE THE FOLDER BEING TEST"
results_folder_unpaired: './results'
```
and run:
```
python test_unpaired.py --cfg cfg/test/test_unpaired.yaml
```

## Evaluation for metrics on LOL Dataset
run: 
```
python test.py --cfg cfg/test/test.yaml
```

## Evaluation Results 
<img width="1162" alt="image" src="https://github.com/user-attachments/assets/ebb81f38-b660-41f5-ba6b-b04d618dc0d8">
<img width="1180" alt="image" src="https://github.com/user-attachments/assets/51aadb6a-7c59-4a84-8753-b486869331a4">

# Training 
To train the model, run the following command:
```
python train.py --cfg cfg/train/train.yaml
```

# LOL-Diff Dataset
In this paper, we have also proposed the LOL-Diff dataset. The dataset can be downloaded from [here](https://1drv.ms/f/s!AvJJYu8Th24UjNAXFOnc7GAvJBAlqw?e=zwE1UX).
The dataset is organized in the following way:
```
LOLDiff
    - low_res (one zip file with 919MB)
        - train 
        - test
    - 4K (120GB with 30 split 7z file)
        - train
        - test 
```

The purpose of this dataset is to provide a large-scale synthetic low-light dataset for training and testing. The dataset is generated using the N2LDiff-BP. 

# Contact
Thanks for looking into this repo! If you have any suggestion or question, feel free to leave a message here or contact me via cy3chan@sfu.edu.hk.
