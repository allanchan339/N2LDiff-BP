# Back Projection Generative Strategy for Low and Normal Light Image Pairs with Enhanced Statistical Fidelity and Diversity
By Cheuk-Yiu Chan, Wan-Chi Siu, Yuk-Hee Chan and H. Anthony Chan

# Abstract
  Low light image enhancement (LLIE) using supervised deep learning is limited by the scarcity of matched low/normal light image pairs. We propose Back Projection Normal-to-Low Diffusion Model (N2LDiff-BP), a novel diffusion-based generative model that realistically transforms normal-light images into diverse low-light counterparts. By injecting noise perturbations over multiple timesteps, our model synthesizes low-light images with authentic noise, blur, and color distortions. We introduce innovative architectural components - Back Projection Attention, BP$^2$ Feedforward, and BP Transformer Blocks - that integrate back projection to model the narrow dynamic range and nuanced noise of real low-light images. Experiment and results show N2LDiff-BP significantly outperforms prior augmentation techniques, enabling effective data augmentation for robust LLIE. We also introduce LOL-Diff, a large-scale synthetic low-light dataset. Our novel framework, architectural innovations, and dataset advance deep learning for low-light vision tasks by addressing data scarcity. N2LDiff-BP establishes a new state-of-the-art in realistic low-light image synthesis for LLIE.

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
# Inference:
## Pre-trained model:
Pre-trained model can be downloaded from [here](). 
The pre-trained model is required to be placed in the folder `model/` with the following hararchy:
```
model
    - final.ckpt
```

## Inference for unpaired data:

To generate low-light images using the pre-trained model in [here](), configure the `cfg/test/test_unpaired.yaml` file:
```
test_folder_unpaired: "WHERE THE FOLDER BEING TEST"
results_folder_unpaired: './results'
```
and run:
```
python test_unpaired.py --config cfg/test/test_unpaired.yaml
```

## Evaluation for metrics on LOL Dataset:
run: 
```
python test_metrics.py --config cfg/test/test.yaml
```

To train the model, run the following command:
```
python train.py --cfg cfg/train/train.yaml
```
# Training 
To train the model, run the following command:
```
python train.py --cfg cfg/train/FS/train.yaml
```

# LOL-Diff Dataset
In this paper, we have also proposed the LOL-Diff dataset. The dataset can be downloaded from [here]().
The dataset is organized in the following way:
```
LOLDiff
    - low_res
        - train 
        - test
    - 4K
        - train
        - test 
```

The purpose of this dataset is to provide a large-scale synthetic low-light dataset for training and testing. The dataset is generated using the N2LDiff-BP. 

# Contact
Thanks for looking into this repo! If you have any suggestion or question, feel free to leave a message here or contact me via cy3chan@sfu.edu.hk.