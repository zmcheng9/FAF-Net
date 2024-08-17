# FAF-Net for FSMIS

![image](https://github.com/zmcheng9/FAF-Net/blob/main/overview.png)

### Abstract
Manual annotation of massive data in the biomedical field necessitates huge costs, expertise, and privacy considerations. However, Few-Shot Medical Image Segmentation (FSMIS) offers the possibility of learning a segmentation model with excellent performance from limited medical data. In this paper, we propose a Frequency-aware Adaptive Filtering Network (FAF-Net) for FSMIS, which is the first to calibrate the correlations between support prototype and query features from a frequency perspective, addressing the problems of FSMIS. The FAF-Net mines the commonalities between the support prototype and query features to reduce the intra-class differences through an Intra-class Commonality Miner (ICM), and introduces an Adaptive Fourier Filter (AFF) to adaptively filter the spectrum of the attention map, thus balancing the foreground and background classes. Additionally, an Adaptive Threshold Learner (ATL) is incorporated to learn an adaptive threshold for each spatial location of the predicted mask from the support features, overcoming the limitations of single thresholding. Extensive experiments on four generic medical datasets showcase that our model significantly outperforms SoTA methods (exceeding SoTA by 2.13\% on average).

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
2) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)
3) **Prostate-MRI**: [Cross-institution Male Pelvic Structures data set](https://zenodo.org/records/7013610)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 
