# NMGrad

This is the source code described in the paper "NMGrad: Advancing Histopathological Bladder Cancer Grading with Weakly Supervised Deep Learning" by Saul Fuster, Umay Kiraz, Trygve Eftestøl, Emiel A.M. Janssen, and Kjersti Engan  - under revision.

### 1 - Abstract
The most prevalent form of bladder cancer is urothelial carcinoma, characterized by a high recurrence rate and substantial lifetime treatment costs for patients. Grading is a prime factor for patient risk stratification, although it suffers from inconsistencies and variations among pathologists. Moreover, absence of annotations in medical imaging difficults training deep learning models. To address these challenges, we introduce a pipeline designed for bladder cancer grading using histological slides. First, it extracts urothelium tissue tiles at different magnification levels, employing a convolutional neural network for processing for feature extraction. Then, it engages in the slide-level prediction process. It employs a nested multiple instance learning approach with attention to predict the grade. To distinguish different levels of malignancy within specific regions of the slide, we include the origins of the tiles in our analysis. The attention scores at region level is shown to correlate with verified high-grade regions, giving some explainability to the model. Clinical evaluations demonstrate that our model consistently outperforms previous state-of-the-art methods.

<p align="center">
    <img src="images/pipeline.png">
</p>

### 2 - How to use

This codebase presents a two-part framework for image analysis, comprising a tissue segmentation module for segmenting tissue regions and a nested Multiple Instance Learning (MIL) classification module.

**Tissue Segmentation Module:**
- The `segmentation/main_segmentation.py` script implements a tissue segmentation algorithm for identifying and segmenting tissue regions in histopathological images.
- The segmentation model is the result of [A Multiscale Approach for Whole-Slide Image Segmentation of five Tissue Classes in Urothelial Carcinoma Slides](https://github.com/Biomedical-Data-Analysis-Laboratory/multiscale-tissue-segmentation-for-urothelial-carcinoma)
- Tiles are extracted at 400x, 100x and 25x. The degree of overlapping, tile size, and others can be tuned.
- A CSV file is generated to maintain the relationships between tile triplets.

**Classification Module:**
- The `grading/main_grading.py` script builds a nested MIL classification model.
- It uses feature embeddings from the contrastive learning module, clinicopathological data, or both for image classification.
- The nested MIL approach hierarchically combines information from image regions (bags) to make a final classification decision.
- The script provides options for configuring the classifier architecture, handling multi-modal data, and specifying MIL pooling techniques.
- You can easily switch between using feature embeddings, clinicopathological data, or a combination as input.

**Usage:**
- Use the `segmentation/main_segmentation.py` script to extract tiles and define regions of urothelium. Adjust extraction settings as needed.
- After data extraction, you can use the `grading/main_grading.py` script to train the nested MIL grading classifier. Specify the input data sources and adjust the model architecture according to your requirements.
- The `grading/inference.py` script allows you to perform inference on new images using the trained classification model.

**Dependencies:**
- Ensure that you have the required dependencies listed in the `requirements.txt` file.

### 3 - Link to paper
TBA

### 4 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper if you use it in your research.
```
TBA
```

