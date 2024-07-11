### Table of Contents
1. [Team](#team)
2. [Overview](#overview)
3. [Repository Structure](#repository-structure)
4. [Program Usage](#program-usage)

### Team
---
<ins>P1 Group 1</ins>
- Peter Febrianto Afandy (2200959)
- Adrian Pang Zi Jian (2200692)
- Ryan Lai Wei Shao (2201159)
- Tng Jian Rong (2201014)
- Lionel Sim Wei Xian (2201132)
- Muhammad Nur Dinie Bin Aziz (2200936)

### Overview
---
This project aims to address the challenge of brain cancer detection, through the multi-class classification of Magnetic Resonance Imaging (MRI) brain image scans using DL approaches. Brain MRI image scans will be classified into four distinct categories: glioma, meningioma, pituitary and no tumour,

The primary objectives of this project are to understand the intricacies of detecting brain tumors from MRI scans, implement deep learning models using various approaches, and evaluate their performance rigorously. In particular, we evaluate four deep learning approaches consisting of nine models, as outlined below.

1. Conditional Generative Adversarial Network (cGAN)
2. U-Net
3. Hybrid (Hybrid of CNN and RNN)
4. Transfer Learning
    - DenseNet-121
    - MobileNetV2
    - MobileNetV3
    - EfficientNet
    - Xception
    - ResNet50V2

### Repository Structure
---
```
requirements.txt

mac_requirements.txt

dataset/ (contains dataset files)

gan/ (directory for cGAN source code)

cnn-rnn-hybrid/ (directory for CNN-RNN Hybrid source code)

u-net/ (directory for U-Net source code)

transfer_learning (directory for transfer learning models' source code)

README.md (this file)
```

### Program Usage
---
1. Create a Python `virtualenv` on your local environment:
    ```
    python3 -m venv .venv
    ```
2. Install the necessary project dependencies:
    ```
    pip3 install -r requirements.txt
    ```

    For macOS users with Apple Silicon, you may want to install additional packages provided by [Apple](https://developer.apple.com/metal/tensorflow-plugin/) to utilise your device's GPU for enhanced model training. The consolidated project dependencies for such users is provided in the `mac_requirements.txt` file:
    ```
    pip3 install -r mac_requirements.txt
    ```
    **Note: tensorflow-metal is currently only supported for Python 3.8, 3.9 and 3.10*
3. Run the interactive Python notebook to train/test the model, ensuring that you've linked the notebook to the correct Python `virtualenv`. To avoid having to train models from scratch, you may choose to load the exported `.pkl` or `.keras` fine-tuned models in each model's respective directory.