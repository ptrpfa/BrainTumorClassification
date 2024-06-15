### Table of Contents
1. [Team](#team)
2. [Overview](#overview)
5. [Dataset](#dataset)
2. [Methodology](#methodology)
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
???

### Dataset
---
???

### Methodology
---
The following methodology was employed for data analysis of ???:
1. Data Pre-processing and Preparation 
    - Dataset Creation
    - Preliminary Exploratory Data Analysis
    - Feature Engineering
        - Outlier Management
        - Feature Creation
        - Feature Reduction
        - Feature Analysis
        - Feature Selection
        - Dataset Subsetting
2. Data Mining
    - Initial Data Mining
    -  Hyper-parameter Fine-tuning
3. Analysis
    - Model Performance Comparison and Analysis


### Repository Structure
---
```
requirements.txt

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

    For macOS users with Apple Silicon, you can choose to install additional packages provided by [Apple](https://developer.apple.com/metal/tensorflow-plugin/) to utilise your device's GPU for enhanced model training. The consolidated project dependencies for such users is provided in the `mac_requirements.txt` file:
    ```
    pip3 install -r mac_requirements.txt
    ```
    **Note: tensorflow-metal is currently only supported for Python 3.8, 3.9 and 3.10*
3. Run the interactive Python notebook to train/test the model, ensuring that you've linked the notebook to the correct Python `virtualenv`. 