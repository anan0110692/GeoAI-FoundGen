# A Sensor Agnostic Framework for Leveraging Geospatial Foundation Models: Enhancing Semantic Segmentation via Synergistic Pseudo-Labeling and Generative Learning
## Paper
![](./New_digram.png)

A Sensor Agnostic Framework for Leveraging Geospatial Foundation Models: Enhancing Semantic Segmentation via Synergistic Pseudo-Labeling and Generative Learning  
 [Anan Yaghmour](https://github.com/anan0110692), Melba M. Crawford, Saurabh Prasad



## Abstract
Remote sensing (RS) plays a pivotal role in addressing diverse environmental and societal challenges, with semantic segmentation enabling critical applications like poverty estimation, crop yield prediction, and pollution detection. Recent advances in satellite technology have dramatically expanded the number and quality of RS datasets, creating new opportunities for data-driven insights. However, achieving high-performance segmentation models requires extensive labeled data, a challenge exacerbated by the scarcity of annotated datasets and the inherent variability caused by a wide range of factors such as different sensor types, illumination conditions, and geographical scales. Domain adaptation offers a promising solution to improve model generalization across differing conditions. In this paper, we propose an approach to domain adaptation that combines soft-alignment pseudo-labeling with source-to-target generative pre-training. Although our approach is tailored for geospatial imagery, it can be beneficial to other applications as well. This method leverages the generalization capacity of foundational models, such as Prithvi, to create a flexible framework adaptable to various RS data modalities, including RGB, multispectral, and hyperspectral imaging. Additionally, we also provide a rigorous mathematical analysis of the impact  of MAE-based generative learning on segmentation tasks, aiming to improve domain-invariant feature learning. Experiments on geospatial datasets representing hyperspectral and multispectral imagery demonstrate the efficacy of our method in boosting model adaptability and performance across domains, confirming its potential as a robust framework for RS applications.

## Preparation

### Pre-requisites
* Python 3.9.18
* PyTorch Version: 2.5.1
* CUDA Version: 12.4
### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/anan0110692/GeoAI-FoundGen.git
$ cd GeoAI-FoundGen
```

1. Setting Up the Environment
```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```
3. Download Prithvi's checkpoint

   Download the `Prithvi_100M.pt` file from [this link](https://github.com/NASA-IMPACT/hls-foundation-os) and place it in the `<root_dict>/prithvi` directory.
### Datasets

Ensure the `Datafiles` folder exists. If it does not, create it using the following command (Linux):

```bash
mkdir -p <root_dir>/Datasets/Datafiles
```
Once the folder is created, all data files should be put  in:
```<root_dir>/Datasets/Datafiles```
* **Download Datafiles Folder**


The datafiles folder  can be downloaded from the following link: [Google Drive Link](https://drive.google.com/drive/folders/1JHUJHgNb8Sa23XLkAWHsIS7z838eYfRI?usp=sharing).

The datasets included are:

- **Flair Dataset** (refer to [Link](https://ignf.github.io/FLAIR/))
- **C2Sseg-AB Dataset** (refer to [Link](https://github.com/danfenghong))


For reproducibility, it is highly recommended to download the provided datafiles folder for consistent results.
## Running the Code

The code is structured as follows:

1. **Training Phase**: Implemented in the `Train` notebook.
2. **Inference Phase**: Implemented in the `Test` notebook.

The notebooks should be executed in the order listed above to ensure proper workflow and reproducibility.

### Train Notebook Guide

1. **Select the Dataset**  
   In the second cell of the notebook, choose the appropriate dataset file by specifying `Dataset.<Dataset file>`. For example:
   ```python
   Dataset.FLAIR
2. **Set the Experiment Name**  
   In the third cell (*interface cell* ), assign a value to `Exp_name`. By default, logs and snapshots will be stored in the following directory structure:   

```bash
<root_dir>/Results/<Dataset file>/Exp_<Exp_name>
<root_dir>/Results/<Dataset file>/Source/Exp_<Exp_name>
<root_dir>/Results/<Dataset file>/lightning_logs/Exp_<Exp_name>
```

  
- The `Source` folder contains the checkpoints .  
- The `Exp_<Exp_name>` folder contains a list of ready-to-use models.  
- The `lightning_logs` folder contains TensorBoard logs for tracking experiments and runs.
3. Run all cells

 
### Test Notebook Guide

1. **Select the Dataset**  
   In the second cell of the notebook, choose the appropriate dataset file by specifying `Dataset.<Dataset file>`. For example:
   ```python
   Dataset.FLAIR
2. **Set Adapted Models Path**
   Typically, this path should be provided after running the Train notebook. For example, for an experiment with the name `dummy_DA`, the path would look like:

```bash
<root_dir>/Results/FLAIR/dummy_DA/FLAIR_lists.pkl
```
   
 
3. Run the desired evaluation cell.

### Pre-trained parameters
Pre-trained parameters can be downloaded from [C2Seg-AV](https://sharedby.blomp.com/gri25h), [FLAIR](https://sharedby.blomp.com/rKtw6x). Each dataset includes one `.pkl` file contains a list of pre-trained models  , with a length equal to the number of runs (10). To start the download please, enter the following email address github.gatafiles@gmail.com

## Acknowledgements

This code utilizes and modifies the pre-trained foundation model provided by [NASA-IMPACT](https://github.com/NASA-IMPACT/hls-foundation-os).

