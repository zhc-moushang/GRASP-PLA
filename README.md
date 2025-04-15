# MMPDIM-DTA
## 1. Conda environment
We provide commands for creating conda environments so you can replicate our work:
```
conda create -n MMPDIM_DTA python=3.8
conda activate MMPDIM_DTA
pip install torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
or
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.11-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.1.0
pip install cython==3.0.11
pip install atom3d==0.2.6
pip install rdkit==2023.3.3
pip install dgl==1.1.0
pip install dgllife==0.3.2
pip install numpy==1.23.5
pip install numba==0.58.1
pip install tensorboard==2.14.0
pip install setuptools==58.0.0
```
The .whl files required for the offline installation of torch, torch_cluster, torch_scatter, and torch_sparse are provided via [Google Drive](https://drive.google.com/drive/folders/1SyVzxgTGPr9dtBRbexzlLA5PMmUuJKPl?usp=sharing) or [pyg](https://pytorch-geometric.com/whl/).
## 2. Dataset
The data set needed to replicate the implementation in the paper is available from [Google Drive](https://drive.google.com/drive/folders/1SyVzxgTGPr9dtBRbexzlLA5PMmUuJKPl?usp=sharing). (test2016.pt, test2013.pt, CSAR.pt)
## 3. Train and Test
We provide scripts for [train_kFold.py](train_kFold.py) and [test_kFold.py](test_kFold.py).

## Acknowledegments
We appreciate [LGI-GT](https://github.com/shuoyinn/LGI-GT), [Gradformer](https://github.com/LiuChuang0059/Gradformer), [AttentionMGT-DTA](https://github.com/JK-Liu7/AttentionMGT-DTA), [esm](https://github.com/facebookresearch/esm) and other related works for their open-sourced contributions.
## Citations
