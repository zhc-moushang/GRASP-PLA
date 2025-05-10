# GRASP-DTA

  <div align="center">
  <img src="https://raw.githubusercontent.com/zhc-moushang/MMPDIM-DTA/main/Fig/model.jpg" width="600">
  </div>
  
## 1. Conda Environment
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
The `.whl` files for offline installation of `torch`, `torch_cluster`, `torch_scatter`, and `torch_sparse` are available via:

- [**Google Drive 📁**](https://drive.google.com/drive/folders/1SyVzxgTGPr9dtBRbexzlLA5PMmUuJKPl?usp=sharing)
- [**PyTorch Geometric WHL page**](https://pytorch-geometric.com/whl/)
## 2. Dataset

The datasets required to replicate our experiments can be downloaded from:

- [**Google Drive 📁**](https://drive.google.com/drive/folders/1SyVzxgTGPr9dtBRbexzlLA5PMmUuJKPl?usp=sharing)

**Included files:**

- `test2016.pt`
- `test2013.pt`
- `CSAR.pt`
- `sph_zong.pt`
- `affinity_data.csv`
- `test2016.tar.xz` — contains protein `.pdb` files and drug `.mol2` files for the **test2016** dataset
- `test2013.tar.xz` — contains protein `.pdb` files and drug `.mol2` files for the **test2013** dataset
## 3. Train and Test

We provide the following scripts to train and evaluate the model:

- [`train_kFold.py`](train_kFold.py): Training with **k-fold cross-validation**
- [`test_kFold.py`](test_kFold.py): Testing using **trained models**

## Acknowledgments

We appreciate the open-source contributions of the following projects:

- [**LGI-GT**](https://github.com/shuoyinn/LGI-GT)
- [**Gradformer**](https://github.com/LiuChuang0059/Gradformer)
- [**AttentionMGT-DTA**](https://github.com/JK-Liu7/AttentionMGT-DTA)
- [**ESM (Facebook Research)**](https://github.com/facebookresearch/esm)
## Citations
```
```
