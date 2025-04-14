from dataset import TestbedDataset
from torch_geometric.loader import DataLoader
from torch import nn, optim
from model import MMPDIM_DTA
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import metrics
import torch.nn.functional as F
from sklearn.model_selection import KFold
def process_sph(data,sph_dic):
    sph_list = []
    for name in data.pdb_name:
        sph_list.append(sph_dic[name])
    max_size = max(tensor.shape[0] for tensor in sph_list)
    padded_tensors = []
    for tensor in sph_list:
        pad = (0, max_size - tensor.shape[1], 0, max_size - tensor.shape[0])
        padded_tensor = F.pad(tensor, pad, value=510)
        padded_tensors.append(padded_tensor)
    sph_tensor = torch.stack(padded_tensors)
    data.sph = sph_tensor.to('cuda')
def train(model, device, train_loader, optimizer,loss_fn,val_sph_dic,scheduler):

    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = data.y
        process_sph(data,val_sph_dic)
        output = model(data)
        loss = loss_fn(output.view(-1, 1), label.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()


def test(model: nn.Module, test_loader, loss_function, device,val_sph_dic):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            label = data.y
            process_sph(data, val_sph_dic)
            output = model(data)
            test_loss += loss_function(output.view(-1), label.view(-1)).item()
            outputs.append(output.cpu().numpy().reshape(-1))
            targets.append(label.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': round(metrics.c_index(targets, outputs), 4),
        'RMSE': round(metrics.RMSE(targets, outputs), 4),
        'MSE': round(metrics.mse(targets, outputs), 4),
        'MAE': round(metrics.MAE(targets, outputs), 4),
        'rm2': round(metrics.rm2(targets, outputs), 4),
        'SD': round(metrics.SD(targets, outputs), 4),
        'Pcc': round(metrics.CORR(targets, outputs), 4),
        'AUC': round(metrics.calculate_auc(targets, outputs), 4)
    }

    return evaluation

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed=42
setup_seed(seed)
device = torch.device("cuda")
file_path = '/home/zhc/pycharm/Gradformer_DTA_new/MMPDIM-DTA/data'
sph_dic      = torch.load(file_path + '/processed/sph_zong.pt')
data_set = TestbedDataset(root=file_path, dataset='train_and_val')
test2016_dataloader = DataLoader(TestbedDataset(root=file_path, dataset='test2016'),batch_size=128 ,shuffle=False,follow_batch=['ESM'])
test2013_dataloader = DataLoader(TestbedDataset(root=file_path, dataset='test2013'),batch_size=128 ,shuffle=False,follow_batch=['ESM'])
csar_dataloader =     DataLoader(TestbedDataset(root=file_path, dataset='CSAR'),batch_size=128 ,shuffle=False,follow_batch=['ESM'])
path = Path(f'result/MMPDIM_DTA_{datetime.now()}_{seed}')
writer = SummaryWriter(path)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kfold.split(data_set)):

    model = MMPDIM_DTA().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss(reduction='sum')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    train_loader = DataLoader(torch.utils.data.dataset.Subset(data_set, train_index), batch_size=128,shuffle=True,follow_batch=['ESM'])
    val_loader = DataLoader(torch.utils.data.dataset.Subset(data_set, val_index), batch_size=128, shuffle=False,follow_batch=['ESM'])
    best_loss = 100

    for epoch in range(1, 51):
        train_performance = train(model, device, train_loader, optimizer, loss_fn, sph_dic,scheduler)
        val_performance =  test(model, val_loader,loss_fn,device ,sph_dic)
        if epoch >= 1 and val_performance['loss'] < best_loss:
            best_loss = val_performance['loss']
            best_epoch = epoch
            print(fold + 1, "epoch=", epoch, val_performance)
            torch.save(model.state_dict(), path / f'best_model_{fold + 1}.pt')



model = MMPDIM_DTA().to(device)
loss_fn = nn.MSELoss(reduction='sum')
print()
with open(path / 'result.txt', 'w') as f:
    f.write('CI,RMSE,R2,MSE,MAE,SD,Pcc,AUC\n')
    f.write('test2016\n')
    ci_list,rmse_list,MAE_list,rm2_list,mse_list,SD_list,Pcc_list,AUC_list = [],[],[],[],[],[],[],[]
    for fold in range(10):
        model.load_state_dict(torch.load(path / f'best_model_{fold+1}.pt'))
        test2016_performance = test(model, test2016_dataloader, loss_fn, device, sph_dic)
        ci   = str(test2016_performance['c_index'])
        rmse = str(test2016_performance['RMSE'])
        rm2  = str(test2016_performance['rm2'])
        mse  = str(test2016_performance['MSE'])
        MAE = str(test2016_performance['MAE'])
        SD = str(test2016_performance['SD'])
        Pcc = str(test2016_performance['Pcc'])
        AUC = str(test2016_performance['AUC'])
        ci_list.append(float(ci))
        rmse_list.append(float(rmse))
        rm2_list.append(float(rm2))
        mse_list.append(float(mse))
        MAE_list.append(float(MAE))
        SD_list.append(float(SD))
        Pcc_list.append(float(Pcc))
        AUC_list.append(float(AUC))

        f.write(ci+','+rmse+','+rm2+','+mse+','+MAE+','+ SD+','+Pcc+','+AUC+ '\n')

    ci,ci_std = str(np.mean(ci_list)),str(np.std(ci_list))
    rmse,rmse_std = str(np.mean(rmse_list)),str(np.std(rmse_list))
    rm2, rm2_std = str(np.mean(rm2_list)), str(np.std(rm2_list))
    mse, mse_std = str(np.mean(mse_list)), str(np.std(mse_list))
    MAE, MAE_std = str(np.mean(MAE_list)), str(np.std(MAE_list))
    SD, SD_std = str(np.mean(SD_list)), str(np.std(SD_list))
    Pcc, Pcc_std = str(np.mean(Pcc_list)), str(np.std(Pcc_list))
    AUC, AUC_std = str(np.mean(AUC_list)), str(np.std(AUC_list))

    f.write('\nmean\n')
    f.write(ci+','+rmse+','+rm2+','+mse+'\n'+ci_std+','+rmse_std+','+rm2_std+','+mse_std+','+SD_std+','+Pcc_std+','+AUC_std)
    f.write(
        ci+','+rmse+','+rm2+','+mse+','+MAE+','+SD+','+Pcc+','+AUC+'\n'+
        ci_std+','+rmse_std + ',' + rm2_std + ',' + mse_std + ','+ MAE_std + ',' + SD_std + ',' + Pcc_std + ',' + AUC_std
    )
    f.write('CI,RMSE,R2,MSE,MAE,SD,Pcc,AUC\n')
    f.write('test2013\n')
    ci_list, rmse_list, MAE_list, rm2_list, mse_list, SD_list, Pcc_list, AUC_list = [], [], [], [], [], [], [], []
    for fold in range(10):
        model.load_state_dict(torch.load(path / f'best_model_{fold + 1}.pt'))
        test2013_performance = test(model, test2013_dataloader, loss_fn, device, sph_dic)

        ci = str(test2013_performance['c_index'])
        rmse = str(test2013_performance['RMSE'])
        rm2 = str(test2013_performance['rm2'])
        mse = str(test2013_performance['MSE'])
        MAE = str(test2013_performance['MAE'])
        SD = str(test2013_performance['SD'])
        Pcc = str(test2013_performance['Pcc'])
        AUC = str(test2013_performance['AUC'])
        ci_list.append(float(ci))
        rmse_list.append(float(rmse))
        rm2_list.append(float(rm2))
        mse_list.append(float(mse))
        MAE_list.append(float(MAE))
        SD_list.append(float(SD))
        Pcc_list.append(float(Pcc))
        AUC_list.append(float(AUC))

        f.write(ci + ',' + rmse + ',' + rm2 + ',' + mse + ',' + MAE + ',' + SD + ',' + Pcc + ',' + AUC + '\n')

    ci, ci_std = str(np.mean(ci_list)), str(np.std(ci_list))
    rmse, rmse_std = str(np.mean(rmse_list)), str(np.std(rmse_list))
    rm2, rm2_std = str(np.mean(rm2_list)), str(np.std(rm2_list))
    mse, mse_std = str(np.mean(mse_list)), str(np.std(mse_list))
    MAE, MAE_std = str(np.mean(MAE_list)), str(np.std(MAE_list))
    SD, SD_std = str(np.mean(SD_list)), str(np.std(SD_list))
    Pcc, Pcc_std = str(np.mean(Pcc_list)), str(np.std(Pcc_list))
    AUC, AUC_std = str(np.mean(AUC_list)), str(np.std(AUC_list))

    f.write('\nmean\n')
    f.write(
        ci + ',' + rmse + ',' + rm2 + ',' + mse + '\n' + ci_std + ',' + rmse_std + ',' + rm2_std + ',' + mse_std + ',' + SD_std + ',' + Pcc_std + ',' + AUC_std)
    f.write(
        ci + ',' + rmse + ',' + rm2 + ',' + mse + ',' + MAE + ',' + SD + ',' + Pcc + ',' + AUC + '\n' +
        ci_std + ',' + rmse_std + ',' + rm2_std + ',' + mse_std + ',' + MAE_std + ',' + SD_std + ',' + Pcc_std + ',' + AUC_std
    )
    f.write('CI,RMSE,R2,MSE,MAE,SD,Pcc,AUC\n')
    f.write('CSAR\n')
    ci_list, rmse_list, MAE_list, rm2_list, mse_list, SD_list, Pcc_list, AUC_list = [], [], [], [], [], [], [], []
    for fold in range(10):
        model.load_state_dict(torch.load(path / f'best_model_{fold + 1}.pt'))
        CSAR_performance = test(model, csar_dataloader, loss_fn, device, sph_dic)

        ci = str(CSAR_performance['c_index'])
        rmse = str(CSAR_performance['RMSE'])
        rm2 = str(CSAR_performance['rm2'])
        mse = str(CSAR_performance['MSE'])
        MAE = str(CSAR_performance['MAE'])
        SD = str(CSAR_performance['SD'])
        Pcc = str(CSAR_performance['Pcc'])
        AUC = str(CSAR_performance['AUC'])
        ci_list.append(float(ci))
        rmse_list.append(float(rmse))
        rm2_list.append(float(rm2))
        mse_list.append(float(mse))
        MAE_list.append(float(MAE))
        SD_list.append(float(SD))
        Pcc_list.append(float(Pcc))
        AUC_list.append(float(AUC))

        f.write(ci + ',' + rmse + ',' + rm2 + ',' + mse + ',' + MAE + ',' + SD + ',' + Pcc + ',' + AUC + '\n')

    ci, ci_std = str(np.mean(ci_list)), str(np.std(ci_list))
    rmse, rmse_std = str(np.mean(rmse_list)), str(np.std(rmse_list))
    rm2, rm2_std = str(np.mean(rm2_list)), str(np.std(rm2_list))
    mse, mse_std = str(np.mean(mse_list)), str(np.std(mse_list))
    MAE, MAE_std = str(np.mean(MAE_list)), str(np.std(MAE_list))
    SD, SD_std = str(np.mean(SD_list)), str(np.std(SD_list))
    Pcc, Pcc_std = str(np.mean(Pcc_list)), str(np.std(Pcc_list))
    AUC, AUC_std = str(np.mean(AUC_list)), str(np.std(AUC_list))

    f.write('\nmean\n')
    f.write(
        ci + ',' + rmse + ',' + rm2 + ',' + mse + '\n' + ci_std + ',' + rmse_std + ',' + rm2_std + ',' + mse_std + ',' + SD_std + ',' + Pcc_std + ',' + AUC_std)
    f.write(
        ci + ',' + rmse + ',' + rm2 + ',' + mse + ',' + MAE + ',' + SD + ',' + Pcc + ',' + AUC + '\n' +
        ci_std + ',' + rmse_std + ',' + rm2_std + ',' + mse_std + ',' + MAE_std + ',' + SD_std + ',' + Pcc_std + ',' + AUC_std
    )
