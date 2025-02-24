import mrcfile
import numpy as np
from scipy.ndimage import zoom

# 生のマップを読み込む
with mrcfile.open('raw_map.mrc', permissive=True) as mrc:
    raw_map = mrc.data

# 座標系の変換（必要に応じて）
# これはmrcファイルのヘッダー情報に基づいて行います
# （具体的な変換コードはデータによって異なります）
transposed_map = np.transpose(raw_map, axes=(desired_axes))

# 現在のボクセルサイズを取得
current_voxel_size = mrc.voxel_size.x  # x, y, zが同じと仮定

# 目標のボクセルサイズ（1Å）にリサイズ
scale_factor = current_voxel_size / 1.0
normalized_map = zoom(transposed_map, zoom=scale_factor, order=1)

M0 = normalized_map
M0_med = np.median(M0)
M0_top1 = np.percentile(M0, 99)

N = np.zeros_like(M0)

N[M0 < M0_med] = 0
mask1 = (M0 >= M0_med) & (M0 < M0_top1)
N[mask1] = (M0[mask1] - M0_med) / (M0_top1 - M0_med)
N[M0 >= M0_top1] = 1

# Nが正規化されたマップ
normalized_voxel_data = N

from biopandas.pdb import PandasPdb

ppdb = PandasPdb().read_pdb('structure.pdb')
atom_df = ppdb.df['ATOM']

# 原子座標を取得
atom_coords = atom_df[['x_coord', 'y_coord', 'z_coord']].values
atom_types = atom_df['atom_name'].values
residue_names = atom_df['residue_name'].values

# ボクセル空間の構築
voxel_size = 1.0  # Å
grid_shape = normalized_voxel_data.shape
label_backbone = np.zeros(grid_shape, dtype=np.int8)
label_calpha = np.zeros(grid_shape, dtype=np.int8)
label_aa = np.zeros(grid_shape, dtype=np.int8)

# 原子座標をボクセルインデックスに変換
voxel_indices = (atom_coords / voxel_size).astype(int)

# バックボーンラベルの付与
for idx, atom_type in zip(voxel_indices, atom_types):
    x, y, z = idx
    if atom_type in ['N', 'CA', 'C', 'O']:  # 主鎖原子
        label_backbone[x, y, z] = 1  # メインチェインボクセル
    else:
        label_backbone[x, y, z] = 2  # サイドチェインボクセル

# マスクボクセルの設定（主鎖・側鎖原子の近傍）
from scipy.ndimage import binary_dilation

structure_mask = label_backbone > 0
dilated_mask = binary_dilation(structure_mask, iterations=1)
mask_voxels = dilated_mask & (~structure_mask)
label_backbone[mask_voxels] = 3  # マスクボクセル

# label_backboneが0のままのボクセルは非構造ボクセル

# 同様にCαラベルとアミノ酸タイプラベルを設定
# Cα予測用ラベル
for idx, atom_type in zip(voxel_indices, atom_types):
    x, y, z = idx
    if atom_type == 'CA':
        label_calpha[x, y, z] = 1  # Cαボクセル
    else:
        label_calpha[x, y, z] = 2  # その他の原子ボクセル

# マスクボクセルの設定
structure_mask = label_calpha > 0
dilated_mask = binary_dilation(structure_mask, iterations=1)
mask_voxels = dilated_mask & (~structure_mask)
label_calpha[mask_voxels] = 3  # マスクボクセル

# アミノ酸タイプ予測用ラベル
aa_mapping = {'ALA':1, 'ARG':2, 'ASN':3, 'ASP':4, 'CYS':5, 'GLN':6, 'GLU':7, 'GLY':8, 'HIS':9,
              'ILE':10, 'LEU':11, 'LYS':12, 'MET':13, 'PHE':14, 'PRO':15, 'SER':16, 'THR':17,
              'TRP':18, 'TYR':19, 'VAL':20}

for idx, res_name in zip(voxel_indices, residue_names):
    x, y, z = idx
    aa_type = aa_mapping.get(res_name.strip(), 0)
    if aa_type > 0:
        label_aa[x, y, z] = aa_type  # アミノ酸タイプ


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.elu(out)
        return out

class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[32, 64, 128, 256]):
        super(ResidualUNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        for idx, f in enumerate(filters):
            if idx == 0:
                self.encoders.append(ResidualBlock(in_channels, f))
            else:
                self.encoders.append(ResidualBlock(filters[idx-1], f))
        
        # Decoder path
        for idx in range(len(filters)-1, 0, -1):
            self.decoders.append(nn.ConvTranspose3d(filters[idx], filters[idx-1], kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock(filters[idx], filters[idx-1]))
        
        self.final_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_outs.append(x)
            x = F.max_pool3d(x, kernel_size=2)
        
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip_connection = enc_outs[-(idx//2)-2]
            x = torch.cat((x, skip_connection), dim=1)
            x = self.decoders[idx+1](x)
        
        out = self.final_conv(x)
        return out

# クラスウェイトの設定
backbone_weights = torch.tensor([1.0, 0.3, 0.03, 0.0])  # main chain, side chain, non-structural, masked
calpha_weights = torch.tensor([1.0, 0.1, 0.01, 0.0])  # Cα, other-atom, non-structural, masked
aa_weights = torch.ones(21)  # アミノ酸タイプ間のウェイトは均一

# 損失関数の定義
criterion_backbone = nn.CrossEntropyLoss(weight=backbone_weights, ignore_index=3)  # マスクボクセルを無視
criterion_calpha = nn.CrossEntropyLoss(weight=calpha_weights, ignore_index=3)
criterion_aa = nn.CrossEntropyLoss(weight=aa_weights, ignore_index=0)  # マスクボクセルを無視
from torch.utils.data import Dataset, DataLoader

class CryoEMDataset(Dataset):
    def __init__(self, em_data, backbone_labels, calpha_labels, aa_labels):
        self.em_data = em_data  # 正規化されたマップ
        self.backbone_labels = backbone_labels
        self.calpha_labels = calpha_labels
        self.aa_labels = aa_labels
    
    def __len__(self):
        return len(self.em_data)  # サブボリュームの数
    
    def __getitem__(self, idx):
        # 必要に応じてデータ拡張やスライドウィンドウでサブボリュームを取得
        em_volume = self.em_data[idx]
        bb_label = self.backbone_labels[idx]
        ca_label = self.calpha_labels[idx]
        aa_label = self.aa_labels[idx]
        return em_volume, bb_label, ca_label, aa_label

# データセットとデータローダーの作成
dataset = CryoEMDataset(em_data, backbone_labels, calpha_labels, aa_labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

import torch.optim as optim

optimizer_bb = optim.Adam(backbone_model.parameters(), lr=1e-4)
optimizer_ca = optim.Adam(calpha_model.parameters(), lr=1e-4)
optimizer_aa = optim.Adam(aa_model.parameters(), lr=1e-4)

num_epochs = 100
lambda_S = 1.0
lambda_C = 1.0
lambda_A = 0.0  # ウォームアップ時は0から開始

for epoch in range(num_epochs):
    for em_volume, bb_label, ca_label, aa_label in dataloader:
        # デバイスに転送
        em_volume = em_volume.to(device)
        bb_label = bb_label.to(device)
        ca_label = ca_label.to(device)
        aa_label = aa_label.to(device)
        
        # バックボーン予測
        optimizer_bb.zero_grad()
        bb_output = backbone_model(em_volume)
        loss_bb = criterion_backbone(bb_output, bb_label)
        loss_bb.backward()
        optimizer_bb.step()
        
        # Cα予測
        optimizer_ca.zero_grad()
        # NとN_Bを結合
        em_input_ca = torch.cat((em_volume, bb_output), dim=1)
        ca_output = calpha_model(em_input_ca)
        loss_ca = criterion_calpha(ca_output, ca_label)
        loss_ca.backward()
        optimizer_ca.step()
        
        # アミノ酸タイプ予測
        optimizer_aa.zero_grad()
        em_input_aa = torch.cat((em_volume, bb_output), dim=1)
        aa_output = aa_model(em_input_aa)
        loss_aa = criterion_aa(aa_output, aa_label)
        loss_aa.backward()
        optimizer_aa.step()
        
        # 総損失（必要に応じて）
        total_loss = lambda_S * loss_bb + lambda_C * loss_ca + lambda_A * loss_aa
        
    # エポックごとにλを調整（ウォームアップ戦略）
    if epoch < warmup_epochs:
        lambda_A = 0.0
    else:
        lambda_S = 0.0
        lambda_C = 0.0
        lambda_A = 1.0
