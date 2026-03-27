import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from torchcrf import CRF
import warnings
from collections import Counter
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ================= 🚀 K-Fold SOTA 配置 (深度感知版) =================
DATA_DIR = "newtrain"  
OUTPUT_DIR = "checkpoints_kfold_depth15" # 新目录
MODEL_NAME = "TransResUNet_SE_Depth"

# K-Fold 设置
NUM_FOLDS = 5

# 模型参数
# 🌟 核心修改：输入维度改为 9 (原始4 + 梯度4 + 深度1)
INPUT_DIM = 9           
HIDDEN_DIM = 128        
NUM_LAYERS = 3          
TRANS_LAYERS = 4        
NHEAD = 4               
DROPOUT = 0.2           

# 训练参数
BATCH_SIZE = 64         
NUM_EPOCHS = 60         
LR_BASE = 2e-4          
WEIGHT_DECAY = 1e-3     

# 损失权重
LAMBDA_BOUNDARY = 10.0  
LAMBDA_DICE = 1.0       

# 数据参数
MAX_SEQ_LEN = 1024      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COL = 'LAYER'

FEATURE_CHANNELS = {
    'GR': ['GR', 'GR_QYZ', 'NG'], 'AC': ['AC', 'DT'], 
    'DEN': ['DEN', 'ZDEN'], 'RT': ['RLLD', 'ILD', 'R400', 'LLD']
}
ORDERED_CHANNELS = ['GR', 'AC', 'DEN', 'RT']

# ================= 🔧 核心组件 =================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            SEBlock(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x): return self.conv(x) + self.shortcut(x)

# ================= 🔧 损失函数 =================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super(DiceLoss, self).__init__(); self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1); targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction; self.pos_weight=pos_weight
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce); focal = self.alpha * (1-pt)**self.gamma * bce
        return focal.mean() if self.reduction == 'mean' else focal

# ================= 📦 数据加载 (深度感知版) =================

class StratigraphyAugmenter:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, x):
        if np.random.rand() > self.p: return x
        # 仅对前8个通道（物理曲线）做增强，不干扰第9个通道（深度）
        feat = x[:, :8]
        depth = x[:, 8:]
        
        scale = np.random.uniform(0.9, 1.1)
        feat = feat * scale
        
        shift = np.random.normal(0, 0.05, size=(1, feat.shape[1]))
        feat = feat + shift
        
        noise = np.random.normal(0, 0.02, size=feat.shape)
        feat = feat + noise
        
        return np.concatenate([feat, depth], axis=1)

class SOTADataset(Dataset):
    def __init__(self, file_list, window_size, stride, augment=False, label_encoder=None):
        self.samples = []; self.sample_weights = [] 
        self.augmenter = StratigraphyAugmenter(p=0.5) if augment else None
        
        all_labels_flat = []
        data_buffer = []
        
        print(f"🔄 预加载数据 (Augment={augment})...")
        for filepath in tqdm(file_list):
            try: df = pd.read_csv(filepath, engine='c')
            except: 
                try: df = pd.read_csv(filepath, delim_whitespace=True, engine='c')
                except: continue
            if LABEL_COL not in df.columns: continue
            df = df.dropna(subset=[LABEL_COL]); df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip(); df = df[df[LABEL_COL] != '']
            if len(df) < window_size: continue
            
            valid_feats = []
            for channel_name in ORDERED_CHANNELS:
                candidates = FEATURE_CHANNELS.get(channel_name, [])
                found = False
                for cand in candidates:
                    if cand in df.columns: series = pd.to_numeric(df[cand], errors='coerce').fillna(0); valid_feats.append(series.values); found = True; break
                if not found: valid_feats.append(np.zeros(len(df)))
            
            base_data = np.stack(valid_feats, axis=1)
            grad_data = np.gradient(base_data, axis=0)
            
            # 🌟 核心修改：构建相对深度特征
            if 'DEPTH' in df.columns:
                d_vals = df['DEPTH'].values
                # 归一化到 0-1
                d_norm = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-6)
            else:
                d_norm = np.linspace(0, 1, len(df))
            d_norm = d_norm[:, np.newaxis] # (Length, 1)

            # 拼接：4原始 + 4梯度 + 1深度 = 9通道
            x_data = np.concatenate([base_data, grad_data, d_norm], axis=1)
            
            # 归一化策略：只对前8列做Z-Score，保留深度列的线性关系
            feats = x_data[:, :8]
            x_data[:, :8] = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
            
            labels = df[LABEL_COL].values; all_labels_flat.extend(labels); data_buffer.append((x_data, labels))
            
        if not all_labels_flat: return

        if label_encoder is None: self.label_encoder = LabelEncoder(); self.label_encoder.fit(all_labels_flat)
        else: self.label_encoder = label_encoder
        self.num_classes = len(self.label_encoder.classes_)
        
        counter = Counter(all_labels_flat); total = len(all_labels_flat)
        self.class_weights_map = {k: min(total/(v+1), 50.0) for k, v in counter.items()}
        
        for x_data, labels in data_buffer:
            try: y_data = self.label_encoder.transform(labels)
            except: continue
            
            is_boundary = (y_data[1:] != y_data[:-1]).astype(float); boundary_target = np.concatenate(([0], is_boundary)); boundary_target = gaussian_filter1d(boundary_target, sigma=1.5)
            if boundary_target.max() > 0: boundary_target /= boundary_target.max()
            
            n_windows = (len(x_data) - window_size) // stride + 1
            if n_windows <= 0: continue
            for i in range(n_windows):
                start = i * stride; end = start + window_size; seq_labels = labels[start:end]; weight = 1.0
                if np.sum(y_data[start:end-1] != y_data[start+1:end]) > 0: weight = 5.0
                self.samples.append({'x': x_data[start:end], 'y': y_data[start:end], 'b': boundary_target[start:end]})
                self.sample_weights.append(weight)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]; x = item['x'].copy()
        if self.augmenter: x = self.augmenter(x)
        return (torch.FloatTensor(x), torch.LongTensor(item['y']), torch.FloatTensor(item['b']))

def collate_fn(batch):
    inputs, targets, b_targets = zip(*batch); inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=0); targets_pad = pad_sequence(targets, batch_first=True, padding_value=0); b_targets_pad = pad_sequence(b_targets, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in inputs]); mask = torch.arange(inputs_pad.size(1))[None, :] < lengths[:, None]
    return inputs_pad, targets_pad, b_targets_pad, mask

# ================= 🧠 TransResUNet 1D 模型 =================

class TransResUNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nhead=8, num_layers=3):
        super(TransResUNet1D, self).__init__()
        self.inc = ResDoubleConv(input_dim, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), ResDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), ResDoubleConv(128, 256))
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2048, 256))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=nhead, dim_feedforward=512, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ResDoubleConv(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ResDoubleConv(128, 64)
        self.hidden2tag = nn.Linear(64, num_classes)
        self.boundary_head = nn.Linear(64, 1)
        self.crf = CRF(num_classes, batch_first=True)
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=torch.tensor([10.0]).to(DEVICE))
        self.dice_loss = DiceLoss()

    def forward(self, x, mask=None):
        x_cnn = x.permute(0, 2, 1)
        x1 = self.inc(x_cnn); x2 = self.down1(x1); x3 = self.down2(x2)
        tr_in = x3.permute(0, 2, 1); seq_len = tr_in.size(1); tr_in = tr_in + self.pos_encoder[:, :seq_len, :]
        tr_out = self.transformer(tr_in); x_bot = tr_out.permute(0, 2, 1)
        x_up2 = self.up2(x_bot)
        if x_up2.size(2) != x2.size(2): x_up2 = F.interpolate(x_up2, size=x2.size(2))
        x_dec2 = self.conv2(torch.cat([x2, x_up2], dim=1)) 
        x_up1 = self.up1(x_dec2)
        if x_up1.size(2) != x1.size(2): x_up1 = F.interpolate(x_up1, size=x1.size(2))
        x_dec1 = torch.cat([x1, x_up1], dim=1)
        x_out = self.conv1(x_dec1)
        final_feat = x_out.permute(0, 2, 1)
        emissions = self.hidden2tag(final_feat)
        boundary_logits = self.boundary_head(final_feat).squeeze(-1)
        return emissions, boundary_logits

    def compute_loss(self, x, tags, boundary_targets, mask):
        emissions, boundary_logits = self.forward(x, mask)
        nll_loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        if mask is not None:
            active_logits = boundary_logits.masked_select(mask)
            active_targets = boundary_targets.masked_select(mask)
            f_loss = self.focal_loss(active_logits, active_targets)
            d_loss = self.dice_loss(active_logits, active_targets)
        else:
            f_loss = self.focal_loss(boundary_logits, boundary_targets)
            d_loss = self.dice_loss(boundary_logits, boundary_targets)
        b_loss = f_loss + d_loss
        return nll_loss, b_loss, emissions

    def decode(self, x, mask):
        emissions, _ = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)

# ================= 🚀 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not all_files: return
    all_files = np.array(all_files)
    
    print("🔄 扫描文件...")
    all_labels = []
    for f in tqdm(all_files):
        try: df = pd.read_csv(f)
        except: 
            try: df = pd.read_csv(f, delim_whitespace=True)
            except: continue
        if LABEL_COL in df.columns:
            df = df.dropna(subset=[LABEL_COL]); all_labels.extend(df[LABEL_COL].astype(str).str.strip().values)
    
    global_le = LabelEncoder(); global_le.fit(all_labels)
    num_classes = len(global_le.classes_)
    print(f"✅ 全局类别数: {num_classes}")
    np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), global_le.classes_)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(f"\n[START] 🚀 TransResUNet-Depth 训练 (9 Channel Input)")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n{'='*20} Fold {fold+1}/{NUM_FOLDS} {'='*20}")
        
        train_files_fold = all_files[train_idx]
        val_files_fold = all_files[val_idx]
        
        train_ds = SOTADataset(train_files_fold, MAX_SEQ_LEN, stride=MAX_SEQ_LEN//2, augment=True, label_encoder=global_le)
        val_ds = SOTADataset(val_files_fold, MAX_SEQ_LEN, stride=MAX_SEQ_LEN, augment=False, label_encoder=global_le)
        
        if len(train_ds) == 0: continue

        sampler = WeightedRandomSampler(train_ds.sample_weights, len(train_ds.sample_weights))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model = TransResUNet1D(INPUT_DIM, HIDDEN_DIM, num_classes, nhead=NHEAD, num_layers=TRANS_LAYERS).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = GradScaler()
        
        best_fold_f1 = -1
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False)
            
            for inputs, targets, b_targets, mask in pbar:
                inputs, targets, b_targets, mask = inputs.to(DEVICE), targets.to(DEVICE), b_targets.to(DEVICE), mask.to(DEVICE)
                
                optimizer.zero_grad()
                with autocast():
                    crf_loss, b_loss, _ = model.compute_loss(inputs, targets, b_targets, mask)
                    loss = crf_loss + LAMBDA_BOUNDARY * b_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.2f}"})
            
            scheduler.step()
            
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets, b_targets, mask in val_loader:
                    inputs, targets, b_targets, mask = inputs.to(DEVICE), targets.to(DEVICE), b_targets.to(DEVICE), mask.to(DEVICE)
                    with autocast(): _, _, emissions = model.compute_loss(inputs, targets, b_targets, mask)
                    preds = model.crf.decode(emissions, mask=mask)
                    for i, p in enumerate(preds):
                        valid_len = int(mask[i].sum())
                        all_preds.extend(p[:valid_len]); all_targets.extend(targets[i, :valid_len].cpu().tolist())
                        
            val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_fold_{fold}.pth"))
        
        print(f"✅ Fold {fold+1} Best F1: {best_fold_f1:.4f}")

if __name__ == "__main__":
    main()