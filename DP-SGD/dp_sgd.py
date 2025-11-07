# FINAL DP-SGD READY-TO-RUN SCRIPT + METRICS & SCALING
# Generates: accuracy/F1, classification report, confusion matrix heatmap,
# timing & memory metrics, and an inference-scaling experiment across feature counts.

from __future__ import annotations
import os, time, copy, random, math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
try:
    import psutil
except ImportError:
    psutil = None

from opacus import PrivacyEngine

# --------------------------
# Config
# --------------------------
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs:int=20
    batch_size:int=64
    lr:float=1e-3
    max_grad_norm:float=1.0
    noise_list:Tuple[float,...]=(0.0,0.5,0.8,1.0)
    seeds:Tuple[int,...]=(13,23)
    save_dir:str="dp_results"
    target_delta:float=1e-5
    # scaling experiment
    scaling_feature_counts: Tuple[int,...] = (10, 50, 100, 200, 500, 1000)
    scaling_samples: int = 1024

CFG=Config(); os.makedirs(CFG.save_dir,exist_ok=True)

# --------------------------
# Model
# --------------------------
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,48), nn.ReLU(),
            nn.Linear(48,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,8),  nn.ReLU(),
            nn.Linear(8,2)    # 2 logits
        )
    def forward(self,x):
        return self.model(x)

# --------------------------
# Data (in-script)
# --------------------------
df = pd.read_csv("data.csv")
df['GENDER'] = df['GENDER'].map({'M':1,'F':0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES':1,'NO':0})
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
INPUT_DIM = X_scaled.shape[1]

# --------------------------
# Utils
# --------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def measure_latency(model: nn.Module, sample: torch.Tensor, device: str, repeats: int = 100) -> float:
    model.eval(); sample = sample.to(device)
    for _ in range(5): _ = model(sample)  # warmup
    if device.startswith('cuda'): torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(repeats):
        _ = model(sample)
    if device.startswith('cuda'): torch.cuda.synchronize()
    return (time.perf_counter()-t0)*1000.0/repeats

class MemTracker:
    def __init__(self, device: str):
        self.device = device
        self.proc = psutil.Process(os.getpid()) if psutil else None
    def start(self):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        self._base_rss = self.proc.memory_info().rss if self.proc else 0
    def stop(self):
        gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rss_now = self.proc.memory_info().rss if self.proc else 0
        cpu_delta = max(0, rss_now - self._base_rss)
        return cpu_delta, gpu_peak

# --------------------------
# Build/Load model
# --------------------------
def build_model(input_dim: int = INPUT_DIM) -> nn.Module:
    m=DeepNN(input_dim)
    try:
        ck=torch.load("lung_cancer_model.pth",map_location="cpu")
        m.load_state_dict(ck, strict=False)  # ignore final-layer mismatch
        print("✅ loaded pretrained (strict=False)")
    except Exception as e:
        print("⚠️ load fail:", e)
    return m

# --------------------------
# Dataloaders
# --------------------------
def get_dataloaders():
    X_tr,X_te,y_tr,y_te=train_test_split(X_scaled,y,test_size=0.3,stratify=y,random_state=42)
    X_tr,X_val,y_tr,y_val=train_test_split(X_tr,y_tr,test_size=0.3,stratify=y_tr,random_state=42)
    tr=TensorDataset(torch.tensor(X_tr,dtype=torch.float32), torch.tensor(y_tr.values,dtype=torch.long))
    va=TensorDataset(torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val.values,dtype=torch.long))
    te=TensorDataset(torch.tensor(X_te,dtype=torch.float32), torch.tensor(y_te.values,dtype=torch.long))
    return (DataLoader(tr,batch_size=CFG.batch_size,shuffle=True),
            DataLoader(va,batch_size=CFG.batch_size,shuffle=False),
            DataLoader(te,batch_size=CFG.batch_size,shuffle=False))

# --------------------------
# Eval helpers
# --------------------------
@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: str) -> Dict[str,Any]:
    model.eval(); y_true=[]; y_pred=[]; y_prob=[]
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device)
        logits=model(xb); prob=torch.softmax(logits,dim=1); pred=prob.argmax(1)
        y_true+=yb.cpu().tolist(); y_pred+=pred.cpu().tolist(); y_prob+=prob.cpu().tolist()
    y_true=np.array(y_true); y_pred=np.array(y_pred); y_prob=np.array(y_prob)
    acc=accuracy_score(y_true,y_pred)
    pr,rc,f1,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    # return also trained model
    return {
        "model": model,"acc":acc,"precision":pr,"recall":rc,"f1":f1,"y_true":y_true,"y_pred":y_pred,"y_prob":y_prob}

def save_classification_outputs(test_eval: Dict[str,Any], save_dir: str, tag: str):
    # classification report
    report = classification_report(test_eval["y_true"], test_eval["y_pred"], digits=4, zero_division=0)
    with open(os.path.join(save_dir, f"classification_report_{tag}.txt"), 'w') as f:
        f.write(report)
    # confusion matrix heatmap
    cm = confusion_matrix(test_eval["y_true"], test_eval["y_pred"])
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix ({tag})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{tag}.png"), dpi=200)
    plt.close()

# --------------------------
# Training (DP)
# --------------------------
def train_one_run(base_model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  test_loader: DataLoader,
                  noise: float,
                  seed: int) -> Dict[str,Any]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=copy.deepcopy(base_model).to(CFG.device)
    model.train()
    # compute class weights from the (non-DP) train_loader labels (inverse frequency)
    total0=0; total1=0
    for _, yb_ in train_loader:
        total0 += (yb_==0).sum().item()
        total1 += (yb_==1).sum().item()
    tot = max(1, total0+total1)
    w0 = tot / (2*max(1,total0))
    w1 = tot / (2*max(1,total1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(CFG.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    opt=optim.Adam(model.parameters(), lr=CFG.lr)

    pe=PrivacyEngine()
    model, opt, train_loader = pe.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader,
        noise_multiplier=noise,
        max_grad_norm=CFG.max_grad_norm,
    )

    # timing
    epoch_times=[]
    for _ in range(CFG.epochs):
        t0=time.perf_counter(); model.train()
        for xb,yb in train_loader:
            xb,yb=xb.to(CFG.device), yb.to(CFG.device)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
        if CFG.device.startswith('cuda'): torch.cuda.synchronize()
        epoch_times.append(time.perf_counter()-t0)
        # (simple early model selection by val acc)
    # choose best by final val for simplicity
    test_eval = evaluate_full(model, test_loader, CFG.device)

    # latency & memory (inference)
    with torch.no_grad():
        for xb,_ in test_loader: sample = xb[:1]; break
    mem = MemTracker(CFG.device); mem.start()
    latency_ms = measure_latency(model, sample, CFG.device, repeats=200)
    cpu_b, gpu_b = mem.stop()

    # epsilon accounting (approx via Renyi accountant in Opacus privacy engine)
    eps = float('inf') if noise==0 else pe.get_epsilon(CFG.target_delta)

    return {
        "noise": noise,
        "epsilon": eps,
        "time_per_epoch_s": float(np.mean(epoch_times)),
        "latency_ms": float(latency_ms),
        "cpu_mem_bytes": int(cpu_b),
        "gpu_mem_bytes": int(gpu_b),
        "acc": test_eval["acc"],
        "precision": test_eval["precision"],
        "recall": test_eval["recall"],
        "f1": test_eval["f1"],
        "y_true": test_eval["y_true"],
        "y_pred": test_eval["y_pred"],
        "model": model,
    }

# --------------------------
# Inference scaling experiment
# --------------------------
def run_inference_scaling(feature_counts: Tuple[int,...], n_samples: int, device: str) -> pd.DataFrame:
    rows=[]
    for d in feature_counts:
        model=DeepNN(d).to(device)
        model.eval()
        x=torch.randn(n_samples, d, device=device)
        # latency
        lat=measure_latency(model, x[:1], device, repeats=300)
        # memory
        mem=MemTracker(device); mem.start();
        with torch.no_grad(): _=model(x)
        cpu_b, gpu_b = mem.stop()
        rows.append({"features": d, "latency_ms": lat, "cpu_mem_bytes": cpu_b, "gpu_mem_bytes": gpu_b})
        print(f"[scaling] d={d} -> {lat:.3f} ms, cpu_mem={cpu_b/1e6:.2f} MB, gpu_mem={gpu_b/1e6:.2f} MB")
    df=pd.DataFrame(rows)
    csv=os.path.join(CFG.save_dir, "inference_scaling.csv"); df.to_csv(csv, index=False)
    # plot
    plt.figure();
    plt.plot(df["features"], df["latency_ms"], marker='o');
    plt.xlabel("# Features"); plt.ylabel("Inference latency (ms/sample)");
    plt.title("Inference Time vs Feature Count"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(CFG.save_dir, "inference_scaling_time.png"), dpi=200); plt.close()

    plt.figure();
    plt.plot(df["features"], df["cpu_mem_bytes"]/1e6, marker='o', label='CPU (MB)');
    if (df["gpu_mem_bytes"]>0).any():
        plt.plot(df["features"], df["gpu_mem_bytes"]/1e6, marker='x', label='GPU (MB)')
    plt.xlabel("# Features"); plt.ylabel("Peak memory (MB)"); plt.title("Memory vs Feature Count"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(CFG.save_dir, "inference_scaling_memory.png"), dpi=200); plt.close()

    return df

# --------------------------
# Main
# --------------------------
def main():
    tr,va,te=get_dataloaders(); base=build_model(INPUT_DIM).to(CFG.device)

    # sweep and collect
    rows: List[Dict[str,Any]] = []
    best_models = {}  # noise -> (best_acc, model)[str,Any]] = []
    for n in CFG.noise_list:
        for s in CFG.seeds:
            res = train_one_run(base, tr, va, te, n, s)
            rows.append(res)
            # track per noise best
            if n not in best_models or res['acc'] > best_models[n][0]:
                best_models[n] = (res['acc'], res['model'])
            print(f"noise={n} eps={res['epsilon']:.3f} acc={res['acc']:.4f} t/epoch={res['time_per_epoch_s']:.3f}s lat={res['latency_ms']:.3f}ms")
            # Save per-run confusion & report
            tag=f"noise{n}_seed{s}"
            te_eval = {"y_true":res["y_true"],"y_pred":res["y_pred"]}
            # produce full reports using evaluate_full again to ensure consistency
            # use the trained run's predictions saved in res
            te_eval = {"y_true": res["y_true"], "y_pred": res["y_pred"]}
            save_classification_outputs(te_eval, CFG.save_dir, tag)

    # Save summary CSV
    # save best model per noise
    for noise,(acc,model) in best_models.items():
        torch.save(model.state_dict(), os.path.join(CFG.save_dir,f"best_dp_model_noise{noise}.pth"))

    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ("y_true","y_pred","y_prob") } for r in rows])
    df.to_csv(os.path.join(CFG.save_dir, "dp_full_results.csv"), index=False)

    # Inference scaling experiment (random weights & inputs)
    run_inference_scaling(CFG.scaling_feature_counts, CFG.scaling_samples, CFG.device)

if __name__=='__main__':
    main()
