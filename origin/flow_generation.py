# -*- coding: utf-8 -*-
"""
Traffic dataset generator: VAE -> conditional DDPM (latent diffusion) -> typed decoding
目标：基于真实“流量日志/曝光日志”数据学习分布，生成同 schema 的新数据

Schema：
apppkg, slotid, slotsize, adtype, scenario, maxcount, devicetype,
aid, adid, auid,
oaid, phonebrand, language, isacrossmaker, originalquery, keywords, timehour

关键设计：
1) 连续列（maxcount）：进入 VAE 做连续重构
2) 离散列/文本列：通过 Embedding 联合编码进 VAE，并以分类头解码
3) DDPM 在 VAE 的隐空间上学习分布，仅以 timehour 作为生成条件
4) 高基数字符串 ID：不进入模型，生成时按规则 formatter 直接生成
"""

import os, math, uuid, secrets
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange, tqdm


# ===================== 配置 =====================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# 输入/输出路径
REPO_ROOT = Path.cwd()
OUTPUT_ROOT = REPO_ROOT / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# 自动发现输入 CSV：取 REPO_ROOT.parent/dataset/ 下第一个 .csv
_dataset_dir = REPO_ROOT.parent / "dataset"
_csv_candidates = sorted(_dataset_dir.rglob("*.csv"))
if not _csv_candidates:
    raise FileNotFoundError(f"在 {_dataset_dir} 下未找到任何 .csv 文件")
INPUT_CSV: str = str(_csv_candidates[0])
OUTPUT_CSV: str = str(OUTPUT_ROOT / "generated_traffic.csv")
print(f"[Path] INPUT_CSV  = {INPUT_CSV}")
print(f"[Path] OUTPUT_CSV = {OUTPUT_CSV}")


# VAE
Z_DIM = 32
HIDDEN = 256
EPOCHS_VAE = 100
BATCH_VAE = 256
LR_VAE = 2e-3
KL_BETA_START, KL_BETA_END = 1e-3, 5e-3

# latent DDPM
TIMESTEPS = 100
DDPM_STEPS = 2200
LR_DENOISER = 5e-3
CFG_DROP_PROB = 0.2
CFG_SCALE = 2.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 可调参数 =====================
TARGET_TIMEHOUR: int = 2026010100  
N_GEN = 100_000
print(f"[Config] timehour={TARGET_TIMEHOUR}  n_gen={N_GEN}")

# 文本池TopK（originalquery / keywords）
TOPK_TEXT = 1024              # 对应 originalquery 和 keywords 字段

# 低基数离散列 TopK（apppkg 可能很大，建议TopK截断 + OOV）
TOPK_APP = 8                  # 对应 apppkg 字段
TOPK_SLOTSIZE = 256           # 对应 slotsize 字段
TOPK_ADTYPE = 64              # 对应 adtype 字段
TOPK_PHONEBRAND = 256         # 对应 phonebrand 字段
TOPK_LANG = 64                # 对应 language 字段
TOPK_TIMEHOUR = 256           # 对应 timehour 字段
TOPK_SCENARIO = 8             # 对应 scenario 字段
TOPK_DEVICETYPE = 32          # 对应 devicetype 字段
TOPK_ISACROSSMAKER = 2        # 对应 isacrossmaker 字段（0/1）

# ID 规则（你可按真实数据统计后微调）
SLOTID_LEN_FIXED = 10         # 像 10（base36）
AID_HEX_LEN_FIXED = 32        # 32位hex（匹配真实数据）

# ===================== Schema / 字段分组 =====================
SCHEMA = [
    "apppkg",         # 离散列 (Categorical)
    "slotid",         # 高基数 ID 列，表示曝光/点击的某个广告主ID
    "slotsize",       # 离散列 (Categorical)
    "adtype",         # 离散列 (Categorical)
    "scenario",       # 离散列 (Categorical)
    "maxcount",       # 连续/小整数列 (Numerical)
    "devicetype",     # 离散列 (Categorical)
    "aid",            # 高基数 ID 列，表示曝光/点击的用户的某一ID
    "adid",           # 高基数 ID 列，表示曝光/点击的用户的某一ID
    "auid",           # 高基数 ID 列，表示曝光/点击的用户的某一ID
    "oaid",           # 高基数 ID 列，表示曝光/点击的用户的某一ID
    "phonebrand",     # 离散列 (Categorical)
    "language",       # 离散列 (Categorical)
    "isacrossmaker",  # 离散列 (Categorical) 且为0-1变量
    "originalquery",  # 文本列 (Text)
    "keywords",       # 文本列 (Text)
    "timehour",       # 离散列 (Categorical)表示时间，如 2026010100
]

# 连续/数值列（进入 VAE 的 X_cont）
NUM_COLS: Dict[str, str] = {
    "maxcount": "int",
}

# 离散列（VAE 联合编码 + 分类头）
CAT_COLS_TOPK: Dict[str, int] = {
    "apppkg": TOPK_APP,
    "slotsize": TOPK_SLOTSIZE,
    "adtype": TOPK_ADTYPE,
    "scenario": TOPK_SCENARIO,
    "devicetype": TOPK_DEVICETYPE,
    "isacrossmaker": TOPK_ISACROSSMAKER,
    "phonebrand": TOPK_PHONEBRAND,
    "language": TOPK_LANG,
    "timehour": TOPK_TIMEHOUR,
}

# 文本列：用 topK 池作为分类（工业上更稳；真要生成文本建议另起模型）
TEXT_COLS_TOPK: Dict[str, int] = {
    "originalquery": TOPK_TEXT,
    "keywords": TOPK_TEXT,
}

# 高基数 ID 字段：不进入模型，生成时按 formatter 规则直接生成
ID_COLS = ["slotid", "aid", "adid", "auid", "oaid"]


# ===================== 数据加载 =====================
def load_df() -> pd.DataFrame:
    # 依次尝试常见编码，解决中文 Windows 下 CSV 编码问题
    for enc in ("utf-8", "gb18030", "gbk", "latin-1"):
        try:
            df = pd.read_csv(INPUT_CSV, encoding=enc)
            print(f"[Encoding] 成功使用 {enc} 读取 CSV")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"无法用常见编码读取 {INPUT_CSV}")
    miss = [c for c in SCHEMA if c not in df.columns]
    if miss:
        raise ValueError(f"CSV缺列: {miss}")
    return df[SCHEMA].copy()


df_raw = load_df()


# ===================== Vocab / 编码 =====================
def build_vocab(series: pd.Series, topk: int) -> Tuple[Dict[str, int], int, List[str], np.ndarray]:
    vc = series.fillna("").astype(str).value_counts()
    items = vc.index.tolist()[:topk]
    vocab = {s: i for i, s in enumerate(items)}
    oov = len(items)  # OOV index
    probs = (vc.loc[items].values.astype(np.float64) + 1e-8)
    probs = probs / probs.sum()
    return vocab, oov, items, probs


VOCABS: Dict[str, Dict[str, int]] = {}
OOV_IDX: Dict[str, int] = {}
VOCAB_ITEMS: Dict[str, List[str]] = {}
VOCAB_P: Dict[str, np.ndarray] = {}

for c, k in CAT_COLS_TOPK.items():
    vocab, oov, items, p = build_vocab(df_raw[c], k)
    VOCABS[c], OOV_IDX[c], VOCAB_ITEMS[c], VOCAB_P[c] = vocab, oov, items, p

for c, k in TEXT_COLS_TOPK.items():
    vocab, oov, items, p = build_vocab(df_raw[c], k)
    VOCABS[c], OOV_IDX[c], VOCAB_ITEMS[c], VOCAB_P[c] = vocab, oov, items, p


def cat_to_idx(col: str, s) -> int:
    s = "" if pd.isna(s) else str(s)
    return VOCABS[col].get(s, OOV_IDX[col])


# oaid 真实池及其频次分布（直接以 oaid 为用户身份）
_oaid_counts = df_raw["oaid"].fillna("").astype(str).value_counts()
_oaid_counts = _oaid_counts[_oaid_counts.index != ""]  # 去掉空字符串
OAID_POOL = _oaid_counts.index.tolist()                 # 去重 oaid 列表
OAID_FREQ = _oaid_counts.values.astype(np.float64)       # 对应频次
OAID_PROB = OAID_FREQ / OAID_FREQ.sum()                  # 归一化为概率

# 数值缩放（将连续列归一化到 [0,1]）
class MinMaxScaler:
    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        close = (self.max_ - self.min_) < 1e-6
        self.max_[close] = self.min_[close] + 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * (self.max_ - self.min_) + self.min_

NUM_KEYS = list(NUM_COLS.keys())
X_num = np.zeros((len(df_raw), len(NUM_KEYS)), dtype=np.float32)

for j, c in enumerate(NUM_KEYS):
    x = pd.to_numeric(df_raw[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    x = np.clip(x, 0, 50)
    X_num[:, j] = x.astype(np.float32)

assert np.isfinite(X_num).all()



scaler = MinMaxScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num).astype(np.float32)


# ===================== labels：cat / text =====================
Y_cat = {c: df_raw[c].apply(lambda s: cat_to_idx(c, s)).astype(np.int64).values for c in CAT_COLS_TOPK}
Y_text = {c: df_raw[c].apply(lambda s: cat_to_idx(c, s)).astype(np.int64).values for c in TEXT_COLS_TOPK}


# ===================== 训练/验证划分 =====================
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X_scaled))
cut = int(0.9 * len(perm))
tr_idx, va_idx = perm[:cut], perm[cut:]


def pack(idx):
    X = torch.tensor(X_scaled[idx], dtype=torch.float32)
    ycat = {c: torch.tensor(Y_cat[c][idx], dtype=torch.long) for c in CAT_COLS_TOPK}
    ytxt = {c: torch.tensor(Y_text[c][idx], dtype=torch.long) for c in TEXT_COLS_TOPK}
    return X, ycat, ytxt


Xtr, ycat_tr, ytxt_tr = pack(tr_idx)
Xva, ycat_va, ytxt_va = pack(va_idx)


# ===================== 模型：VAE + 多头 =====================
class MLP(nn.Module):
    def __init__(self, din, dh, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dh), nn.ReLU(),
            nn.Linear(dh, dh), nn.ReLU(),
            nn.Linear(dh, dout),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, num_dim, zdim, hidden,
                 cat_sizes: Dict[str, int],
                 text_sizes: Dict[str, int],
                 cat_emb_dims: Dict[str, int],
                 text_emb_dims: Dict[str, int]):
        super().__init__()
        self.zd = zdim
        self.num_dim = num_dim

        # 编码器端：为每个离散/文本字段创建 Embedding（维度按字段基数自适应）
        self.cat_embs = nn.ModuleDict({c: nn.Embedding(cat_sizes[c], cat_emb_dims[c]) for c in cat_sizes})
        self.text_embs = nn.ModuleDict({c: nn.Embedding(text_sizes[c], text_emb_dims[c]) for c in text_sizes})

        enc_in = num_dim + sum(cat_emb_dims.values()) + sum(text_emb_dims.values())
        self.enc = MLP(enc_in, hidden, 2 * zdim)
        self.dec = MLP(zdim, hidden, num_dim)   # 解码器只重构数值列

        # 分类头：从 z 预测离散/文本字段
        self.cat_heads = nn.ModuleDict({c: nn.Linear(zdim, sz) for c, sz in cat_sizes.items()})
        self.text_heads = nn.ModuleDict({c: nn.Linear(zdim, sz) for c, sz in text_sizes.items()})

    def encode(self, x_num, cat_idx, text_idx):
        parts = [x_num]
        for c, emb in self.cat_embs.items():
            parts.append(emb(cat_idx[c]))
        for c, emb in self.text_embs.items():
            parts.append(emb(text_idx[c]))
        h = self.enc(torch.cat(parts, dim=-1))
        return h[:, :self.zd], h[:, self.zd:]

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon = self.dec(z)
        cat_logits = {c: head(z) for c, head in self.cat_heads.items()}
        text_logits = {c: head(z) for c, head in self.text_heads.items()}
        return recon, cat_logits, text_logits

    def forward(self, x_num, cat_idx, text_idx):
        mu, lv = self.encode(x_num, cat_idx, text_idx)
        z = self.reparam(mu, lv)
        recon, cat_logits, text_logits = self.decode(z)
        return recon, cat_logits, text_logits, mu, lv


# 分类空间大小（+1 for OOV）
CAT_SIZES = {c: (len(VOCAB_ITEMS[c]) + 1) for c in CAT_COLS_TOPK}
TEXT_SIZES = {c: (len(VOCAB_ITEMS[c]) + 1) for c in TEXT_COLS_TOPK}

# 每个字段的 embedding 维度：log2(TOPK)，钳位到 [2, 32]
def _emb_dim(topk: int) -> int:
    return min(32, max(2, int(math.log2(topk))))

CAT_EMB_DIMS = {c: _emb_dim(CAT_COLS_TOPK[c]) for c in CAT_COLS_TOPK}
TEXT_EMB_DIMS = {c: _emb_dim(TEXT_COLS_TOPK[c]) for c in TEXT_COLS_TOPK}

vae = VAE(
    num_dim=len(NUM_KEYS),
    zdim=Z_DIM,
    hidden=HIDDEN,
    cat_sizes=CAT_SIZES,
    text_sizes=TEXT_SIZES,
    cat_emb_dims=CAT_EMB_DIMS,
    text_emb_dims=TEXT_EMB_DIMS,
).to(DEVICE)

opt_vae = torch.optim.Adam(vae.parameters(), lr=LR_VAE)


def vae_step(xb, ycat_b, ytxt_b, beta):
    recon, cat_logits, text_logits, mu, lv = vae(xb, ycat_b, ytxt_b)

    # 连续重构
    mse = F.mse_loss(recon, xb)

    # categorical CE
    ce_cat = 0.0
    for c in CAT_COLS_TOPK:
        ce_cat = ce_cat + F.cross_entropy(cat_logits[c], ycat_b[c])

    # text CE
    ce_txt = 0.0
    for c in TEXT_COLS_TOPK:
        ce_txt = ce_txt + F.cross_entropy(text_logits[c], ytxt_b[c])

    # KL
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

    # 权重可调：离散字段更重要就增大
    loss = mse + 0.4 * ce_cat + 0.2 * ce_txt + beta * kld
    if not torch.isfinite(loss):
        return torch.tensor(0.0, device=loss.device)  # 或 raise
    return loss


# 构造 DataLoader（把 dict 展平为 tensor 列表）
def make_dataset(X, ycat, ytxt):
    tensors = [X]
    # 固定顺序
    for c in CAT_COLS_TOPK:
        tensors.append(ycat[c])
    for c in TEXT_COLS_TOPK:
        tensors.append(ytxt[c])
    return TensorDataset(*tensors)


def unpack_batch(batch):
    xb = batch[0].to(DEVICE, non_blocking=True)
    i = 1
    ycat_b = {}
    for c in CAT_COLS_TOPK:
        ycat_b[c] = batch[i].to(DEVICE, non_blocking=True)
        i += 1
    ytxt_b = {}
    for c in TEXT_COLS_TOPK:
        ytxt_b[c] = batch[i].to(DEVICE, non_blocking=True)
        i += 1
    return xb, ycat_b, ytxt_b


train_ds = make_dataset(
    Xtr,
    {c: ycat_tr[c] for c in CAT_COLS_TOPK},
    {c: ytxt_tr[c] for c in TEXT_COLS_TOPK},
)
val_ds = make_dataset(
    Xva,
    {c: ycat_va[c] for c in CAT_COLS_TOPK},
    {c: ytxt_va[c] for c in TEXT_COLS_TOPK},
)

# 训练 VAE
for ep in trange(EPOCHS_VAE, desc="[VAE]"):
    beta = KL_BETA_START + (KL_BETA_END - KL_BETA_START) * ep / max(1, EPOCHS_VAE - 1)

    vae.train()
    tl = []
    for b in DataLoader(train_ds, batch_size=BATCH_VAE, shuffle=True, drop_last=True, num_workers=0):
        xb, ycat_b, ytxt_b = unpack_batch(b)
        opt_vae.zero_grad()
        loss = vae_step(xb, ycat_b, ytxt_b, beta)
        loss.backward()
        opt_vae.step()
        tl.append(loss.item())

    vae.eval()
    vl = []
    with torch.no_grad():
        for b in DataLoader(val_ds, batch_size=BATCH_VAE, shuffle=False, num_workers=0):
            xb, ycat_b, ytxt_b = unpack_batch(b)
            vl.append(vae_step(xb, ycat_b, ytxt_b, beta).item())

    tqdm.write(f"ep{ep+1:02d} train={np.mean(tl):.4f} valid={np.mean(vl):.4f}")


# 取潜向量均值作为“data latent”
vae.eval()
with torch.no_grad():
    _all_cat = {c: torch.tensor(Y_cat[c], dtype=torch.long, device=DEVICE) for c in CAT_COLS_TOPK}
    _all_txt = {c: torch.tensor(Y_text[c], dtype=torch.long, device=DEVICE) for c in TEXT_COLS_TOPK}
    MU, _ = vae.encode(
        torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE),
        _all_cat, _all_txt,
    )
Z_all = MU.detach()


# ===================== 条件 DDPM：仅以 timehour 作为条件 =====================
EMB_DIM = 32
COND_KEYS = ["timehour"]

# 条件 embedding 表（仅 timehour）
emb_tables = nn.ModuleDict({
    "timehour": nn.Embedding(CAT_SIZES["timehour"], EMB_DIM),
}).to(DEVICE)
opt_emb = torch.optim.Adam(emb_tables.parameters(), lr=LR_DENOISER * 0.5)


def get_cond_from_index(idx: np.ndarray) -> Dict[str, torch.Tensor]:
    """从 df_raw 的索引取 timehour cond label（训练DDPM用）"""
    return {"timehour": torch.tensor(Y_cat["timehour"][idx], dtype=torch.long, device=DEVICE)}


def cond_embed(cond_labels: Dict[str, torch.Tensor]) -> torch.Tensor:
    return emb_tables["timehour"](cond_labels["timehour"])  # [B, EMB_DIM]


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class CondDenoiser(nn.Module):
    def __init__(self, zdim, tdim=64, cdim=EMB_DIM * len(COND_KEYS), hidden=HIDDEN):
        super().__init__()
        self.t = TimeEmbed(tdim)
        self.net = nn.Sequential(
            nn.Linear(zdim + tdim + cdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, zdim),
        )

    def forward(self, zt, t, cond_vec):
        return self.net(torch.cat([zt, self.t(t), cond_vec], dim=-1))


def schedule(T=TIMESTEPS, beta_start=1e-4, beta_end=2e-2, device=DEVICE):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


betas, alphas, abar = schedule()
den = CondDenoiser(Z_DIM).to(DEVICE)
opt_den = torch.optim.Adam(den.parameters(), lr=LR_DENOISER)

# DDPM 训练集 latent
Ztr = Z_all[:cut]  # 与 tr_idx 对齐切分

def rand_t(n): return torch.randint(0, TIMESTEPS, (n,), device=DEVICE)

def add_noise(z0, t):
    a_bar = abar[t].unsqueeze(-1)
    eps = torch.randn_like(z0)
    zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * eps
    return zt, eps


for step in trange(DDPM_STEPS, desc="[LDM]"):
    bs = 256
    sel = np.random.randint(0, Ztr.shape[0], size=bs)
    z0 = Ztr[sel]
    t = rand_t(z0.size(0))
    zt, eps = add_noise(z0, t)

    cond_labels = get_cond_from_index(sel)
    cond_vec = cond_embed(cond_labels)

    # CFG drop
    drop = (torch.rand(bs, device=DEVICE) < CFG_DROP_PROB).float().unsqueeze(-1)
    cond_in = cond_vec * (1.0 - drop)

    pred = den(zt, t, cond_in)
    loss = F.mse_loss(pred, eps)

    opt_den.zero_grad()
    opt_emb.zero_grad()
    loss.backward()
    opt_den.step()
    opt_emb.step()

    if (step + 1) % 200 == 0:
        tqdm.write(f"step {step+1}  loss={loss.item():.5f}")


# ===================== 采样：latent DDPM + typed decode =====================
@torch.no_grad()
def p_sample(n, cond_labels: Dict[str, torch.Tensor], cfg_scale=CFG_SCALE, show=True):
    z = torch.randn(n, Z_DIM, device=DEVICE)
    c = cond_embed(cond_labels)
    c0 = torch.zeros_like(c)

    iterator = range(TIMESTEPS - 1, -1, -1)
    if show:
        iterator = trange(TIMESTEPS - 1, -1, -1, desc="[Sampling]")

    for t in iterator:
        tt = torch.full((n,), t, dtype=torch.long, device=DEVICE)
        eps_c = den(z, tt, c)
        eps_uc = den(z, tt, c0)
        eps = eps_uc + cfg_scale * (eps_c - eps_uc)

        a = alphas[t]
        z = (z - (1 - a).sqrt() * eps) / a.sqrt()
        if t > 0:
            z = z + betas[t].sqrt() * torch.randn_like(z)

    return z


# ---------- ID formatter ----------
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"

def gen_base36(n_chars: int) -> str:
    return "".join(secrets.choice(_BASE36) for _ in range(n_chars))

def gen_hex(n_chars: int) -> str:
    n_bytes = (n_chars + 1) // 2
    s = secrets.token_hex(n_bytes)
    return s[:n_chars]

def gen_uuid() -> str:
    return str(uuid.uuid4())


@torch.no_grad()
def decode_typed(z, cond_labels: Dict[str, torch.Tensor]) -> pd.DataFrame:
    """
    解码：
    - 连续数值：vae.dec + inverse scaler + int化
    - 离散：vae 分类头 argmax -> vocab
    - text：分类头 argmax -> 文本池
    - ID：按规则 formatter 直接生成（不经过模型）
    """
    recon_num, cat_logits, text_logits = vae.decode(z)

    # 数值
    X_np = recon_num.detach().cpu().numpy().astype(np.float32)
    X_np = scaler.inverse(X_np)

    out = {}

    for j, c in enumerate(NUM_KEYS):
        v = np.clip(np.round(X_np[:, j]), 0, 50).astype(int)
        out[c] = v
    def sample_from_logits(logits, temperature=1.2):
        p = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(p, 1).squeeze(-1)

    # 离散列：argmax -> vocab string
    for c in CAT_COLS_TOPK:
        idx = sample_from_logits(cat_logits[c], temperature=1.3).detach().cpu().numpy()

        items = VOCAB_ITEMS[c]
        oov = OOV_IDX[c]
        # idx==oov 时给一个“按真实分布采样”的替代，更稳
        vals = []
        for t in idx:
            if t >= len(items):
                vals.append(str(np.random.choice(items, p=VOCAB_P[c])))
            else:
                vals.append(items[int(t)])
        out[c] = vals

    # 文本列：同理
    for c in TEXT_COLS_TOPK:
        idx = torch.argmax(text_logits[c], dim=-1).detach().cpu().numpy()
        items = VOCAB_ITEMS[c]
        vals = []
        for t in idx:
            if t >= len(items):
                # OOV：给空或按分布采样都可以；这里按分布采样
                vals.append(str(np.random.choice(items, p=VOCAB_P[c])))
            else:
                vals.append(items[int(t)])
        out[c] = vals

    # ---- 生成 ID：按 oaid 真实频次分布采样，相同 oaid 共享 aid/adid/auid ----
    n = len(z)

    # 1) 按真实频次分布为每行采样一个 oaid
    row_oaids = np.random.choice(OAID_POOL, size=n, p=OAID_PROB)

    # 2) 相同 oaid 绑定同一组 aid/adid/auid
    _id_cache = {}  # oaid -> (aid, adid, auid)
    aids, adids, auids = [], [], []
    for oid in row_oaids:
        if oid not in _id_cache:
            _id_cache[oid] = (gen_hex(AID_HEX_LEN_FIXED), gen_uuid(), gen_uuid())
        a, d, u = _id_cache[oid]
        aids.append(a)
        adids.append(d)
        auids.append(u)

    # slotid 是广告位 ID，与用户无关，每行独立生成
    out["slotid"] = [gen_base36(SLOTID_LEN_FIXED) for _ in range(n)]
    out["aid"]  = aids
    out["adid"] = adids
    out["auid"] = auids
    out["oaid"] = list(row_oaids)

    # 组装 DataFrame 并按 SCHEMA 排列
    df = pd.DataFrame(out)

    # 补齐 schema 中但不在 out 的列（安全）
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = ""

    return df[SCHEMA]


# ---------- 条件采样：仅指定 timehour ----------
def sample_cond_from_real(n: int) -> Dict[str, torch.Tensor]:
    th_idx = VOCABS["timehour"].get(str(TARGET_TIMEHOUR), OOV_IDX["timehour"])
    return {"timehour": torch.full((n,), th_idx, dtype=torch.long, device=DEVICE)}


@torch.no_grad()
def generate_traffic(n=N_GEN, cfg_scale=CFG_SCALE):
    cond = sample_cond_from_real(n)
    z = p_sample(n, cond, cfg_scale=cfg_scale, show=True)
    df = decode_typed(z, cond)
    return df


# ===================== run & export =====================
df_gen = generate_traffic(N_GEN, cfg_scale=CFG_SCALE)
tqdm.write("Saving CSV ...")
df_gen.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)
tqdm.write(f"Done. -> {os.path.abspath(OUTPUT_CSV)}")


# ===================== Validation =====================
# 数值列：KS 检验（分布距离）
# 离散/文本列：TV 距离（Top100 频率差异）

def ks_stat(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    xs = np.sort(x); ys = np.sort(y)
    z = np.sort(np.unique(np.concatenate([xs, ys])))
    def ecdf(a, t): return np.searchsorted(a, t, side="right") / len(a)
    d = 0.0
    for t in z[::max(1, len(z)//2000)]:  # 降采样加速
        d = max(d, abs(ecdf(xs, t) - ecdf(ys, t)))
    return float(d)

def tv_dist(p, q):
    # total variation distance = 0.5*L1
    keys = set(p.keys()) | set(q.keys())
    s = 0.0
    for k in keys:
        s += abs(p.get(k, 0.0) - q.get(k, 0.0))
    return 0.5 * s

def freq_dist(series: pd.Series, topk=50):
    vc = series.fillna("").astype(str).value_counts(normalize=True)
    top = vc.head(topk)
    return top.to_dict()

try:
    real_df = df_raw
    gen_df = pd.read_csv(OUTPUT_CSV)

    # 数值 KS
    ks_rows = []
    for c in NUM_KEYS:
        xr = pd.to_numeric(real_df[c], errors="coerce").values
        xg = pd.to_numeric(gen_df[c], errors="coerce").values
        ks_rows.append({"feature": c, "KS_D": ks_stat(xr, xg)})

    # 离散 TV（看 top 分布差异）
    tv_rows = []
    for c in list(CAT_COLS_TOPK.keys()) + list(TEXT_COLS_TOPK.keys()):
        pr = freq_dist(real_df[c], topk=100)
        pg = freq_dist(gen_df[c], topk=100)
        tv_rows.append({"feature": c, "TV_top100": tv_dist(pr, pg)})

    val = pd.DataFrame(ks_rows).sort_values("KS_D", ascending=False)
    tvt = pd.DataFrame(tv_rows).sort_values("TV_top100", ascending=False)

    ks_path = str(OUTPUT_ROOT / "traffic_validation_ks.csv")
    tv_path = str(OUTPUT_ROOT / "traffic_validation_tv.csv")
    val.to_csv(ks_path, index=False, encoding="utf-8-sig")
    tvt.to_csv(tv_path, index=False, encoding="utf-8-sig")

    print("\n[Validation Done]")
    print(f" - KS saved: {ks_path}")
    print(f" - TV saved: {tv_path}")
except Exception as e:
    print("\n[Validation Skipped] reason:", repr(e))
