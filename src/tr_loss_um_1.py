import csv, random, math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformer_lens import HookedTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); random.seed(SEED)

HOOK_NAME = "blocks.6.hook_resid_pre"
D_MODEL = 768
SEQ_LEN = 128
TRAIN_DOC_FRACTION = 0.90
MAX_TRAIN_CHUNKS = 4000
MAX_VAL_CHUNKS = 500
BATCH_SIZE = 32
N_STEPS = 5000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HIDDEN_DIM = 1536
T_EMBED_DIM = 64
T_MIN, T_MAX = 0.50, 1.00
VAL_T_VALUES = [0.99,0.95,0.90,0.80,0.70,0.60,0.50]
LOG_EVERY = 50
VAL_EVERY = 250

CHECKPOINT_PATH = "improved_interp_denoiser_block7.pt"
HISTORY_CSV = "improved_interp_denoiser_block7_history.csv"

print("device:", DEVICE)
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
for p in model.parameters(): p.requires_grad_(False)

dataset = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
documents = [r["text"].strip() for r in dataset if len(r["text"].strip()) >= 100]
random.Random(SEED).shuffle(documents)
split_idx = int(TRAIN_DOC_FRACTION*len(documents))
train_docs, val_docs = documents[:split_idx], documents[split_idx:]

@torch.no_grad()
def build_chunks(docs,max_chunks):
    chunks=[]
    for text in docs:
        toks=model.to_tokens(text,prepend_bos=True)[0].cpu()
        for s in range(0,max(0,toks.numel()-SEQ_LEN+1),SEQ_LEN):
            chunks.append(toks[s:s+SEQ_LEN])
            if len(chunks)>=max_chunks: return torch.stack(chunks)
    if not chunks: raise RuntimeError("No chunks")
    return torch.stack(chunks)

train_tokens=build_chunks(train_docs,MAX_TRAIN_CHUNKS)
val_tokens=build_chunks(val_docs,MAX_VAL_CHUNKS)

@torch.no_grad()
def extract(tokens):
    out=[]
    for s in range(0,tokens.shape[0],8):
        b=tokens[s:s+8].to(DEVICE)
        _,cache=model.run_with_cache(b,names_filter=[HOOK_NAME])
        out.append(cache[HOOK_NAME].detach().float().cpu().reshape(-1,D_MODEL))
    return torch.cat(out,dim=0)

print("Extracting activations...")
train_h=extract(train_tokens)
val_h=extract(val_tokens)
TRAIN_MEAN_H_NORM=torch.linalg.vector_norm(train_h,dim=-1).mean().item()
print("mean ||h7|| =",TRAIN_MEAN_H_NORM)

class TEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2
        f=torch.exp(torch.linspace(math.log(1.0),math.log(1000.0),half,device=t.device,dtype=t.dtype))
        a=t*f.view(1,-1)
        return torch.cat([torch.sin(a),torch.cos(a)],dim=-1)

class ImprovedInterpolationDenoiser(nn.Module):
    def __init__(self,d_model=768,hidden_dim=1536,t_embed_dim=64):
        super().__init__()
        self.t_embedding=TEmbedding(t_embed_dim)
        self.net=nn.Sequential(
            nn.Linear(d_model+t_embed_dim,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,d_model)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,t):
        e=self.t_embedding(t)
        raw_corr=self.net(torch.cat([x,e],dim=-1))
        corr=(1.0-t)*raw_corr
        return x+corr

denoiser=ImprovedInterpolationDenoiser(D_MODEL,HIDDEN_DIM,T_EMBED_DIM).to(DEVICE)
opt=torch.optim.AdamW(denoiser.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

def corrupt(clean):
    t=torch.empty(clean.shape[0],1,device=clean.device,dtype=clean.dtype).uniform_(T_MIN,T_MAX)
    eps=torch.randn_like(clean)
    xt=t*clean+(1.0-t)*eps
    return xt,t

@torch.no_grad()
def validate():
    denoiser.eval()
    clean=val_h[:min(2048,val_h.shape[0])].to(DEVICE)
    out={}
    for tv in VAL_T_VALUES:
        torch.manual_seed(SEED+int(tv*1000))
        t=torch.full((clean.shape[0],1),tv,device=DEVICE,dtype=clean.dtype)
        eps=torch.randn_like(clean)
        xt=t*clean+(1-t)*eps
        pred=denoiser(xt,t)
        raw=((xt-clean)**2).mean().item()
        den=((pred-clean)**2).mean().item()
        out[tv]=(raw,den,raw/den if den>0 else float("inf"))
    return out

history=[]
for step in range(1,N_STEPS+1):
    denoiser.train()
    idx=torch.randint(0,train_h.shape[0],(BATCH_SIZE,))
    clean=train_h[idx].to(DEVICE)
    xt,t=corrupt(clean)
    pred=denoiser(xt,t)
    loss=((pred-clean)**2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(),GRAD_CLIP)
    opt.step()

    if step==1 or step%LOG_EVERY==0:
        print(f"step={step:5d} | MSE={loss.item():.6f} | mean t={t.mean().item():.3f}")

    if step==1 or step%VAL_EVERY==0 or step==N_STEPS:
        metrics=validate()
        row={"step":step,"train_mse":loss.item()}
        print("\nVALIDATION")
        for tv,(raw,den,imp) in metrics.items():
            key=str(tv).replace(".","_")
            row[f"raw_t{key}"]=raw; row[f"den_t{key}"]=den; row[f"imp_t{key}"]=imp
            print(f"t={tv:.2f} | raw={raw:.6f} | den={den:.6f} | x{imp:.3f}")
        history.append(row)

torch.save({
    "state_dict":denoiser.state_dict(),
    "hook_name":HOOK_NAME,
    "d_model":D_MODEL,
    "hidden_dim":HIDDEN_DIM,
    "t_embed_dim":T_EMBED_DIM,
    "t_min":T_MIN,
    "t_max":T_MAX,
    "train_mean_h_norm":TRAIN_MEAN_H_NORM,
    "architecture":"D(x,t)=x+(1-t)*Delta(x,t)",
    "objective":"MSE(D(t*h+(1-t)*eps,t),h)",
    "uses_target_steering_vector":False
},CHECKPOINT_PATH)

with open(HISTORY_CSV,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

print("\nFINAL")
for tv,(raw,den,imp) in validate().items():
    print(f"t={tv:.2f} | raw MSE={raw:.6f} | den MSE={den:.6f} | improvement={imp:.3f}x")
print("Saved:",CHECKPOINT_PATH)
