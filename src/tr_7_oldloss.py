import math
import random
import csv

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformer_lens import HookedTransformer



DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


HOOK_NAME = "blocks.6.hook_resid_pre"

D_MODEL = 768

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

SEQ_LEN = 128
TRAIN_DOC_FRACTION = 0.90

MAX_TRAIN_CHUNKS = 4000
MAX_VAL_CHUNKS = 500

BATCH_SIZE = 32
N_STEPS = 5000

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


MIN_NOISE_NORM = 4.0
MAX_NOISE_NORM = 128.0

HIDDEN_DIM = 1536
SIGMA_DIM = 64

VAL_BATCHES = 20
LOG_EVERY = 50
VAL_EVERY = 250

CHECKPOINT_PATH = "baseline_denoiser_block7.pt"
HISTORY_CSV = "baseline_denoiser_block7_history.csv"

PLOT_MSE = "baseline_denoiser_block7_mse.png"
PLOT_RATIO = "baseline_denoiser_block7_ratio.png"




print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

print("GPT-2 loaded.")
print("Activation layer:", HOOK_NAME)




print("\nLoading WikiText-2...")

dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="train"
)

documents = [
    row["text"].strip()
    for row in dataset
    if len(row["text"].strip()) >= 100
]

random.Random(SEED).shuffle(documents)

split_idx = int(
    TRAIN_DOC_FRACTION * len(documents)
)

train_docs = documents[:split_idx]
val_docs = documents[split_idx:]

print("documents:", len(documents))
print("train docs:", len(train_docs))
print("val docs:", len(val_docs))


@torch.no_grad()
def build_token_chunks(docs, max_chunks, name):
    chunks = []

    for text in docs:
        tokens = model.to_tokens(
            text,
            prepend_bos=True
        )[0].cpu()

        if tokens.numel() < SEQ_LEN:
            continue

        for start in range(
            0,
            tokens.numel() - SEQ_LEN + 1,
            SEQ_LEN
        ):
            chunks.append(
                tokens[start:start + SEQ_LEN]
            )

            if len(chunks) >= max_chunks:
                out = torch.stack(chunks)
                print(name, "token chunks:", tuple(out.shape))
                return out

    if not chunks:
        raise RuntimeError(
            f"No {name} chunks were created."
        )

    out = torch.stack(chunks)
    print(name, "token chunks:", tuple(out.shape))
    return out


train_tokens = build_token_chunks(
    train_docs,
    MAX_TRAIN_CHUNKS,
    "train"
)

val_tokens = build_token_chunks(
    val_docs,
    MAX_VAL_CHUNKS,
    "validation"
)




@torch.no_grad()
def extract_activations(tokens_tensor, name):
    all_h = []

    extraction_batch_size = 8

    for start in range(
        0,
        tokens_tensor.shape[0],
        extraction_batch_size
    ):
        batch = tokens_tensor[
            start:start + extraction_batch_size
        ].to(DEVICE)

        _, cache = model.run_with_cache(
            batch,
            names_filter=[HOOK_NAME]
        )

        h = cache[HOOK_NAME].detach().float().cpu()

        
        all_h.append(
            h.reshape(-1, D_MODEL)
        )

        if (
            start == 0
            or
            start % 200 == 0
        ):
            print(
                f"{name}: processed "
                f"{min(start + extraction_batch_size, tokens_tensor.shape[0])}"
                f"/{tokens_tensor.shape[0]} chunks"
            )

    out = torch.cat(all_h, dim=0)

    print(
        name,
        "activations:",
        tuple(out.shape)
    )

    return out


print("\nExtracting clean block-7 activations...")

train_h = extract_activations(
    train_tokens,
    "train"
)

val_h = extract_activations(
    val_tokens,
    "validation"
)




class SigmaEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        # sigma: [B, 1]

        log_sigma = torch.log(
            sigma.clamp_min(1e-8)
        )

        half = self.dim // 2

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=sigma.device,
                dtype=sigma.dtype
            )
        )

        angles = (
            log_sigma
            *
            frequencies.view(1, -1)
        )

        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=-1
        )




class ResidualDenoiser(nn.Module):

    def __init__(
        self,
        d_model=768,
        hidden_dim=1536,
        sigma_dim=64
    ):
        super().__init__()

        self.sigma_embedding = SigmaEmbedding(
            sigma_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                d_model + sigma_dim,
                hidden_dim
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                d_model
            )
        )

        
        nn.init.zeros_(
            self.net[-1].weight
        )
        nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(self, x, sigma):
        sigma_emb = self.sigma_embedding(
            sigma
        )

        correction = self.net(
            torch.cat(
                [x, sigma_emb],
                dim=-1
            )
        )

        return x + correction


denoiser = ResidualDenoiser(
    d_model=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    sigma_dim=SIGMA_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    denoiser.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)



def sample_noise(batch_size):
    log_r = torch.empty(
        batch_size,
        1,
        device=DEVICE
    ).uniform_(
        math.log(MIN_NOISE_NORM),
        math.log(MAX_NOISE_NORM)
    )

    r = torch.exp(log_r)

    sigma = (
        r
        /
        math.sqrt(D_MODEL)
    )

    epsilon = (
        torch.randn(
            batch_size,
            D_MODEL,
            device=DEVICE
        )
        *
        sigma
    )

    return epsilon, sigma, r




VAL_NORMS = [
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    96.0,
    128.0,
]


@torch.no_grad()
def validate():
    denoiser.eval()

    # Use a fixed subset for speed.
    n = min(
        val_h.shape[0],
        VAL_BATCHES * BATCH_SIZE
    )

    clean = val_h[:n].to(DEVICE)

    results = {}

    for r in VAL_NORMS:
        sigma_value = (
            r
            /
            math.sqrt(D_MODEL)
        )

        # Fixed random generator is not guaranteed identically
        # implemented across devices, so seed before each r.
        torch.manual_seed(
            SEED + int(r * 10)
        )

        sigma = torch.full(
            (n, 1),
            sigma_value,
            device=DEVICE,
            dtype=clean.dtype
        )

        epsilon = (
            torch.randn_like(clean)
            *
            sigma
        )

        noisy = clean + epsilon

        pred = denoiser(
            noisy,
            sigma
        )

        raw_mse = (
            (noisy - clean)
            .pow(2)
            .mean()
            .item()
        )

        den_mse = (
            (pred - clean)
            .pow(2)
            .mean()
            .item()
        )

        improvement = (
            raw_mse / den_mse
            if den_mse > 0
            else float("inf")
        )

        results[r] = {
            "raw_mse": raw_mse,
            "den_mse": den_mse,
            "improvement": improvement,
        }

    return results




history = []

print("\nStarting baseline denoiser training...")

for step in range(
    1,
    N_STEPS + 1
):
    denoiser.train()

    idx = torch.randint(
        0,
        train_h.shape[0],
        (BATCH_SIZE,)
    )

    clean = train_h[idx].to(DEVICE)

    epsilon, sigma, r = sample_noise(
        BATCH_SIZE
    )

    noisy = clean + epsilon

    pred = denoiser(
        noisy,
        sigma
    )

    loss = (
        (pred - clean)
        .pow(2)
        .mean()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        denoiser.parameters(),
        GRAD_CLIP
    )

    optimizer.step()

    if (
        step == 1
        or
        step % LOG_EVERY == 0
    ):
        print(
            f"step={step:5d} | "
            f"MSE={loss.item():.6f} | "
            f"mean r={r.mean().item():.2f}"
        )

    if (
        step == 1
        or
        step % VAL_EVERY == 0
        or
        step == N_STEPS
    ):
        metrics = validate()

        row = {
            "step": step,
            "train_mse": loss.item(),
        }

        print("\nVALIDATION")

        for r in VAL_NORMS:
            m = metrics[r]

            row[f"raw_mse_r{int(r)}"] = (
                m["raw_mse"]
            )

            row[f"den_mse_r{int(r)}"] = (
                m["den_mse"]
            )

            row[f"improvement_r{int(r)}"] = (
                m["improvement"]
            )

            print(
                f"r={r:>5.0f} | "
                f"raw={m['raw_mse']:.6f} | "
                f"den={m['den_mse']:.6f} | "
                f"x{m['improvement']:.3f}"
            )

        print()

        history.append(row)




checkpoint = {
    "state_dict":
        denoiser.state_dict(),

    "hook_name":
        HOOK_NAME,

    "d_model":
        D_MODEL,

    "hidden_dim":
        HIDDEN_DIM,

    "sigma_dim":
        SIGMA_DIM,

    "min_noise_norm":
        MIN_NOISE_NORM,

    "max_noise_norm":
        MAX_NOISE_NORM,

    "objective":
        "MSE(D(h + epsilon, sigma), h)",

    "corruption":
        "epsilon ~ N(0, sigma^2 I)",

    "uses_target_steering_vector":
        False,

    "seed":
        SEED,
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

print(
    "Saved checkpoint:",
    CHECKPOINT_PATH
)


if history:
    with open(
        HISTORY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            )
        )
        writer.writeheader()
        writer.writerows(history)



if history:
    steps = [
        row["step"]
        for row in history
    ]

    plt.figure(
        figsize=(9, 6)
    )

    for r in [
        8.0,
        16.0,
        32.0,
        64.0,
        128.0
    ]:
        plt.plot(
            steps,
            [
                row[
                    f"den_mse_r{int(r)}"
                ]
                for row in history
            ],
            label=f"r={int(r)}"
        )

    plt.xlabel("Training step")
    plt.ylabel("Validation reconstruction MSE")
    plt.title(
        "Block-7 baseline denoiser"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_MSE,
        dpi=200
    )
    plt.show()


    plt.figure(
        figsize=(9, 6)
    )

    for r in [
        8.0,
        16.0,
        32.0,
        64.0,
        128.0
    ]:
        plt.plot(
            steps,
            [
                row[
                    f"improvement_r{int(r)}"
                ]
                for row in history
            ],
            label=f"r={int(r)}"
        )

    plt.axhline(
        1.0,
        linestyle="--"
    )

    plt.xlabel("Training step")
    plt.ylabel("raw MSE / denoised MSE")
    plt.title(
        "Block-7 denoising improvement"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_RATIO,
        dpi=200
    )
    plt.show()




final = validate()

print(
    "\n"
    +
    "=" * 80
)

print(
    "FINAL GAUSSIAN RECONSTRUCTION VALIDATION"
)

print(
    "=" * 80
)

for r in VAL_NORMS:
    m = final[r]

    print(
        f"r={r:>5.0f} | "
        f"raw MSE={m['raw_mse']:.6f} | "
        f"den MSE={m['den_mse']:.6f} | "
        f"improvement={m['improvement']:.3f}x"
    )
