import math
import random
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformer_lens import HookedTransformer


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


PERTURB_HOOK = "blocks.6.hook_resid_pre"
REPAIR_HOOK = "blocks.8.hook_resid_pre"

D_MODEL = 768

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

MAX_SEQ_LEN = 128
TRAIN_DOC_FRACTION = 0.90


BATCH_SIZE = 6
N_STEPS = 3000

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


MIN_NOISE_NORM = 4.0
MAX_NOISE_NORM = 128.0


HIDDEN_DIM = 1024
BOTTLENECK_DIM = 256
SIGMA_DIM = 64

# ============================================================
# LOSS
#
# L = LM loss
#   + lambda_corr * ||correction||^2
#   + lambda_preserve * hinge_preservation_loss
#
# The key change:
#
#   preserve_loss = max(0, rho_min - rho)^2
# ============================================================

LAMBDA_CORRECTION = 1e-3
LAMBDA_PRESERVE = 0.5
RHO_MIN = 0.50

LOG_EVERY = 25
VAL_EVERY = 100
VAL_BATCHES = 20

CHECKPOINT_PATH = "gated_gaussian_repair_block7_to_block9.pt"
HISTORY_CSV = "gated_gaussian_repair_block7_to_block9_history.csv"

PLOT_LOSS = "gated_gaussian_repair_block7_to_block9_loss.png"
PLOT_PPL = "gated_gaussian_repair_block7_to_block9_ppl.png"



print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)

model.eval()


for p in model.parameters():
    p.requires_grad_(False)

print("GPT-2 loaded and frozen.")
print("noise at:", PERTURB_HOOK)
print("repair at:", REPAIR_HOOK)



print("\nLoading WikiText...")

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
    TRAIN_DOC_FRACTION
    *
    len(documents)
)

train_docs = documents[:split_idx]
val_docs = documents[split_idx:]

print("documents:", len(documents))
print("train:", len(train_docs))
print("val:", len(val_docs))



@torch.no_grad()
def build_chunks(
    docs,
    max_chunks,
    name
):
    chunks = []

    for text in docs:
        tokens = model.to_tokens(
            text,
            prepend_bos=True
        )[0].cpu()

        if tokens.numel() < MAX_SEQ_LEN:
            continue

        start = 0

        while (
            start + MAX_SEQ_LEN
            <= tokens.numel()
        ):
            chunks.append(
                tokens[
                    start:
                    start + MAX_SEQ_LEN
                ]
            )

            if len(chunks) >= max_chunks:
                out = torch.stack(chunks)
                print(name, "chunks:", out.shape)
                return out

            start += MAX_SEQ_LEN

    if not chunks:
        raise RuntimeError(
            f"No {name} chunks created."
        )

    out = torch.stack(chunks)
    print(name, "chunks:", out.shape)

    return out


train_tokens = build_chunks(
    train_docs,
    max_chunks=4000,
    name="train"
)

val_tokens = build_chunks(
    val_docs,
    max_chunks=500,
    name="validation"
)




class SigmaEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        # sigma shape: [B, T, 1]

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
            frequencies.view(
                1,
                1,
                -1
            )
        )

        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=-1
        )




class GatedRepair(nn.Module):

    def __init__(
        self,
        d_model=768,
        hidden_dim=1024,
        bottleneck_dim=256,
        sigma_dim=64,
    ):
        super().__init__()

        self.sigma_embedding = SigmaEmbedding(
            sigma_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                d_model + sigma_dim,
                hidden_dim
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                bottleneck_dim
            ),
            nn.GELU(),
        )

        self.correction_head = nn.Linear(
            bottleneck_dim,
            d_model
        )

        self.gate_head = nn.Linear(
            bottleneck_dim,
            1
        )

        # Start approximately as identity.
        nn.init.zeros_(
            self.correction_head.weight
        )
        nn.init.zeros_(
            self.correction_head.bias
        )

        nn.init.zeros_(
            self.gate_head.weight
        )
        nn.init.constant_(
            self.gate_head.bias,
            -2.0
        )

    def forward(
        self,
        x,
        sigma
    ):
        sigma_emb = self.sigma_embedding(
            sigma
        )

        z = self.encoder(
            torch.cat(
                [x, sigma_emb],
                dim=-1
            )
        )

        correction_raw = (
            self.correction_head(z)
        )

        gate = torch.sigmoid(
            self.gate_head(z)
        )

        correction = (
            gate
            *
            correction_raw
        )

        repaired = (
            x
            +
            correction
        )

        return (
            repaired,
            correction,
            gate
        )


repair = GatedRepair(
    d_model=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    bottleneck_dim=BOTTLENECK_DIM,
    sigma_dim=SIGMA_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    repair.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)




def sample_gaussian_corruption(
    batch_size,
    seq_len,
    device,
    dtype
):

    log_r = torch.empty(
        batch_size,
        1,
        1,
        device=device,
        dtype=dtype
    ).uniform_(
        math.log(MIN_NOISE_NORM),
        math.log(MAX_NOISE_NORM)
    )

    r = torch.exp(
        log_r
    )

    sigma = (
        r
        /
        math.sqrt(D_MODEL)
    )

    epsilon = (
        torch.randn(
            batch_size,
            seq_len,
            D_MODEL,
            device=device,
            dtype=dtype
        )
        *
        sigma
    )

    return (
        epsilon,
        sigma,
        r
    )




@torch.no_grad()
def get_clean_h9(
    tokens
):
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[
            REPAIR_HOOK
        ]
    )

    return (
        cache[
            REPAIR_HOOK
        ]
        .detach()
    )




def repaired_forward(
    tokens,
    epsilon,
    sigma_per_sequence,
    clean_h9
):
    aux = {}

    def perturb_hook(
        resid,
        hook
    ):
        return (
            resid
            +
            epsilon.to(
                device=resid.device,
                dtype=resid.dtype
            )
        )

    def repair_hook(
        resid,
        hook
    ):
        B, T, D = resid.shape

        sigma = (
            sigma_per_sequence
            .to(
                device=resid.device,
                dtype=resid.dtype
            )
            .expand(
                B,
                T,
                1
            )
        )

        repaired, correction, gate = repair(
            resid,
            sigma
        )

        clean = clean_h9.to(
            device=resid.device,
            dtype=resid.dtype
        )

        
        raw_delta = (
            resid
            -
            clean
        ).detach()

        
        repaired_delta = (
            repaired
            -
            clean
        )

        raw_norm_sq = (
            raw_delta
            .pow(2)
            .sum(dim=-1)
            .clamp_min(1e-8)
        )

        rho = (
            (
                repaired_delta
                *
                raw_delta
            )
            .sum(dim=-1)
            /
            raw_norm_sq
        )


        preserve_loss = (
            F.relu(
                RHO_MIN
                -
                rho
            )
            .pow(2)
            .mean()
        )

        correction_mse = (
            correction
            .pow(2)
            .mean()
        )

        aux[
            "preserve_loss"
        ] = preserve_loss

        aux[
            "correction_mse"
        ] = correction_mse

        aux[
            "gate_mean"
        ] = gate.mean()

        aux[
            "rho_mean"
        ] = rho.mean()

        aux[
            "rho_min_batch"
        ] = rho.min()

        return repaired

    with model.hooks(
        fwd_hooks=[
            (
                PERTURB_HOOK,
                perturb_hook
            ),
            (
                REPAIR_HOOK,
                repair_hook
            )
        ]
    ):
        logits = model(
            tokens
        )

    return (
        logits,
        aux
    )




@torch.no_grad()
def raw_perturbed_forward(
    tokens,
    epsilon
):
    def perturb_hook(
        resid,
        hook
    ):
        return (
            resid
            +
            epsilon.to(
                device=resid.device,
                dtype=resid.dtype
            )
        )

    with model.hooks(
        fwd_hooks=[
            (
                PERTURB_HOOK,
                perturb_hook
            )
        ]
    ):
        logits = model(
            tokens
        )

    return logits




def lm_ce(
    logits,
    tokens
):
    return F.cross_entropy(
        logits[
            :,
            :-1,
            :
        ].reshape(
            -1,
            logits.shape[-1]
        ),
        tokens[
            :,
            1:
        ].reshape(-1)
    )



val_generator = torch.Generator(
    device="cpu"
)

val_generator.manual_seed(
    SEED + 999
)

val_uniform = torch.rand(
    val_tokens.shape[0],
    1,
    1,
    generator=val_generator
)

val_log_r = (
    math.log(MIN_NOISE_NORM)
    +
    val_uniform
    *
    (
        math.log(MAX_NOISE_NORM)
        -
        math.log(MIN_NOISE_NORM)
    )
)

val_r = torch.exp(
    val_log_r
)

val_sigma = (
    val_r
    /
    math.sqrt(D_MODEL)
)

val_noise = (
    torch.randn(
        val_tokens.shape[0],
        MAX_SEQ_LEN,
        D_MODEL,
        generator=val_generator
    )
    *
    val_sigma
)




@torch.no_grad()
def evaluate_validation():
    repair.eval()

    clean_losses = []
    raw_losses = []
    repaired_losses = []

    gates = []
    rhos = []

    n_batches = min(
        VAL_BATCHES,
        math.ceil(
            val_tokens.shape[0]
            /
            BATCH_SIZE
        )
    )

    for batch_idx in range(
        n_batches
    ):
        start = (
            batch_idx
            *
            BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            val_tokens.shape[0]
        )

        tokens = (
            val_tokens[
                start:end
            ]
            .to(DEVICE)
        )

        epsilon = (
            val_noise[
                start:end
            ]
            .to(
                device=DEVICE,
                dtype=torch.float32
            )
        )

        sigma = (
            val_sigma[
                start:end
            ]
            .to(
                device=DEVICE,
                dtype=torch.float32
            )
        )

        clean_logits, cache = (
            model.run_with_cache(
                tokens,
                names_filter=[
                    REPAIR_HOOK
                ]
            )
        )

        clean_h9 = (
            cache[
                REPAIR_HOOK
            ]
            .detach()
        )

        clean_loss = lm_ce(
            clean_logits,
            tokens
        )

        raw_logits = (
            raw_perturbed_forward(
                tokens,
                epsilon
            )
        )

        raw_loss = lm_ce(
            raw_logits,
            tokens
        )

        repaired_logits, aux = (
            repaired_forward(
                tokens,
                epsilon,
                sigma,
                clean_h9
            )
        )

        repaired_loss = lm_ce(
            repaired_logits,
            tokens
        )

        clean_losses.append(
            clean_loss.item()
        )

        raw_losses.append(
            raw_loss.item()
        )

        repaired_losses.append(
            repaired_loss.item()
        )

        gates.append(
            aux[
                "gate_mean"
            ].item()
        )

        rhos.append(
            aux[
                "rho_mean"
            ].item()
        )

    clean_ce = (
        sum(clean_losses)
        /
        len(clean_losses)
    )

    raw_ce = (
        sum(raw_losses)
        /
        len(raw_losses)
    )

    repaired_ce = (
        sum(repaired_losses)
        /
        len(repaired_losses)
    )

    return {
        "clean_ce":
            clean_ce,

        "raw_ce":
            raw_ce,

        "repaired_ce":
            repaired_ce,

        "clean_ppl":
            math.exp(clean_ce),

        "raw_ppl":
            math.exp(raw_ce),

        "repaired_ppl":
            math.exp(repaired_ce),

        "gate":
            sum(gates)
            /
            len(gates),

        "rho":
            sum(rhos)
            /
            len(rhos),
    }




history = []

print(
    "\nStarting training..."
)

print(
    "Loss = LM + correction penalty + hinge preservation"
)

for step in range(
    1,
    N_STEPS + 1
):
    repair.train()

    indices = torch.randint(
        0,
        train_tokens.shape[0],
        (BATCH_SIZE,)
    )

    tokens = (
        train_tokens[
            indices
        ]
        .to(DEVICE)
    )

    clean_h9 = get_clean_h9(
        tokens
    )

    epsilon, sigma, r = (
        sample_gaussian_corruption(
            batch_size=
                tokens.shape[0],

            seq_len=
                tokens.shape[1],

            device=
                DEVICE,

            dtype=
                torch.float32
        )
    )

    logits, aux = repaired_forward(
        tokens,
        epsilon,
        sigma,
        clean_h9
    )

    lm_loss = lm_ce(
        logits,
        tokens
    )

    total_loss = (
        lm_loss
        +
        LAMBDA_CORRECTION
        *
        aux[
            "correction_mse"
        ]
        +
        LAMBDA_PRESERVE
        *
        aux[
            "preserve_loss"
        ]
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        repair.parameters(),
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
            f"total={total_loss.item():.4f} | "
            f"LM={lm_loss.item():.4f} | "
            f"pres={aux['preserve_loss'].item():.4f} | "
            f"corr={aux['correction_mse'].item():.6f} | "
            f"gate={aux['gate_mean'].item():.3f} | "
            f"rho={aux['rho_mean'].item():.3f} | "
            f"mean_r={r.mean().item():.2f}"
        )

    if (
        step == 1
        or
        step % VAL_EVERY == 0
        or
        step == N_STEPS
    ):
        metrics = evaluate_validation()

        row = {
            "step":
                step,

            "train_total_loss":
                total_loss.item(),

            "train_lm_loss":
                lm_loss.item(),

            "train_preserve_loss":
                aux[
                    "preserve_loss"
                ].item(),

            "train_correction_mse":
                aux[
                    "correction_mse"
                ].item(),

            "train_gate":
                aux[
                    "gate_mean"
                ].item(),

            "train_rho":
                aux[
                    "rho_mean"
                ].item(),

            **metrics,
        }

        history.append(
            row
        )

        print(
            "\nVALIDATION"
        )

        print(
            f"clean PPL={metrics['clean_ppl']:.3f} | "
            f"raw noisy PPL={metrics['raw_ppl']:.3f} | "
            f"repaired PPL={metrics['repaired_ppl']:.3f}"
        )

        print(
            f"gate={metrics['gate']:.3f} | "
            f"retained projection rho="
            f"{metrics['rho']:.3f}\n"
        )



checkpoint = {
    "state_dict":
        repair.state_dict(),

    "d_model":
        D_MODEL,

    "hidden_dim":
        HIDDEN_DIM,

    "bottleneck_dim":
        BOTTLENECK_DIM,

    "sigma_dim":
        SIGMA_DIM,

    "perturb_hook":
        PERTURB_HOOK,

    "repair_hook":
        REPAIR_HOOK,

    "min_noise_norm":
        MIN_NOISE_NORM,

    "max_noise_norm":
        MAX_NOISE_NORM,

    "lambda_correction":
        LAMBDA_CORRECTION,

    "lambda_preserve":
        LAMBDA_PRESERVE,

    "rho_min":
        RHO_MIN,

    "corruption":
        "isotropic Gaussian epsilon ~ N(0, sigma^2 I), independent per token",

    "objective":
        "LM CE + correction penalty + hinge lower-bound preservation",

    "uses_target_steering_vector":
        False,

    "training_steps":
        N_STEPS,

    "seed":
        SEED,
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

print(
    "\nSaved checkpoint:",
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

        writer.writerows(
            history
        )




if history:
    steps = [
        row["step"]
        for row in history
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        steps,
        [
            row["clean_ce"]
            for row in history
        ],
        label="Clean CE"
    )

    plt.plot(
        steps,
        [
            row["raw_ce"]
            for row in history
        ],
        label="Raw noisy CE"
    )

    plt.plot(
        steps,
        [
            row["repaired_ce"]
            for row in history
        ],
        label="Repaired CE"
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Cross entropy"
    )

    plt.title(
        "Block 7 -> block 9 gated Gaussian repair"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_LOSS,
        dpi=200
    )

    plt.show()


    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        steps,
        [
            row["clean_ppl"]
            for row in history
        ],
        label="Clean PPL"
    )

    plt.plot(
        steps,
        [
            row["raw_ppl"]
            for row in history
        ],
        label="Raw noisy PPL"
    )

    plt.plot(
        steps,
        [
            row["repaired_ppl"]
            for row in history
        ],
        label="Repaired PPL"
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Perplexity"
    )

    plt.title(
        "Validation language quality"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_PPL,
        dpi=200
    )

    plt.show()



final = evaluate_validation()

print(
    "\n"
    +
    "=" * 80
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 80
)

print(
    f"clean PPL:      {final['clean_ppl']:.4f}"
)

print(
    f"raw noisy PPL:  {final['raw_ppl']:.4f}"
)

print(
    f"repaired PPL:   {final['repaired_ppl']:.4f}"
)

print(
    f"mean gate:      {final['gate']:.4f}"
)

print(
    f"retained rho:   {final['rho']:.4f}"
)
