import csv
import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from sae_lens import SAE



DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

CHECKPOINT_PATH = "improved_interp_denoiser_block7.pt"

# Human counting: 7th GPT-2 transformer block.
HOOK_NAME = "blocks.6.hook_resid_pre"

FEATURE_IDX = 17363
D_MODEL = 768

ALPHAS = [0, 4, 8, 16, 24, 32, 40, 48, 64]

PROMPTS = [
    "The article discusses",
    "Recent developments show that",
    "The report describes",
    "Researchers are interested in",
    "One important question is",
    "The following text explains",
]

SEEDS = [1, 2, 3, 4, 5]

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8
TOP_P = 0.95

GENERATIONS_CSV = "improved_interp_block7_generations.csv"
SUMMARY_CSV = "improved_interp_block7_summary.csv"

PLOT_CONCEPT = "improved_interp_block7_concept.png"
PLOT_PPL = "improved_interp_block7_ppl.png"
PLOT_PARETO = "improved_interp_block7_pareto.png"
PLOT_MAPPING = "improved_interp_block7_alpha_to_t.png"




print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE,
)
model.eval()

print("GPT-2 loaded.")




print("\nLoading denoiser checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

required_keys = [
    "state_dict",
    "hook_name",
    "d_model",
    "hidden_dim",
    "t_embed_dim",
    "t_min",
    "t_max",
    "train_mean_h_norm",
]

for key in required_keys:
    if key not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain required key: {key}"
        )

if checkpoint["hook_name"] != HOOK_NAME:
    raise ValueError(
        "Layer mismatch:\n"
        f"checkpoint hook = {checkpoint['hook_name']}\n"
        f"test hook       = {HOOK_NAME}"
    )

if checkpoint["d_model"] != D_MODEL:
    raise ValueError(
        f"d_model mismatch: checkpoint={checkpoint['d_model']}, "
        f"expected={D_MODEL}"
    )

T_MIN = float(checkpoint["t_min"])
T_MAX = float(checkpoint["t_max"])
TRAIN_MEAN_H_NORM = float(
    checkpoint["train_mean_h_norm"]
)

print("checkpoint layer:", checkpoint["hook_name"])
print("training t range:", T_MIN, "to", T_MAX)
print("training mean ||h_7||:", TRAIN_MEAN_H_NORM)
print("architecture:", checkpoint.get("architecture", "not stored"))
print("objective:", checkpoint.get("objective", "not stored"))




class TEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [N, 1]

        half = self.dim // 2

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=t.device,
                dtype=t.dtype,
            )
        )

        angles = (
            t
            *
            frequencies.view(1, -1)
        )

        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )


class ImprovedInterpolationDenoiser(nn.Module):

    def __init__(
        self,
        d_model=768,
        hidden_dim=1536,
        t_embed_dim=64,
    ):
        super().__init__()

        self.t_embedding = TEmbedding(
            t_embed_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                d_model + t_embed_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                d_model,
            ),
        )

    def forward(self, x, t):
        t_emb = self.t_embedding(t)

        raw_correction = self.net(
            torch.cat(
                [x, t_emb],
                dim=-1,
            )
        )

        
        correction = (
            (1.0 - t)
            *
            raw_correction
        )

        return x + correction


denoiser = ImprovedInterpolationDenoiser(
    d_model=checkpoint["d_model"],
    hidden_dim=checkpoint["hidden_dim"],
    t_embed_dim=checkpoint["t_embed_dim"],
).to(DEVICE)

denoiser.load_state_dict(
    checkpoint["state_dict"]
)
denoiser.eval()

print("Denoiser loaded.")




print("\nLoading block-7 SAE...")

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=HOOK_NAME,
    device=DEVICE,
)
sae.eval()

v = sae.W_dec[
    FEATURE_IDX
].detach()

V_NORM = float(
    v.norm().item()
)

print("feature:", FEATURE_IDX)
print("||v||:", V_NORM)



def alpha_to_t(alpha):


    if alpha == 0:
        return 1.0

    steering_norm = (
        abs(float(alpha))
        *
        V_NORM
    )

    q = (
        steering_norm
        /
        max(
            TRAIN_MEAN_H_NORM,
            1e-12,
        )
    )

    t = 1.0 / (1.0 + q)

    t = max(
        T_MIN,
        min(T_MAX, t),
    )

    return float(t)


print("\nalpha -> t mapping")
print("-" * 40)

for alpha in ALPHAS:
    print(
        f"alpha={alpha:>3}  "
        f"||alpha v||={abs(alpha) * V_NORM:8.3f}  "
        f"t={alpha_to_t(alpha):.4f}"
    )




def make_intervention_hook(
    alpha,
    method,
):


    def hook(
        resid,
        hook,
    ):
        
        if alpha == 0:
            return resid

        vv = v.to(
            device=resid.device,
            dtype=resid.dtype,
        )

        steered = (
            resid
            +
            alpha * vv
        )

        if method == "raw":
            return steered

        if method != "denoised":
            raise ValueError(
                f"Unknown method: {method}"
            )

        B, T, D = steered.shape

        t_value = alpha_to_t(
            alpha
        )

        t_tensor = torch.full(
            (B * T, 1),
            t_value,
            device=steered.device,
            dtype=steered.dtype,
        )

        flat = steered.reshape(
            B * T,
            D,
        )

        repaired_flat = denoiser(
            flat,
            t_tensor,
        )

        repaired = repaired_flat.reshape(
            B,
            T,
            D,
        )

        return repaired.to(
            dtype=resid.dtype
        )

    return hook




@torch.no_grad()
def generate_text(
    prompt,
    alpha,
    seed,
    method,
):

    torch.manual_seed(seed)

    with model.hooks(
        fwd_hooks=[
            (
                HOOK_NAME,
                make_intervention_hook(
                    alpha,
                    method,
                ),
            )
        ]
    ):
        text = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            verbose=False,
        )

    return text



@torch.no_grad()
def concept_score(text):


    tokens = model.to_tokens(
        text
    )

    _, cache = model.run_with_cache(
        tokens,
        names_filter=[HOOK_NAME],
    )

    h = cache[HOOK_NAME]

    z = sae.encode(h)

    values = z[
        0,
        :,
        FEATURE_IDX,
    ]

    return float(
        values.max().item()
    )



@torch.no_grad()
def completion_perplexity(
    prompt,
    full_text,
):


    full_tokens = model.to_tokens(
        full_text
    )

    prompt_tokens = model.to_tokens(
        prompt
    )

    prompt_len = int(
        prompt_tokens.shape[1]
    )

    if full_tokens.shape[1] <= prompt_len:
        return float("nan")

    input_tokens = full_tokens[:, :-1]
    targets = full_tokens[:, 1:]

    logits = model(
        input_tokens
    )

    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    token_log_probs = (
        log_probs
        .gather(
            -1,
            targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    
    start = max(
        prompt_len - 1,
        0,
    )

    completion_log_probs = (
        token_log_probs[
            :,
            start:
        ]
    )

    if completion_log_probs.numel() == 0:
        return float("nan")

    nll = (
        -completion_log_probs.mean()
    )

    return float(
        torch.exp(nll).item()
    )



def distinct_n(
    text,
    n,
):
    words = (
        text
        .lower()
        .split()
    )

    if len(words) < n:
        return 0.0

    ngrams = [
        tuple(
            words[i:i+n]
        )
        for i in range(
            len(words) - n + 1
        )
    ]

    return (
        len(set(ngrams))
        /
        len(ngrams)
    )




METHODS = [
    "raw",
    "denoised",
]

rows = []

total = (
    len(METHODS)
    *
    len(ALPHAS)
    *
    len(PROMPTS)
    *
    len(SEEDS)
)

counter = 0

for method in METHODS:

    for alpha in ALPHAS:

        for prompt in PROMPTS:

            for seed in SEEDS:

                counter += 1

                print(
                    f"\n[{counter}/{total}] "
                    f"{method} | "
                    f"alpha={alpha} | "
                    f"t={alpha_to_t(alpha):.4f} | "
                    f"seed={seed}"
                )

                text = generate_text(
                    prompt=prompt,
                    alpha=alpha,
                    seed=seed,
                    method=method,
                )

                concept = concept_score(
                    text
                )

                ppl = completion_perplexity(
                    prompt,
                    text,
                )

                d1 = distinct_n(
                    text,
                    1,
                )

                d2 = distinct_n(
                    text,
                    2,
                )

                d3 = distinct_n(
                    text,
                    3,
                )

                row = {
                    "method": method,
                    "alpha": alpha,
                    "t_condition": alpha_to_t(alpha),
                    "steering_norm": abs(alpha) * V_NORM,
                    "prompt": prompt,
                    "seed": seed,
                    "concept": concept,
                    "ppl_completion": ppl,
                    "dist1": d1,
                    "dist2": d2,
                    "dist3": d3,
                    "rep3": 1.0 - d3,
                    "text": text,
                }

                rows.append(row)

                print(
                    f"concept={concept:.4f} | "
                    f"PPL={ppl:.4f} | "
                    f"dist3={d3:.4f}"
                )




with open(
    GENERATIONS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)

print(
    "\nSaved:",
    GENERATIONS_CSV,
)




def finite_values(
    selected,
    key,
):
    values = []

    for row in selected:
        value = row[key]

        if (
            isinstance(
                value,
                (int, float),
            )
            and
            math.isfinite(value)
        ):
            values.append(value)

    return values


def mean_metric(
    selected,
    key,
):
    values = finite_values(
        selected,
        key,
    )

    if not values:
        return float("nan")

    return (
        sum(values)
        /
        len(values)
    )


def std_metric(
    selected,
    key,
):
    values = finite_values(
        selected,
        key,
    )

    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )




summary = []

for method in METHODS:

    for alpha in ALPHAS:

        selected = [
            row
            for row in rows
            if (
                row["method"] == method
                and
                row["alpha"] == alpha
            )
        ]

        summary.append(
            {
                "method":
                    method,

                "alpha":
                    alpha,

                "t_condition":
                    alpha_to_t(alpha),

                "steering_norm":
                    abs(alpha) * V_NORM,

                "concept_mean":
                    mean_metric(
                        selected,
                        "concept",
                    ),

                "concept_std":
                    std_metric(
                        selected,
                        "concept",
                    ),

                "ppl_mean":
                    mean_metric(
                        selected,
                        "ppl_completion",
                    ),

                "ppl_std":
                    std_metric(
                        selected,
                        "ppl_completion",
                    ),

                "dist1":
                    mean_metric(
                        selected,
                        "dist1",
                    ),

                "dist2":
                    mean_metric(
                        selected,
                        "dist2",
                    ),

                "dist3":
                    mean_metric(
                        selected,
                        "dist3",
                    ),

                "rep3":
                    mean_metric(
                        selected,
                        "rep3",
                    ),
            }
        )


with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(summary)

print(
    "Saved:",
    SUMMARY_CSV,
)




print(
    "\n"
    +
    "=" * 132
)

print(
    "IMPROVED INTERPOLATION DENOISER — BLOCK 7"
)

print(
    "=" * 132
)

print(
    f"{'method':>10} | "
    f"{'alpha':>5} | "
    f"{'t':>7} | "
    f"{'concept':>10} | "
    f"{'PPL(comp)':>12} | "
    f"{'dist1':>8} | "
    f"{'dist2':>8} | "
    f"{'dist3':>8} | "
    f"{'rep3':>8}"
)

print(
    "-" * 132
)

for row in summary:

    print(
        f"{row['method']:>10} | "
        f"{row['alpha']:5.1f} | "
        f"{row['t_condition']:7.4f} | "
        f"{row['concept_mean']:10.4f} | "
        f"{row['ppl_mean']:12.4f} | "
        f"{row['dist1']:8.4f} | "
        f"{row['dist2']:8.4f} | "
        f"{row['dist3']:8.4f} | "
        f"{row['rep3']:8.4f}"
    )




print(
    "\n"
    +
    "=" * 125
)

print(
    "MATCHED-ALPHA EFFECT"
)

print(
    "=" * 125
)

for alpha in ALPHAS:

    raw = next(
        row
        for row in summary
        if (
            row["method"] == "raw"
            and
            row["alpha"] == alpha
        )
    )

    den = next(
        row
        for row in summary
        if (
            row["method"] == "denoised"
            and
            row["alpha"] == alpha
        )
    )

    delta_concept = (
        den["concept_mean"]
        -
        raw["concept_mean"]
    )

    delta_ppl = (
        den["ppl_mean"]
        -
        raw["ppl_mean"]
    )

    ppl_ratio = (
        den["ppl_mean"]
        /
        raw["ppl_mean"]
    )

    if raw["concept_mean"] > 1e-8:
        concept_ratio = (
            den["concept_mean"]
            /
            raw["concept_mean"]
        )
    else:
        concept_ratio = float("nan")

    # Conservative matched-alpha label.
    if (
        raw["concept_mean"] > 1e-8
        and
        concept_ratio >= 0.80
        and
        ppl_ratio < 1.0
    ):
        verdict = "PROMISING"

    elif (
        raw["concept_mean"] <= 1e-8
        and
        ppl_ratio < 1.0
    ):
        verdict = (
            "QUALITY RECOVERY ONLY "
            "(RAW CONCEPT ~ 0)"
        )

    elif (
        ppl_ratio < 1.0
    ):
        verdict = (
            "PPL BETTER, "
            "CONCEPT NOT PRESERVED"
        )

    elif (
        raw["concept_mean"] > 1e-8
        and
        concept_ratio >= 0.80
    ):
        verdict = (
            "CONCEPT PRESERVED, "
            "PPL NOT BETTER"
        )

    else:
        verdict = "NO IMPROVEMENT"

    print(
        f"alpha={alpha:>3} | "
        f"t={alpha_to_t(alpha):.4f} | "
        f"Δconcept={delta_concept:+.4f} | "
        f"ΔPPL={delta_ppl:+.4f} | "
        f"concept ratio={concept_ratio:7.3f} | "
        f"PPL ratio={ppl_ratio:7.3f} | "
        f"{verdict}"
    )




def curve(
    method,
    key,
):
    return [
        next(
            row[key]
            for row in summary
            if (
                row["method"] == method
                and
                row["alpha"] == alpha
            )
        )
        for alpha in ALPHAS
    ]


raw_concept = curve(
    "raw",
    "concept_mean",
)

den_concept = curve(
    "denoised",
    "concept_mean",
)

raw_ppl = curve(
    "raw",
    "ppl_mean",
)

den_ppl = curve(
    "denoised",
    "ppl_mean",
)




plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    raw_concept,
    marker="o",
    label="Raw steering",
)

plt.plot(
    ALPHAS,
    den_concept,
    marker="o",
    label="Steering + interpolation denoiser",
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Concept score"
)

plt.title(
    "Block 7: concept preservation"
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_CONCEPT,
    dpi=200,
)

plt.show()




plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    raw_ppl,
    marker="o",
    label="Raw steering",
)

plt.plot(
    ALPHAS,
    den_ppl,
    marker="o",
    label="Steering + interpolation denoiser",
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Completion perplexity"
)

plt.title(
    "Block 7: generation quality"
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_PPL,
    dpi=200,
)

plt.show()




plt.figure(
    figsize=(9, 7)
)

plt.plot(
    raw_ppl,
    raw_concept,
    marker="o",
    label="Raw steering",
)

plt.plot(
    den_ppl,
    den_concept,
    marker="o",
    label="Interpolation denoiser",
)

for alpha, x, y in zip(
    ALPHAS,
    raw_ppl,
    raw_concept,
):
    plt.annotate(
        f"r{alpha}",
        (x, y),
    )

for alpha, x, y in zip(
    ALPHAS,
    den_ppl,
    den_concept,
):
    plt.annotate(
        f"d{alpha}",
        (x, y),
    )

plt.xlabel(
    "Completion PPL (lower is better)"
)

plt.ylabel(
    "Concept score (higher is better)"
)

plt.title(
    "Block 7: quality–concept Pareto comparison"
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_PARETO,
    dpi=200,
)

plt.show()



plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    [
        alpha_to_t(alpha)
        for alpha in ALPHAS
    ],
    marker="o",
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Denoiser conditioning t"
)

plt.title(
    "Fixed alpha-to-t scale mapping"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_MAPPING,
    dpi=200,
)

plt.show()



print("\nEvaluation finished.")
print("Saved:", GENERATIONS_CSV)
print("Saved:", SUMMARY_CSV)
print("Saved:", PLOT_CONCEPT)
print("Saved:", PLOT_PPL)
print("Saved:", PLOT_PARETO)
print("Saved:", PLOT_MAPPING)
