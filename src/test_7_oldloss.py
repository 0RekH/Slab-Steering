import math
import csv
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from sae_lens import SAE



DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

CHECKPOINT_PATH = "baseline_denoiser_block7.pt"

HOOK_NAME = "blocks.6.hook_resid_pre"

FEATURE_IDX = 17363

D_MODEL = 768

ALPHAS = [
    0,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    64,
]

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

ALL_CSV = "baseline_block7_raw_vs_denoised_generations.csv"
SUMMARY_CSV = "baseline_block7_raw_vs_denoised_summary.csv"

PLOT_CONCEPT = "baseline_block7_concept.png"
PLOT_PPL = "baseline_block7_ppl.png"
PLOT_PARETO = "baseline_block7_pareto.png"




model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)
model.eval()




checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if checkpoint["hook_name"] != HOOK_NAME:
    raise ValueError(
        f"Denoiser was trained on "
        f"{checkpoint['hook_name']}, "
        f"but test uses {HOOK_NAME}."
    )




class SigmaEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
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
    d_model=checkpoint["d_model"],
    hidden_dim=checkpoint["hidden_dim"],
    sigma_dim=checkpoint["sigma_dim"]
).to(DEVICE)

denoiser.load_state_dict(
    checkpoint["state_dict"]
)

denoiser.eval()

print("Denoiser loaded.")




sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=HOOK_NAME,
    device=DEVICE
)
sae.eval()

v = sae.W_dec[
    FEATURE_IDX
].detach()

V_NORM = v.norm().item()

print("feature:", FEATURE_IDX)
print("||v||:", V_NORM)




def make_intervention_hook(
    alpha,
    method
):

    def hook(
        resid,
        hook
    ):
        if alpha == 0:
            
            return resid

        vv = v.to(
            device=resid.device,
            dtype=resid.dtype
        )

        steered = (
            resid
            +
            alpha
            *
            vv
        )

        if method == "raw":
            return steered

        if method == "denoised":
            B, T, D = steered.shape


            r_alpha = (
                abs(alpha)
                *
                V_NORM
            )

            sigma_value = (
                r_alpha
                /
                math.sqrt(D_MODEL)
            )

            sigma = torch.full(
                (B * T, 1),
                sigma_value,
                device=steered.device,
                dtype=steered.dtype
            )

            flat = steered.reshape(
                B * T,
                D
            )

            repaired_flat = denoiser(
                flat,
                sigma
            )

            return repaired_flat.reshape(
                B,
                T,
                D
            ).to(resid.dtype)

        raise ValueError(method)

    return hook




@torch.no_grad()
def generate_text(
    prompt,
    alpha,
    seed,
    method
):
    torch.manual_seed(seed)

    with model.hooks(
        fwd_hooks=[
            (
                HOOK_NAME,
                make_intervention_hook(
                    alpha,
                    method
                )
            )
        ]
    ):
        text = model.generate(
            prompt,
            max_new_tokens=
                MAX_NEW_TOKENS,
            temperature=
                TEMPERATURE,
            top_p=
                TOP_P,
            do_sample=True,
            verbose=False
        )

    return text




@torch.no_grad()
def concept_score(text):

    tokens = model.to_tokens(text)

    _, cache = model.run_with_cache(
        tokens,
        names_filter=[HOOK_NAME]
    )

    h = cache[HOOK_NAME]

    z = sae.encode(h)

    return (
        z[
            0,
            :,
            FEATURE_IDX
        ]
        .max()
        .item()
    )



@torch.no_grad()
def completion_perplexity(
    prompt,
    full_text
):
    # Also evaluated with CLEAN GPT-2.
    full_tokens = model.to_tokens(
        full_text
    )

    prompt_tokens = model.to_tokens(
        prompt
    )

    prompt_len = prompt_tokens.shape[1]

    if (
        full_tokens.shape[1]
        <= prompt_len
    ):
        return float("nan")

    logits = model(
        full_tokens[:, :-1]
    )

    targets = full_tokens[:, 1:]

    log_probs = F.log_softmax(
        logits,
        dim=-1
    )

    target_log_probs = (
        log_probs
        .gather(
            -1,
            targets.unsqueeze(-1)
        )
        .squeeze(-1)
    )

    start = max(
        prompt_len - 1,
        0
    )

    completion_log_probs = (
        target_log_probs[:, start:]
    )

    nll = (
        -completion_log_probs.mean()
    )

    return (
        torch.exp(nll)
        .item()
    )




def distinct_n(text, n):
    words = text.lower().split()

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
                    f"seed={seed}"
                )

                text = generate_text(
                    prompt,
                    alpha,
                    seed,
                    method
                )

                concept = concept_score(
                    text
                )

                ppl = completion_perplexity(
                    prompt,
                    text
                )

                d1 = distinct_n(
                    text,
                    1
                )

                d2 = distinct_n(
                    text,
                    2
                )

                d3 = distinct_n(
                    text,
                    3
                )

                rows.append(
                    {
                        "method":
                            method,

                        "alpha":
                            alpha,

                        "prompt":
                            prompt,

                        "seed":
                            seed,

                        "concept":
                            concept,

                        "ppl":
                            ppl,

                        "dist1":
                            d1,

                        "dist2":
                            d2,

                        "dist3":
                            d3,

                        "rep3":
                            1.0 - d3,

                        "text":
                            text,
                    }
                )

                print(
                    f"concept={concept:.4f} | "
                    f"PPL={ppl:.4f} | "
                    f"dist3={d3:.4f}"
                )




with open(
    ALL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        )
    )
    writer.writeheader()
    writer.writerows(rows)




def finite_values(
    selected,
    key
):
    return [
        row[key]
        for row in selected
        if (
            isinstance(
                row[key],
                (int, float)
            )
            and
            math.isfinite(
                row[key]
            )
        )
    ]


def mean_metric(
    selected,
    key
):
    x = finite_values(
        selected,
        key
    )

    return (
        sum(x) / len(x)
        if x
        else float("nan")
    )


def std_metric(
    selected,
    key
):
    x = finite_values(
        selected,
        key
    )

    if len(x) < 2:
        return 0.0

    return statistics.stdev(x)


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

                "concept":
                    mean_metric(
                        selected,
                        "concept"
                    ),

                "concept_std":
                    std_metric(
                        selected,
                        "concept"
                    ),

                "ppl":
                    mean_metric(
                        selected,
                        "ppl"
                    ),

                "ppl_std":
                    std_metric(
                        selected,
                        "ppl"
                    ),

                "dist1":
                    mean_metric(
                        selected,
                        "dist1"
                    ),

                "dist2":
                    mean_metric(
                        selected,
                        "dist2"
                    ),

                "dist3":
                    mean_metric(
                        selected,
                        "dist3"
                    ),

                "rep3":
                    mean_metric(
                        selected,
                        "rep3"
                    ),
            }
        )


with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary[0].keys()
        )
    )
    writer.writeheader()
    writer.writerows(summary)



print(
    "\n"
    +
    "=" * 120
)

print(
    "BLOCK-7 SAME-LAYER BASELINE: RAW vs DENOISED"
)

print(
    "=" * 120
)

print(
    f"{'method':>10} | "
    f"{'alpha':>6} | "
    f"{'concept':>10} | "
    f"{'PPL':>12} | "
    f"{'dist1':>8} | "
    f"{'dist2':>8} | "
    f"{'dist3':>8} | "
    f"{'rep3':>8}"
)

print(
    "-" * 120
)

for row in summary:
    print(
        f"{row['method']:>10} | "
        f"{row['alpha']:6.1f} | "
        f"{row['concept']:10.4f} | "
        f"{row['ppl']:12.4f} | "
        f"{row['dist1']:8.4f} | "
        f"{row['dist2']:8.4f} | "
        f"{row['dist3']:8.4f} | "
        f"{row['rep3']:8.4f}"
    )




print(
    "\n"
    +
    "=" * 110
)

print(
    "DENOISER EFFECT RELATIVE TO RAW"
)

print(
    "=" * 110
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

    ppl_ratio = (
        den["ppl"]
        /
        raw["ppl"]
    )

    if raw["concept"] > 1e-8:
        concept_ratio = (
            den["concept"]
            /
            raw["concept"]
        )
    else:
        concept_ratio = float("nan")

    if (
        raw["concept"] > 1e-8
        and
        concept_ratio >= 0.8
        and
        ppl_ratio < 1.0
    ):
        verdict = "MATCHED-ALPHA IMPROVEMENT"
    else:
        verdict = "NO MATCHED-ALPHA IMPROVEMENT"

    print(
        f"alpha={alpha:>3} | "
        f"Δconcept="
        f"{den['concept'] - raw['concept']:+.4f} | "
        f"ΔPPL="
        f"{den['ppl'] - raw['ppl']:+.4f} | "
        f"PPL ratio={ppl_ratio:.3f} | "
        f"concept ratio={concept_ratio:.3f} | "
        f"{verdict}"
    )




def curve(method, key):
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


raw_c = curve(
    "raw",
    "concept"
)

den_c = curve(
    "denoised",
    "concept"
)

raw_ppl = curve(
    "raw",
    "ppl"
)

den_ppl = curve(
    "denoised",
    "ppl"
)



plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    raw_c,
    marker="o",
    label="Raw steering"
)

plt.plot(
    ALPHAS,
    den_c,
    marker="o",
    label="Steering + denoiser"
)

plt.xlabel("alpha")
plt.ylabel("Concept score")
plt.title(
    "Block-7 same-layer denoiser: concept"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PLOT_CONCEPT,
    dpi=200
)
plt.show()


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    raw_ppl,
    marker="o",
    label="Raw steering"
)

plt.plot(
    ALPHAS,
    den_ppl,
    marker="o",
    label="Steering + denoiser"
)

plt.xlabel("alpha")
plt.ylabel("Completion PPL")
plt.title(
    "Block-7 same-layer denoiser: quality"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PLOT_PPL,
    dpi=200
)
plt.show()


plt.figure(
    figsize=(9, 7)
)

plt.plot(
    raw_ppl,
    raw_c,
    marker="o",
    label="Raw steering"
)

plt.plot(
    den_ppl,
    den_c,
    marker="o",
    label="Steering + denoiser"
)

for alpha, x, y in zip(
    ALPHAS,
    raw_ppl,
    raw_c
):
    plt.annotate(
        f"r{alpha}",
        (x, y)
    )

for alpha, x, y in zip(
    ALPHAS,
    den_ppl,
    den_c
):
    plt.annotate(
        f"d{alpha}",
        (x, y)
    )

plt.xlabel(
    "Completion PPL (lower is better)"
)
plt.ylabel(
    "Concept score (higher is better)"
)
plt.title(
    "Block-7 baseline Pareto comparison"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    PLOT_PARETO,
    dpi=200
)
plt.show()


print("\nSaved:", ALL_CSV)
print("Saved:", SUMMARY_CSV)
print("Saved:", PLOT_CONCEPT)
print("Saved:", PLOT_PPL)
print("Saved:", PLOT_PARETO)
