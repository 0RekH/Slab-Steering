import csv
import math
import statistics

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from sae_lens import SAE



DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

# 7th transformer block in human counting
HOOK_NAME = "blocks.6.hook_resid_pre"

FEATURE_IDX = 17363

ALPHAS = [
    0,
    4,
    8,
    16,
    32,
    64,
    96,
    128,
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

ALL_CSV = "block7_raw_steering_generations.csv"
SUMMARY_CSV = "block7_raw_steering_summary.csv"

PLOT_CONCEPT = "block7_raw_concept.png"
PLOT_PPL = "block7_raw_ppl.png"
PLOT_PARETO = "block7_raw_pareto.png"



print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)
model.eval()

print("GPT-2 loaded")



print("\nLoading SAE...")

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=HOOK_NAME,
    device=DEVICE,
)
sae.eval()

print("SAE loaded")



v = sae.W_dec[
    FEATURE_IDX
].detach()

V_NORM = v.norm().item()

print("feature:", FEATURE_IDX)
print("||v||:", V_NORM)




def make_steering_hook(alpha):

    def hook(resid, hook):

        vv = v.to(
            device=resid.device,
            dtype=resid.dtype
        )

        return (
            resid
            +
            alpha * vv
        )

    return hook




@torch.no_grad()
def generate_text(
    prompt,
    alpha,
    seed
):

    torch.manual_seed(seed)

    hook_fn = make_steering_hook(
        alpha
    )

    with model.hooks(
        fwd_hooks=[
            (
                HOOK_NAME,
                hook_fn
            )
        ]
    ):

        text = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            verbose=False
        )

    return text




@torch.no_grad()
def concept_score(text):

    tokens = model.to_tokens(
        text
    )

    _, cache = model.run_with_cache(
        tokens,
        names_filter=[
            HOOK_NAME
        ]
    )

    h = cache[
        HOOK_NAME
    ]

    z = sae.encode(
        h
    )

    feature_values = z[
        0,
        :,
        FEATURE_IDX
    ]

    return (
        feature_values
        .max()
        .item()
    )




@torch.no_grad()
def completion_perplexity(
    prompt,
    full_text
):

    full_tokens = model.to_tokens(
        full_text
    )

    prompt_tokens = model.to_tokens(
        prompt
    )

    prompt_len = (
        prompt_tokens.shape[1]
    )

    if (
        full_tokens.shape[1]
        <= prompt_len
    ):
        return float("nan")

    logits = model(
        full_tokens[:, :-1]
    )

    targets = (
        full_tokens[:, 1:]
    )

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
        target_log_probs[
            :,
            start:
        ]
    )

    nll = (
        -completion_log_probs.mean()
    )

    return (
        torch.exp(
            nll
        )
        .item()
    )



def distinct_n(
    text,
    n
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
            words[
                i:
                i+n
            ]
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



rows = []

total = (
    len(ALPHAS)
    *
    len(PROMPTS)
    *
    len(SEEDS)
)

counter = 0

for alpha in ALPHAS:

    for prompt in PROMPTS:

        for seed in SEEDS:

            counter += 1

            print(
                f"\n[{counter}/{total}] "
                f"alpha={alpha}, seed={seed}"
            )

            text = generate_text(
                prompt,
                alpha,
                seed
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

    writer.writerows(
        rows
    )




def values_for(
    selected,
    key
):

    return [
        row[key]
        for row in selected
        if math.isfinite(
            row[key]
        )
    ]


def mean_metric(
    selected,
    key
):

    values = values_for(
        selected,
        key
    )

    return (
        sum(values)
        /
        len(values)
    )


def std_metric(
    selected,
    key
):

    values = values_for(
        selected,
        key
    )

    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


summary = []

for alpha in ALPHAS:

    selected = [
        row
        for row in rows
        if row["alpha"] == alpha
    ]

    summary.append(
        {
            "alpha":
                alpha,

            "concept_mean":
                mean_metric(
                    selected,
                    "concept"
                ),

            "concept_std":
                std_metric(
                    selected,
                    "concept"
                ),

            "ppl_mean":
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

    writer.writerows(
        summary
    )




print(
    "\n"
    +
    "=" * 110
)

print(
    "BLOCK 7 RAW STEERING"
)

print(
    "=" * 110
)

print(
    f"{'alpha':>7} | "
    f"{'concept':>10} | "
    f"{'PPL':>12} | "
    f"{'dist1':>8} | "
    f"{'dist2':>8} | "
    f"{'dist3':>8} | "
    f"{'rep3':>8}"
)

print(
    "-" * 110
)

for row in summary:

    print(
        f"{row['alpha']:7.1f} | "
        f"{row['concept_mean']:10.4f} | "
        f"{row['ppl_mean']:12.4f} | "
        f"{row['dist1']:8.4f} | "
        f"{row['dist2']:8.4f} | "
        f"{row['dist3']:8.4f} | "
        f"{row['rep3']:8.4f}"
    )




concept_curve = [
    row["concept_mean"]
    for row in summary
]

ppl_curve = [
    row["ppl_mean"]
    for row in summary
]


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    concept_curve,
    marker="o"
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Concept score"
)

plt.title(
    "Block 7 raw steering: concept"
)

plt.grid(
    alpha=0.3
)

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
    ppl_curve,
    marker="o"
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Completion PPL"
)

plt.title(
    "Block 7 raw steering: generation quality"
)

plt.grid(
    alpha=0.3
)

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
    ppl_curve,
    concept_curve,
    marker="o"
)

for alpha, x, y in zip(
    ALPHAS,
    ppl_curve,
    concept_curve
):
    plt.annotate(
        f"a={alpha}",
        (x, y)
    )

plt.xlabel(
    "Completion PPL (lower is better)"
)

plt.ylabel(
    "Concept score (higher is better)"
)

plt.title(
    "Block 7 raw steering Pareto curve"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_PARETO,
    dpi=200
)

plt.show()


print(
    "\nSaved:",
    ALL_CSV
)

print(
    "Saved:",
    SUMMARY_CSV
)

print(
    "Saved:",
    PLOT_CONCEPT
)

print(
    "Saved:",
    PLOT_PPL
)

print(
    "Saved:",
    PLOT_PARETO
)
