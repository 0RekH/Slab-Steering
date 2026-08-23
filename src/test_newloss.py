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

CHECKPOINT_PATH = "gated_gaussian_repair_block7_to_block9.pt"


STEERING_HOOK = "blocks.6.hook_resid_pre"


REPAIR_HOOK_EXPECTED = "blocks.8.hook_resid_pre"

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

ALL_CSV = "block7_gated_gaussian_repair_generations.csv"
SUMMARY_CSV = "block7_gated_gaussian_repair_summary.csv"

PLOT_CONCEPT = "block7_gated_gaussian_repair_concept.png"
PLOT_PPL = "block7_gated_gaussian_repair_ppl.png"
PLOT_PARETO = "block7_gated_gaussian_repair_pareto.png"
PLOT_GATE = "block7_gated_gaussian_repair_gate.png"
PLOT_CORR = "block7_gated_gaussian_repair_correction.png"



print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)
model.eval()

print("GPT-2 loaded.")



checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

PERTURB_HOOK = checkpoint["perturb_hook"]
REPAIR_HOOK = checkpoint["repair_hook"]

print("\nCheckpoint metadata:")
print(" perturb hook:", PERTURB_HOOK)
print(" repair hook:", REPAIR_HOOK)
print(" noise range:",
      checkpoint.get("min_noise_norm"),
      "to",
      checkpoint.get("max_noise_norm"))
print(" rho_min:", checkpoint.get("rho_min"))
print(" objective:", checkpoint.get("objective"))
print(" corruption:", checkpoint.get("corruption"))

if PERTURB_HOOK != STEERING_HOOK:
    raise ValueError(
        f"Checkpoint perturb hook is {PERTURB_HOOK}, "
        f"but STEERING_HOOK is {STEERING_HOOK}."
    )

if REPAIR_HOOK != REPAIR_HOOK_EXPECTED:
    raise ValueError(
        f"Checkpoint repair hook is {REPAIR_HOOK}, "
        f"expected {REPAIR_HOOK_EXPECTED}."
    )




print("\nLoading SAE...")

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=STEERING_HOOK,
    device=DEVICE
)
sae.eval()

v = sae.W_dec[
    FEATURE_IDX
].detach()

V_NORM = v.norm().item()

print("feature:", FEATURE_IDX)
print("||v||:", V_NORM)




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
            self.correction_head(
                z
            )
        )

        gate = torch.sigmoid(
            self.gate_head(
                z
            )
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
    d_model=checkpoint["d_model"],
    hidden_dim=checkpoint["hidden_dim"],
    bottleneck_dim=checkpoint["bottleneck_dim"],
    sigma_dim=checkpoint["sigma_dim"],
).to(DEVICE)

repair.load_state_dict(
    checkpoint["state_dict"]
)

repair.eval()

print("Repair model loaded.")




def make_steering_hook(alpha):

    def hook(
        resid,
        hook
    ):
        vv = v.to(
            device=resid.device,
            dtype=resid.dtype
        )

        return (
            resid
            +
            alpha
            *
            vv
        )

    return hook


def make_repair_hook(
    alpha,
    diagnostics
):


    def hook(
        resid,
        hook
    ):
        if alpha == 0:
            return resid

        B, T, D = resid.shape

        perturbation_norm = (
            abs(alpha)
            *
            V_NORM
        )

        sigma_value = (
            perturbation_norm
            /
            math.sqrt(D_MODEL)
        )

        sigma = torch.full(
            (B, T, 1),
            sigma_value,
            device=resid.device,
            dtype=resid.dtype
        )

        repaired, correction, gate = repair(
            resid,
            sigma
        )

        diagnostics["gate_sum"] += (
            gate.detach().float().sum().item()
        )
        diagnostics["gate_n"] += (
            gate.numel()
        )

        corr_norm = (
            torch.linalg.vector_norm(
                correction.detach().float(),
                dim=-1
            )
        )

        diagnostics["corr_sum"] += (
            corr_norm.sum().item()
        )

        diagnostics["corr_n"] += (
            corr_norm.numel()
        )

        return repaired.to(
            resid.dtype
        )

    return hook




@torch.no_grad()
def generate_text(
    prompt,
    alpha,
    seed,
    method
):

    torch.manual_seed(
        seed
    )

    if method == "raw":

        diagnostics = None

        hooks = [
            (
                STEERING_HOOK,
                make_steering_hook(
                    alpha
                )
            )
        ]

    elif method == "repaired":

        diagnostics = {
            "gate_sum": 0.0,
            "gate_n": 0,
            "corr_sum": 0.0,
            "corr_n": 0,
        }

        hooks = [
            (
                STEERING_HOOK,
                make_steering_hook(
                    alpha
                )
            ),
            (
                REPAIR_HOOK,
                make_repair_hook(
                    alpha,
                    diagnostics
                )
            ),
        ]

    else:
        raise ValueError(
            method
        )

    with model.hooks(
        fwd_hooks=hooks
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

    if diagnostics is None:

        gate_mean = float("nan")
        corr_norm = float("nan")

    else:

        gate_mean = (
            diagnostics["gate_sum"]
            /
            max(
                diagnostics["gate_n"],
                1
            )
        )

        corr_norm = (
            diagnostics["corr_sum"]
            /
            max(
                diagnostics["corr_n"],
                1
            )
        )

    return (
        text,
        gate_mean,
        corr_norm
    )



@torch.no_grad()
def concept_score(text):

    tokens = model.to_tokens(
        text
    )

    _, cache = model.run_with_cache(
        tokens,
        names_filter=[
            STEERING_HOOK
        ]
    )

    h = cache[
        STEERING_HOOK
    ]

    z = sae.encode(
        h
    )

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
        <=
        prompt_len
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




METHODS = [
    "raw",
    "repaired",
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

                (
                    text,
                    gate_mean,
                    correction_norm
                ) = generate_text(
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

                        "gate_mean":
                            gate_mean,

                        "correction_norm":
                            correction_norm,

                        "text":
                            text,
                    }
                )

                print(
                    f"concept={concept:.4f} | "
                    f"PPL={ppl:.4f} | "
                    f"gate={gate_mean:.4f} | "
                    f"corr={correction_norm:.4f}"
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
    values = finite_values(
        selected,
        key
    )

    return (
        sum(values)
        /
        len(values)
        if values
        else float("nan")
    )


def std_metric(
    selected,
    key
):
    values = finite_values(
        selected,
        key
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
                row["method"]
                ==
                method
                and
                row["alpha"]
                ==
                alpha
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

                "gate":
                    mean_metric(
                        selected,
                        "gate_mean"
                    ),

                "corr_norm":
                    mean_metric(
                        selected,
                        "correction_norm"
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
    "=" * 130
)

print(
    "BLOCK 7 GATED GAUSSIAN REPAIR: RAW vs REPAIRED"
)

print(
    "=" * 130
)

print(
    f"{'method':>10} | "
    f"{'alpha':>6} | "
    f"{'concept':>10} | "
    f"{'PPL':>12} | "
    f"{'dist1':>8} | "
    f"{'dist2':>8} | "
    f"{'dist3':>8} | "
    f"{'rep3':>8} | "
    f"{'gate':>8} | "
    f"{'corr':>10}"
)

print(
    "-" * 130
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
        f"{row['rep3']:8.4f} | "
        f"{row['gate']:8.4f} | "
        f"{row['corr_norm']:10.4f}"
    )




print(
    "\n"
    +
    "=" * 115
)

print(
    "MATCHED-ALPHA EFFECT"
)

print(
    "=" * 115
)


for alpha in ALPHAS:

    raw = next(
        row
        for row in summary
        if (
            row["method"]
            ==
            "raw"
            and
            row["alpha"]
            ==
            alpha
        )
    )

    rep = next(
        row
        for row in summary
        if (
            row["method"]
            ==
            "repaired"
            and
            row["alpha"]
            ==
            alpha
        )
    )

    ppl_ratio = (
        rep["ppl"]
        /
        raw["ppl"]
    )

    if raw["concept"] > 1e-8:

        concept_ratio = (
            rep["concept"]
            /
            raw["concept"]
        )

    else:

        concept_ratio = float(
            "nan"
        )

    
    if (
        raw["concept"] > 1e-8
        and
        concept_ratio >= 0.8
        and
        ppl_ratio < 1.0
    ):

        verdict = "PROMISING"

    elif (
        raw["concept"] <= 1e-8
        and
        ppl_ratio < 1.0
    ):

        verdict = (
            "QUALITY RECOVERY ONLY; "
            "RAW CONCEPT IS ZERO"
        )

    elif ppl_ratio < 1.0:

        verdict = (
            "QUALITY BETTER, "
            "CONCEPT LOST"
        )

    elif (
        raw["concept"] > 1e-8
        and
        concept_ratio >= 0.8
    ):

        verdict = (
            "CONCEPT PRESERVED, "
            "QUALITY NOT BETTER"
        )

    else:

        verdict = "NO IMPROVEMENT"

    print(
        f"alpha={alpha:>3} | "
        f"concept ratio={concept_ratio:>7.3f} | "
        f"PPL ratio={ppl_ratio:>7.3f} | "
        f"{verdict}"
    )




def curve(
    method,
    key
):
    return [
        next(
            row[key]
            for row in summary
            if (
                row["method"]
                ==
                method
                and
                row["alpha"]
                ==
                alpha
            )
        )
        for alpha in ALPHAS
    ]


raw_concept = curve(
    "raw",
    "concept"
)

rep_concept = curve(
    "repaired",
    "concept"
)

raw_ppl = curve(
    "raw",
    "ppl"
)

rep_ppl = curve(
    "repaired",
    "ppl"
)

rep_gate = curve(
    "repaired",
    "gate"
)

rep_corr = curve(
    "repaired",
    "corr_norm"
)




plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    raw_concept,
    marker="o",
    label="Raw steering"
)

plt.plot(
    ALPHAS,
    rep_concept,
    marker="o",
    label="Gated Gaussian repair"
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
    rep_ppl,
    marker="o",
    label="Gated Gaussian repair"
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Completion PPL"
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
    dpi=200
)

plt.show()


plt.figure(
    figsize=(9, 7)
)

plt.plot(
    raw_ppl,
    raw_concept,
    marker="o",
    label="Raw steering"
)

plt.plot(
    rep_ppl,
    rep_concept,
    marker="o",
    label="Gated Gaussian repair"
)

for alpha, x, y in zip(
    ALPHAS,
    raw_ppl,
    raw_concept
):
    plt.annotate(
        f"r{alpha}",
        (x, y)
    )

for alpha, x, y in zip(
    ALPHAS,
    rep_ppl,
    rep_concept
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
    "Block 7 quality-concept Pareto comparison"
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_PARETO,
    dpi=200
)

plt.show()


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    rep_gate,
    marker="o"
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Mean gate"
)

plt.title(
    "Repair intervention strength"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_GATE,
    dpi=200
)

plt.show()


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    ALPHAS,
    rep_corr,
    marker="o"
)

plt.xlabel(
    "Steering strength alpha"
)

plt.ylabel(
    "Mean correction L2 norm"
)

plt.title(
    "Magnitude of block-9 repair"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_CORR,
    dpi=200
)

plt.show()


print("\nEvaluation finished.")
print("Saved:", ALL_CSV)
print("Saved:", SUMMARY_CSV)
print("Saved:", PLOT_CONCEPT)
print("Saved:", PLOT_PPL)
print("Saved:", PLOT_PARETO)
print("Saved:", PLOT_GATE)
print("Saved:", PLOT_CORR)
