import csv

import torch

from transformer_lens import HookedTransformer
from sae_lens import SAE



DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

# Human counting:
# 7th transformer block = blocks.6
HOOK_NAME = "blocks.6.hook_resid_pre"

TOP_K = 20




SCIENCE_DISCOVERY = [
    "Scientists study the physical laws that govern matter and energy.",
    "The experiment measured the behavior of particles under controlled conditions.",
    "Researchers developed a mathematical model of fluid dynamics.",
    "The telescope collected observations of distant astronomical objects.",
    "The laboratory investigated the properties of materials at high temperature.",
    "Physics explains interactions between forces, fields, matter, and radiation.",
    "The simulation solves differential equations describing a physical system.",
    "The research paper presents experimental evidence and theoretical analysis.",
    "Astronomers study stars, planets, galaxies, and the evolution of the universe.",
    "Scientific measurements are compared with predictions from a mathematical theory.",
]

CONTROL_DISCOVERY = [
    "The restaurant serves breakfast and dinner every day.",
    "She bought a new jacket at the shopping center.",
    "The family spent the weekend visiting friends.",
    "The movie tells the story of two people who meet in a small town.",
    "He walked to the store and bought some groceries.",
    "The hotel offers comfortable rooms near the city center.",
    "They discussed their vacation plans over dinner.",
    "The company announced a new advertising campaign.",
    "The musician released a new album earlier this year.",
    "The children played outside after school.",
]




SCIENCE_VALIDATION = [
    "A numerical simulation was used to investigate turbulent flow.",
    "The researchers analyzed the dynamics of a multiphase physical system.",
    "Experimental data were compared with a theoretical prediction.",
    "The study examines the motion of particles in a gravitational field.",
    "A computational model describes how energy is transported through the system.",
    "Measurements revealed a relationship between pressure and velocity.",
    "The equations describe the evolution of the physical state over time.",
    "Scientists investigated the formation and dynamics of planetary systems.",
    "The experiment tests a hypothesis about the behavior of matter.",
    "Mathematical methods are used to model complex physical processes.",
]

CONTROL_VALIDATION = [
    "The café was crowded during lunch.",
    "She called her friend after work.",
    "The book became popular with readers.",
    "They drove to the countryside for the weekend.",
    "The store closes at nine in the evening.",
    "He ordered coffee and a sandwich.",
    "The actor appeared in a new television series.",
    "The apartment has two bedrooms and a small kitchen.",
    "They celebrated the birthday with friends and family.",
    "The train arrived at the station in the afternoon.",
]




print("\nLoading GPT-2...")

model = HookedTransformer.from_pretrained(
    "gpt2",
    device=DEVICE
)

model.eval()

print("Loading SAE for:", HOOK_NAME)

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=HOOK_NAME,
    device=DEVICE
)

sae.eval()

print("SAE loaded.")
print("W_dec shape:", tuple(sae.W_dec.shape))




@torch.no_grad()
def feature_scores(
    texts
):

    all_scores = []

    for i, text in enumerate(texts):
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

        # z: [1, T, d_sae]
        score = (
            z[
                0
            ]
            .max(
                dim=0
            )
            .values
        )

        all_scores.append(
            score.detach().float().cpu()
        )

    return torch.stack(
        all_scores
    )




science = feature_scores(
    SCIENCE_DISCOVERY
)

control = feature_scores(
    CONTROL_DISCOVERY
)

science_mean = (
    science.mean(
        dim=0
    )
)

control_mean = (
    control.mean(
        dim=0
    )
)

delta = (
    science_mean
    -
    control_mean
)

science_active = (
    science > 0
).float().mean(
    dim=0
)

control_active = (
    control > 0
).float().mean(
    dim=0
)


robust = (
    delta.clamp_min(0)
    *
    science_active
    *
    (
        1.0
        -
        control_active
    )
)

top_values, top_indices = torch.topk(
    robust,
    k=TOP_K
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "BLOCK 7 FEATURE DISCOVERY"
)

print(
    "=" * 120
)

print(
    f"{'rank':>5} | "
    f"{'feature':>8} | "
    f"{'robust':>10} | "
    f"{'delta':>10} | "
    f"{'science':>10} | "
    f"{'control':>10} | "
    f"{'sci_active':>10} | "
    f"{'ctrl_active':>11}"
)

print(
    "-" * 120
)


rows = []

for rank, feature_idx in enumerate(
    top_indices.tolist(),
    start=1
):
    row = {
        "rank":
            rank,

        "feature":
            feature_idx,

        "robust":
            robust[
                feature_idx
            ].item(),

        "delta":
            delta[
                feature_idx
            ].item(),

        "science":
            science_mean[
                feature_idx
            ].item(),

        "control":
            control_mean[
                feature_idx
            ].item(),

        "science_active":
            science_active[
                feature_idx
            ].item(),

        "control_active":
            control_active[
                feature_idx
            ].item(),
    }

    rows.append(
        row
    )

    print(
        f"{rank:5d} | "
        f"{feature_idx:8d} | "
        f"{row['robust']:10.4f} | "
        f"{row['delta']:10.4f} | "
        f"{row['science']:10.4f} | "
        f"{row['control']:10.4f} | "
        f"{row['science_active']:10.4f} | "
        f"{row['control_active']:11.4f}"
    )


with open(
    "block7_feature_candidates.csv",
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




science_val = feature_scores(
    SCIENCE_VALIDATION
)

control_val = feature_scores(
    CONTROL_VALIDATION
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "HELD-OUT VALIDATION OF TOP 5"
)

print(
    "=" * 120
)


validation_rows = []

for rank, feature_idx in enumerate(
    top_indices[
        :5
    ].tolist(),
    start=1
):

    sci_scores = science_val[
        :,
        feature_idx
    ]

    ctrl_scores = control_val[
        :,
        feature_idx
    ]

    sci_mean = (
        sci_scores
        .mean()
        .item()
    )

    ctrl_mean = (
        ctrl_scores
        .mean()
        .item()
    )

    sci_active = (
        sci_scores > 0
    ).float().mean().item()

    ctrl_active = (
        ctrl_scores > 0
    ).float().mean().item()

    validation_rows.append(
        {
            "discovery_rank":
                rank,

            "feature":
                feature_idx,

            "science_mean":
                sci_mean,

            "control_mean":
                ctrl_mean,

            "science_active":
                sci_active,

            "control_active":
                ctrl_active,
        }
    )

    print(
        f"\nfeature: {feature_idx}"
    )

    print(
        "science mean:",
        sci_mean
    )

    print(
        "control mean:",
        ctrl_mean
    )

    print(
        "science active rate:",
        sci_active
    )

    print(
        "control active rate:",
        ctrl_active
    )

    print(
        "science scores:",
        sci_scores.tolist()
    )

    print(
        "control scores:",
        ctrl_scores.tolist()
    )


with open(
    "block7_feature_validation.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            validation_rows[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        validation_rows
    )



best = max(
    validation_rows,
    key=lambda row:
        (
            (
                row["science_mean"]
                -
                row["control_mean"]
            )
            *
            row["science_active"]
            *
            (
                1.0
                -
                row["control_active"]
            )
        )
)

FEATURE_IDX = best[
    "feature"
]

v = sae.W_dec[
    FEATURE_IDX
].detach()

print(
    "\n"
    +
    "=" * 80
)

print(
    "SUGGESTED BLOCK-7 FEATURE"
)

print(
    "=" * 80
)

print(
    "FEATURE_IDX =",
    FEATURE_IDX
)

print(
    "||v|| =",
    v.norm().item()
)

print(
    '\nUse in steering code:'
)

print(
    f'FEATURE_IDX = {FEATURE_IDX}'
)

print(
    'v = sae.W_dec[FEATURE_IDX].detach()'
)


torch.save(
    {
        "hook_name":
            HOOK_NAME,

        "feature_idx":
            FEATURE_IDX,

        "v":
            v.cpu(),

        "v_norm":
            v.norm().item(),
    },
    "block7_selected_steering_vector.pt"
)

print(
    "\nSaved block7_selected_steering_vector.pt"
)
