"""Environment/norm-base comparisons, independent of each method's objectives."""

from dataclasses import dataclass

from experiments import specifications as specs
from experiments.pacman_bolts import BUDGETS, experiment_id
from experiments.paper import (
    EXPERIMENTS_BY_ENVIRONMENT,
    PACMAN_TRAPPED_OFTEN_ID,
    PACMAN_VEGAN_OFTEN_ID,
    PACMAN_VEGETARIAN_OFTEN_ID,
)


@dataclass(frozen=True)
class Benchmark:
    """Comparable experiments and the task metrics and violation counts to report."""

    id: str
    title: str
    experiments: tuple[str, ...]
    counts: tuple[tuple[str, str, str], ...]
    task_metrics: tuple[str, ...]


PACMAN_PAPER = Benchmark(
    "pacman-paper",
    "Pacman smallClassic — all paper norms",
    EXPERIMENTS_BY_ENVIRONMENT["pacman"],
    (
        ("Blue eaten", "pacman/vegan-v0", "VegetarianBlue"),
        ("Orange eaten", "pacman/vegan-v0", "VegetarianOrange"),
        ("Hungry", "pacman/hungry-vegan-v0", "Hungr"),
        ("Trapped", "pacman/trapped-v1", "count"),
        ("Missed blue obligation", "pacman/vegan-conflict-v1", "OblBlue"),
        ("Missed pause", "pacman/hungry-vegan-penalty-v1", "CTD"),
    ),
    ("won", "lost", "food_remaining"),
)


GARDENER_PAPER = Benchmark(
    "gardener-paper",
    "Gardener — all paper norms",
    EXPERIMENTS_BY_ENVIRONMENT["gardener"],
    (
        ("Collect One", "gardener/collect-one-v0", "count"),
        ("Rescue per frog", "gardener/rescue-v2", "Rescue"),
        ("Unpermitted collection", "gardener/permission-aware-v0", "Unpermitted"),
        ("Drain", "gardener/drain-v0", "Drain"),
        ("No Collect", "gardener/no-collect-v0", "NoCollect"),
    ),
    ("score",),
)


_PACMAN_BASES = (specs.PACMAN_DQN_EXPERIMENT_ID, specs.PACMAN_EXPERIMENT_ID)
BENCHMARKS = (
    Benchmark(
        "taxi-emergency",
        "Taxi — Emergency and Warn",
        (
            specs.TAXI_EXPERIMENT_ID,
            specs.TAXI_POLICY_FIX_EXPERIMENT_ID,
            specs.TAXI_BOLT_EXPERIMENT_ID,
            specs.TAXI_WARN_BOLT_EXPERIMENT_ID,
        ),
        tuple(
            (name, "taxi/emergency-v0", name)
            for name in (
                "Warn Violations",
                "Stay Violations",
                "Seven-Step Safety Violations",
                "Three-Step Safety Violations",
                "Safety Violations",
                "Emergency Violations",
            )
        ),
        ("success",),
    ),
    Benchmark(
        "merchant",
        "Merchant — Environment Friendly and DeliveryPacifist",
        (
            specs.MERCHANT_EXPERIMENT_ID,
            specs.MERCHANT_POLICY_FIX_EXPERIMENT_ID,
            *specs.MERCHANT_MINIMIZED_BOLT_EXPERIMENT_IDS,
        ),
        (
            ("Environment Friendly", "merchant/env-friendly-v0", "count"),
            ("Delivery", "merchant/delivery-v0", "count"),
            ("Danger", "merchant/pacifist-v0", "Danger"),
            ("CTD", "merchant/pacifist-v0", "CTD"),
            ("Pacifist total", "merchant/pacifist-v0", "Pacifist(total)"),
            ("DeliveryPacifist total", "merchant/delivery-pacifist-v0", "DeliveryPacifist"),
        ),
        ("unload_market", "death"),
    ),
    Benchmark(
        "gardener",
        "Gardener — collection with permission, Drain and Rescue",
        (
            specs.GARDENER_EXPERIMENT_ID,
            specs.GARDENER_PPO_EXPERIMENT_ID,
            specs.GARDENER_NO_COLLECT_OFTEN_EXPERIMENT_ID,
            specs.GARDENER_DRAIN_OFTEN_EXPERIMENT_ID,
            specs.GARDENER_PERMISSION_OFTEN_EXPERIMENT_ID,
            specs.GARDENER_PERMISSION_DRAIN_OFTEN_EXPERIMENT_ID,
        ),
        (
            ("Unpermitted collection", "gardener/permission-aware-v0", "Unpermitted"),
            ("Drain", "gardener/drain-v0", "Drain"),
            ("Rescue", "gardener/rescue-v1", "count"),
        ),
        ("score",),
    ),
    Benchmark(
        "gardener-collection-rescue",
        "Gardener — Collect One, per-frog Rescue, permission-aware collection and Drain",
        specs.GARDENER_BOLT_EXPERIMENT_IDS,
        (
            ("Collect One", "gardener/collect-one-v0", "count"),
            ("Rescue per frog", "gardener/rescue-v2", "Rescue"),
            ("Unpermitted collection", "gardener/permission-aware-v0", "Unpermitted"),
            ("Drain", "gardener/drain-v0", "Drain"),
        ),
        ("score",),
    ),
    *(
        Benchmark(
            f"pacman-{name}",
            f"Pacman small — {title}",
            (*_PACMAN_BASES, experiment),
            ((title, monitor, member),),
            ("won", "lost", "food_remaining"),
        )
        for name, title, experiment, monitor, member in (
            ("vegan", "Vegan", specs.PACMAN_OFTEN_EXPERIMENT_ID, "pacman/vegan-v0", "Vegan"),
            (
                "vegetarian",
                "Vegetarian Orange",
                specs.PACMAN_VEGETARIAN_OFTEN_EXPERIMENT_ID,
                "pacman/vegetarian-orange-v0",
                "count",
            ),
            ("trapped", "Trapped", specs.PACMAN_TRAPPED_OFTEN_EXPERIMENT_ID, "pacman/trapped-v1", "count"),
        )
    ),
    # Paper runs use smallClassic with the random Berkeley-compatible ghosts.
    *(
        Benchmark(
            f"pacman-smallclassic-{name}",
            f"Pacman smallClassic — {title}",
            ("pacman-dqn-unconstrained-v0", "pacman-ppo-unconstrained-v1", experiment),
            ((title, monitor, member),),
            ("won", "lost", "food_remaining"),
        )
        for name, title, experiment, monitor, member in (
            ("vegan", "Vegan", PACMAN_VEGAN_OFTEN_ID, "pacman/vegan-v0", "Vegan"),
            (
                "vegetarian",
                "Vegetarian Orange",
                PACMAN_VEGETARIAN_OFTEN_ID,
                "pacman/vegetarian-orange-v0",
                "count",
            ),
            ("trapped", "Trapped", PACMAN_TRAPPED_OFTEN_ID, "pacman/trapped-v1", "count"),
        )
    ),
    Benchmark(
        "pacman-images-vegan",
        "Pacman images — Vegan",
        (specs.PACMAN_IMAGES_EXPERIMENT_ID,),
        (("Vegan", "pacman/vegan-v0", "Vegan"),),
        ("won", "lost", "food_remaining"),
    ),
    *(
        Benchmark(
            f"pacman-smallclassic-bolts-{norm}",
            f"Pacman smallClassic — {norm.replace('-', ' ').title()} bolts",
            tuple(experiment_id(norm, algorithm) for algorithm in ("dqn", "ppo")),
            (
                ("Hungry", "pacman/hungry-vegan-v0", "Hungr"),
                ("Vegan", "pacman/vegan-v0", "Vegan"),
            )
            if norm == "hungry-vegan"
            else (("Vegan", "pacman/vegan-v0", "Vegan"),)
            if norm == "vegan"
            else (("Vegetarian Orange", "pacman/vegetarian-orange-v0", "count"),),
            ("won", "lost", "food_remaining", "blue_eaten"),
        )
        for norm in BUDGETS
    ),
)


# Keep pilot candidates in separate configuration groups; final runs use the
# selected candidate ID with seeds 0--7 in a different output root.
BENCHMARKS += tuple(
    Benchmark(
        f"pacman-smallclassic-bolts-{norm}",
        f"Pacman smallClassic — {norm.replace('-', ' ').title()} bolts",
        tuple(
            identifier
            for name, _algorithm, _steps, identifier in (
                *specs.PACMAN_BOLT_STUDY_CONFIGURATIONS,
                *specs.PACMAN_BOLT_EXTENDED_PILOT_CONFIGURATIONS,
            )
            if name == norm
        ),
        counts,
        ("won", "lost", "food_remaining", "blue_eaten"),
    )
    for norm, counts in (
        ("trapped", (("Trapped", "pacman/trapped-v1", "count"),)),
        (
            "vegan-conflict",
            (
                ("Blue", "pacman/vegan-conflict-v1", "VegetarianBlue"),
                ("Orange", "pacman/vegan-conflict-v1", "VegetarianOrange"),
                ("Obligation Blue", "pacman/vegan-conflict-v1", "OblBlue"),
            ),
        ),
        (
            "hungry-vegan-penalty",
            (
                ("Hungry", "pacman/hungry-vegan-penalty-v1", "Hungr"),
                ("Blue", "pacman/hungry-vegan-penalty-v1", "VegetarianBlue"),
                ("Orange", "pacman/hungry-vegan-penalty-v1", "VegetarianOrange"),
                ("Pause", "pacman/hungry-vegan-penalty-v1", "CTD"),
            ),
        ),
    )
)

TARGETS = {
    "pacman/vegan-v0": "Vegan",
    "pacman/vegetarian-orange-v0": "Vegetarian Orange",
    "pacman/trapped-v1": "Trapped",
    "gardener/no-collect-v0": "No Collect",
    "gardener/drain-v0": "Drain",
    "gardener/permission-aware-v0": "Unpermitted collection",
    "gardener/permission-drain-v0": "Unpermitted collection + Drain",
}
