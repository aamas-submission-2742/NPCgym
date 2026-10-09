"""Generate compact LaTeX paper tables from the selected completed results.

Run ``python -m experiments.generate_paper_tables --details`` from the checkout.
Requires the plots extra. Inputs remain unchanged; output must be a new directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean, stdev
from tempfile import mkdtemp

from experiments.benchmarks import BENCHMARKS, GARDENER_PAPER, PACMAN_PAPER
from experiments.pacman_bolts import FINAL_SEEDS
from experiments.report import BenchmarkRow, collect_benchmark
from experiments.specifications import RESTRAINING_BOLT_TECHNIQUE_ID, UNCONSTRAINED_TECHNIQUE_ID

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Column:
    """Sum these per-seed episode means before calculating between-seed spread."""

    label: str
    metrics: tuple[str, ...]
    decimals: int | None = None
    scale: float = 1


@dataclass(frozen=True)
class PaperRow:
    label: str
    policies: tuple[BenchmarkRow | None, ...]


@dataclass(frozen=True)
class PaperTable:
    key: str
    title: str
    groups: tuple[tuple[str, tuple[Column, ...]], ...]
    rows: tuple[PaperRow, ...]


WIN = Column("Win", ("Win rate",), 1, 100)
GHOSTS = ("Blue eaten", "Orange eaten")
PACMAN_GROUPS = (
    ("Vegan", (WIN, Column("Viol.", GHOSTS))),
    ("Vegetarian", (WIN, Column("Viol.", ("Orange eaten",)))),
    ("HungryVegan", (WIN, Column("hun.", ("Hungry",)), Column("veg.", GHOSTS))),
    ("VeganConflict", (WIN, Column("Viol.", (*GHOSTS, "Missed blue obligation")))),
    ("HungryVeganPenalty", (WIN, Column("Viol.", (*GHOSTS, "Hungry", "Missed pause")))),
)
GRID_COLUMNS = {
    "merchant": (
        Column("Return", ("Return",), 0),
        Column(r"Deaths\,\%", ("Death rate",), 1, 100),
        Column(r"\textit{env.}", ("Environment Friendly",), 2),
        Column(r"\textit{deliv.}", ("Delivery",), 2),
    ),
    "merchant-pacifist": (
        Column("Return", ("Return",), 0),
        Column(r"Deaths\,\%", ("Death rate",), 1, 100),
        Column(r"\textit{pacif.}", ("Danger", "CTD"), 2),
        Column(r"\textit{deliv.}", ("Delivery",), 2),
    ),
    "taxi": (
        Column("Return", ("Return",), 2),
        Column(r"\textit{warn}", ("Warn Violations",), 2),
        Column(r"\textit{stay}", ("Stay Violations",), 2),
        Column(r"\textit{shelter}", ("Seven-Step Safety Violations", "Three-Step Safety Violations"), 2),
    ),
    "gardener": (
        Column("Return", ("Return",), 1),
        Column(r"\textit{collect}", ("Unpermitted collection",), 2),
        Column(r"\textit{drain}", ("Drain",), 2),
        Column(r"\textit{rescue}", ("Rescue per frog",), 2),
    ),
}
GRID_TITLES = {
    "taxi": r"\textbf{Taxi} -- \textit{Emergency}",
    "merchant": r"\textbf{Merchant} -- \textit{EnvFriendly}, \textit{delivery}",
    "merchant-pacifist": r"\textbf{Merchant} -- \textit{Pacifist}, \textit{delivery}",
    "gardener": r"\textbf{Gardener} -- \textit{Collector} (perm.), \textit{Samaritan}",
}
# Algorithm, technique, optimization target and compact display label.
GRID_METHODS = {
    "taxi": (
        ("Q-learning", "base", "None", "Q-learning"),
        ("Q-learning", "fixes", "Warn", r"\quad + fixes (\textit{warn})"),
        ("Q-learning", "bolts", "Emergency", "Q-learning + bolts (50M)"),
    ),
    "merchant": (
        ("Q-learning", "base", "None", "Q-learning"),
        ("Q-learning", "fixes", "Environment Friendly", r"\quad + fixes (\textit{env.})"),
        ("Q-learning", "bolts", "Environment Friendly", "Q-learning + bolts"),
    ),
    "merchant-pacifist": (
        ("Q-learning", "base", "None", "Q-learning"),
        ("Q-learning", "bolts", "DeliveryPacifist", "Q-learning + bolts"),
    ),
    "gardener": (
        ("PPO", "base", "None", "PPO (100k)"),
        ("DQN", "fixes", "Unpermitted collection", "DQN + fixes (perm.)"),
        ("DQN", "fixes", "Drain", "DQN + fixes (drain)"),
        ("DQN", "fixes", "Unpermitted collection + Drain", "DQN + fixes (perm.+drain)"),
        ("DQN", "OFTEN", "Unpermitted collection", "DQN + OFTEN (perm.)"),
        ("DQN", "OFTEN", "Drain", "DQN + OFTEN (drain)"),
        ("DQN", "OFTEN", "Unpermitted collection + Drain", "DQN + OFTEN (perm.+drain)"),
        ("DQN", "bolts", "Collection and rescue", "DQN + bolts (1M)"),
        ("PPO", "bolts", "Collection and rescue", "PPO + bolts (1M)"),
    ),
}
TARGET_LABELS = {
    "Vegetarian Orange": "Vegetarian",
    "Hungry Vegan": "HungryVegan",
    "Hungry Vegan Penalty": "HungryVeganPenalty",
    "Vegan Conflict": "VeganConflict",
    "Unpermitted collection": "Permission",
    "Unpermitted collection + Drain": "Permission + Drain",
}


def _technique(row: BenchmarkRow) -> str:
    if row.phase == "reference":
        return "fixes"
    if row.phase == "teaching":
        return "OFTEN"
    if row.configuration["technique"]["id"] == RESTRAINING_BOLT_TECHNIQUE_ID:
        return "bolts"
    if row.targets != "None":
        return "fixes"
    return "base"


def _algorithm(row: BenchmarkRow) -> str:
    name = row.configuration["algorithm"]["implementation"].rsplit(".", 1)[-1]
    return "Q-learning" if name == "TabularQLearning" else name


def _paper_rows(results: Path, environment: str) -> list[BenchmarkRow]:
    benchmark = {"pacman": PACMAN_PAPER, "gardener": GARDENER_PAPER}.get(environment)
    if benchmark is None:
        identifier = "taxi-emergency" if environment == "taxi" else environment
        benchmark = next(item for item in BENCHMARKS if item.id == identifier)
    rows = collect_benchmark(
        results / environment, benchmark, evaluation_name="final" if environment == "taxi" else "paper"
    )
    # Paired bases remain in the dataset; the paper shows standalone baselines.
    rows = [
        row
        for row in rows
        if row.targets != "None"
        or (row.phase == "training" and row.configuration["technique"]["id"] == UNCONSTRAINED_TECHNIQUE_ID)
    ]
    for row in rows:
        if set(row.samples) != set(FINAL_SEEDS):
            raise ValueError(f"{environment}/{row.label}: expected training seeds {FINAL_SEEDS}")
        if any(sample["Episodes"] != 1000 for sample in row.samples.values()):
            raise ValueError(f"{environment}/{row.label}: expected 1000 evaluation episodes per seed")
    return rows


def _pacman_table(rows: Sequence[BenchmarkRow]) -> PaperTable:
    rows = [row for row in rows if row.targets != "Trapped"]
    lookup: dict[tuple[str, str, str], BenchmarkRow] = {}
    for row in rows:
        key = (_algorithm(row), _technique(row), TARGET_LABELS.get(row.targets, row.targets))
        if key in lookup:
            raise ValueError(f"Multiple Pacman policies for {key}")
        lookup[key] = row
    methods = (("DQN", "base"), ("PPO", "base"), ("DQN", "fixes"), ("DQN", "OFTEN"), ("DQN", "bolts"), ("PPO", "bolts"))
    paper_rows = tuple(
        PaperRow(
            algorithm if technique == "base" else f"{algorithm} + {technique}",
            tuple(
                lookup.get((algorithm, technique, "None" if technique == "base" else title))
                for title, _ in PACMAN_GROUPS
            ),
        )
        for algorithm, technique in methods
    )
    if len({id(policy) for row in paper_rows for policy in row.policies if policy is not None}) != len(rows):
        raise ValueError("Pacman table does not cover every selected policy's algorithm, technique and norm base")
    return PaperTable("pacman", "Pac-Man", PACMAN_GROUPS, paper_rows)


def _grid_table(section: str, rows: Sequence[BenchmarkRow]) -> PaperTable:
    paper_rows: list[PaperRow] = []
    for algorithm, technique, target, label in GRID_METHODS[section]:
        selected = [
            row for row in rows if (_algorithm(row), _technique(row), row.targets) == (algorithm, technique, target)
        ]
        if len(selected) != 1:
            raise ValueError(f"{section}: expected one {algorithm}/{technique}/{target} policy, found {len(selected)}")
        paper_rows.append(PaperRow(label, (selected[0],)))
    environment = "merchant" if section == "merchant-pacifist" else section
    return PaperTable(environment, GRID_TITLES[section], ((section, GRID_COLUMNS[section]),), tuple(paper_rows))


def _statistics(row: BenchmarkRow, column: Column) -> tuple[float, float]:
    values = []
    for sample in row.samples.values():
        members = [sample.get(metric) for metric in column.metrics]
        if any(value is None for value in members):
            raise ValueError(f"{row.label}: missing measurement for {column.metrics}")
        values.append(column.scale * sum(value for value in members if value is not None))
    return mean(values), stdev(values)


def _rounded(value: float, places: int) -> str:
    return str(Decimal(format(value, ".12g")).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))


def _cell(row: BenchmarkRow | None, column: Column) -> str:
    if row is None:
        return "--"
    average, spread = _statistics(row, column)
    places = column.decimals
    if places is None:
        if average < 0.001:
            return r"$<$0.001"
        places = 3 if average < 0.02 else 2
    value = _rounded(average, places).replace("-", "$-$")
    return value + r"{\tiny$\,\pm\,$" + _rounded(spread, places) + "}"


def _render(tables: Sequence[PaperTable], *, pacman: bool) -> str:
    width = max(sum(len(columns) for _, columns in table.groups) for table in tables) + 1
    name = "pacman" if pacman else "grid"
    float_type = "table*" if pacman else "table"
    caption = "Pac-Man" if pacman else "Taxi, Merchant and Gardener"
    lines = [
        "% Means and sample standard deviations across eight training seeds; 1000 episodes per seed.",
        rf"\begin{{{float_type}}}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.2pt}",
        rf"  \caption{{Baselines for {caption}}}",
        rf"  \label{{tab:{name}-baselines}}",
        r"  \resizebox{\linewidth}{!}{%",
        rf"  \begin{{tabular}}{{@{{}}l {'r' * (width - 1)}@{{}}}}",
        r"    \toprule",
    ]
    for index, table in enumerate(tables):
        if index:
            lines.append(r"    \midrule")
        if pacman:
            headings = " & ".join(
                rf"\multicolumn{{{len(columns)}}}{{c}}{{\textit{{{title}}}}}" for title, columns in table.groups
            )
            lines.append("    & " + headings + r" \\")
            start = 2
            rules = []
            for _, group in table.groups:
                end = start + len(group) - 1
                rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
                start = end + 1
            lines.append("    " + "".join(rules))
        else:
            lines.append(rf"    \multicolumn{{{width}}}{{@{{}}l}}{{{table.title}}} \\")
        columns = [column for _, group in table.groups for column in group]
        padding = width - len(columns) - 1
        lines.append("    Method & " + " & ".join(column.label for column in columns) + " &" * padding + r" \\")
        lines.append(r"    \midrule")
        for index, row in enumerate(table.rows):
            if pacman and index == 2:
                lines.append(r"    \midrule")
            cells = [
                _cell(policy, column)
                for policy, (_, group) in zip(row.policies, table.groups, strict=True)
                for column in group
            ]
            lines.append("    " + row.label + " & " + " & ".join(cells) + " &" * padding + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", "  }", rf"\end{{{float_type}}}"]
    return "\n".join(lines) + "\n"


def generate_tables(results: Path, output: Path, *, table: str = "all") -> Path:
    """Write LaTeX and exact sources atomically, refusing existing or overlapping output.

    Select ``all``, ``pacman`` or ``grid``. Requires all current paper recipes,
    seeds 0--7 and 1000 episodes per seed. Presentation selects the compact
    paper rows and columns; the dataset and complete comparison reports remain
    unchanged. Auxiliary pretrained bases are omitted from these tables.
    """
    if table not in {"all", "pacman", "grid"}:
        raise ValueError("table must be all, pacman or grid")
    root, target = results.resolve(), output.absolute()
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("Table output must not contain symbolic links")
    target = target.resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Table input and output directories must be separate")
    if target.exists():
        raise FileExistsError(f"Table output already exists; choose a new directory: {target}")
    tables = []
    files = {}
    if table != "grid":
        pacman = _pacman_table(_paper_rows(root, "pacman"))
        tables.append(pacman)
        files["pacman_baselines.tex"] = _render((pacman,), pacman=True)
    if table != "pacman":
        rows = {environment: _paper_rows(root, environment) for environment in ("taxi", "merchant", "gardener")}
        grid = [
            _grid_table(section, rows["merchant" if section == "merchant-pacifist" else section])
            for section in GRID_METHODS
        ]
        tables.extend(grid)
        files["grid_baselines.tex"] = _render(grid, pacman=False)
    sources = []
    for item in tables:
        for row in item.rows:
            for policy, (group, _) in zip(row.policies, item.groups, strict=True):
                if policy is not None:
                    sources.append(_source(item.key, row.label, group, policy))
    files["sources.json"] = json.dumps(sources, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, source in files.items():
            (temporary / name).write_text(source)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _source(environment: str, label: str, group: str, policy: BenchmarkRow) -> dict[str, object]:
    statistics = {}
    for metric in next(iter(policy.samples.values())):
        if all(sample[metric] is not None for sample in policy.samples.values()):
            average, spread = _statistics(policy, Column(metric, (metric,)))
            statistics[metric] = {"mean": average, "std": spread}
    return {
        "table": environment,
        "method": label,
        "group": group,
        "optimized_norms": policy.targets,
        "seeds": sorted(policy.samples),
        "evaluations": [f"{environment}/{source}/summary.json" for source in policy.sources],
        "statistics": statistics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/output/paper-tables")
    parser.add_argument("--table", choices=("pacman", "grid", "all"), default="all")
    parser.add_argument("--details", action="store_true", help="print each policy's source evaluations")
    parser.add_argument("--print-latex", action="store_true", help="print generated LaTeX")
    args = parser.parse_args(argv)
    try:
        target = generate_tables(args.results, args.output, table=args.table)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    if args.details:
        for record in json.loads((target / "sources.json").read_text()):
            print(f"{record['table']}: {record['method']} [{record['group']}]")
            for source in record["evaluations"]:
                print(f"  {source}")
    if args.print_latex:
        for path in sorted(target.glob("*.tex")):
            print(path.read_text())
    print(f"LaTeX tables and sources written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
