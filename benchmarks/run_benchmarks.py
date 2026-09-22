#!/usr/bin/env python3
"""Build and benchmark the llama2.c M1 Max variants with repeatable settings.

The harness deliberately compiles every variant with one Clang installation,
uses deterministic greedy decoding, records every trial, and verifies generated
output hashes. It depends only on Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKENS_RE = re.compile(r"achieved tok/s:\s*([0-9.]+)")


@dataclass(frozen=True)
class Configuration:
    variant: str
    threads: int
    schedule: str | None

    @property
    def label(self) -> str:
        suffix = f", {self.schedule}" if self.schedule else ""
        return f"{self.variant}, {self.threads} thread{'s' if self.threads != 1 else ''}{suffix}"


def checked_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_clang(requested: str | None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    if os.environ.get("FASTOLLAMA_CC"):
        candidates.append(Path(os.environ["FASTOLLAMA_CC"]).expanduser())
    if shutil.which("brew"):
        llvm_prefix = checked_output(["brew", "--prefix", "llvm"])
        if llvm_prefix != "unavailable":
            candidates.append(Path(llvm_prefix) / "bin" / "clang")
    if shutil.which("clang"):
        candidates.append(Path(shutil.which("clang") or "clang"))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SystemExit("No usable Clang found. Pass --cc or set FASTOLLAMA_CC.")


def compile_variants(cc: Path, build_dir: Path, include_neon: bool) -> tuple[dict[str, Path], dict[str, list[str]]]:
    source = ROOT / "run.c"
    common = [str(cc), "-std=c11", "-Wall", "-Wextra", "-march=native"]
    commands: dict[str, list[str]] = {
        "baseline": common + ["-O3", str(source), "-lm", "-o", str(build_dir / "baseline")],
        "fast": common + ["-O3", "-ffast-math", str(source), "-lm", "-o", str(build_dir / "fast")],
        "omp": common
        + [
            "-O3",
            "-ffast-math",
            "-fopenmp",
            "-DFASTOLLAMA_RUNTIME_SCHEDULE",
            str(source),
            "-lm",
            "-o",
            str(build_dir / "omp"),
        ],
    }
    if include_neon:
        commands["neon"] = common + [
            "-O3",
            "-ffast-math",
            "-fopenmp",
            "-DFASTOLLAMA_RUNTIME_SCHEDULE",
            "-DFASTOLLAMA_NEON",
            str(source),
            "-lm",
            "-o",
            str(build_dir / "neon"),
        ]

    binaries: dict[str, Path] = {}
    for name, command in commands.items():
        print(f"building {name}...", flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(f"failed to build {name}")
        binaries[name] = build_dir / name
    return binaries, commands


def configurations(threads: list[int], schedules: list[str], include_neon: bool) -> list[Configuration]:
    configs = [Configuration("baseline", 1, None), Configuration("fast", 1, None)]
    configs.extend(Configuration("omp", thread, schedule) for schedule in schedules for thread in threads)
    if include_neon:
        configs.extend(Configuration("neon", thread, "static") for thread in threads)
    return configs


def run_once(binary: Path, config: Configuration, model: Path, tokenizer: Path, steps: int) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(config.threads),
            "OMP_DYNAMIC": "false",
            "OMP_SCHEDULE": config.schedule or "static",
        }
    )
    command = [
        str(binary),
        str(model),
        "-z",
        str(tokenizer),
        "-t",
        "0",
        "-s",
        "42",
        "-n",
        str(steps),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    stderr = completed.stderr.decode("utf-8", errors="replace")
    match = TOKENS_RE.search(stderr)
    if completed.returncode or not match:
        sys.stderr.write(completed.stdout.decode("utf-8", errors="replace"))
        sys.stderr.write(stderr)
        raise SystemExit(f"benchmark failed for {config.label}")
    return {
        "tokens_per_second": float(match.group(1)),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def summarize(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str | None], list[dict[str, object]]] = {}
    for sample in samples:
        key = (str(sample["variant"]), int(sample["threads"]), sample["schedule"])
        grouped.setdefault(key, []).append(sample)

    rows: list[dict[str, object]] = []
    for (variant, threads, schedule), values in grouped.items():
        rates = [float(value["tokens_per_second"]) for value in values]
        hashes = sorted({str(value["stdout_sha256"]) for value in values})
        rows.append(
            {
                "variant": variant,
                "threads": threads,
                "schedule": schedule or "-",
                "median_tokens_per_second": statistics.median(rates),
                "mean_tokens_per_second": statistics.mean(rates),
                "stdev_tokens_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "min_tokens_per_second": min(rates),
                "max_tokens_per_second": max(rates),
                "stdout_sha256": ";".join(hashes),
            }
        )

    baseline = next(float(row["median_tokens_per_second"]) for row in rows if row["variant"] == "baseline")
    for row in rows:
        row["speedup_vs_baseline"] = float(row["median_tokens_per_second"]) / baseline
    return rows


def machine_metadata(cc: Path, model: Path, tokenizer: Path, arguments: argparse.Namespace) -> dict[str, object]:
    git_status = checked_output(["git", "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": checked_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status and git_status != "unavailable"),
        "run_c_sha256": sha256_file(ROOT / "run.c"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": checked_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if sys.platform == "darwin"
        else platform.processor(),
        "physical_cpus": checked_output(["sysctl", "-n", "hw.physicalcpu"])
        if sys.platform == "darwin"
        else os.cpu_count(),
        "logical_cpus": checked_output(["sysctl", "-n", "hw.logicalcpu"])
        if sys.platform == "darwin"
        else os.cpu_count(),
        "performance_cpus": checked_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if sys.platform == "darwin"
        else "unavailable",
        "efficiency_cpus": checked_output(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        if sys.platform == "darwin"
        else "unavailable",
        "memory_bytes": checked_output(["sysctl", "-n", "hw.memsize"])
        if sys.platform == "darwin"
        else "unavailable",
        "os": checked_output(["sw_vers"]) if sys.platform == "darwin" else platform.platform(),
        "compiler": checked_output([str(cc), "--version"]).splitlines()[0],
        "compiler_path": str(cc),
        "model_path": model.name,
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256_file(model),
        "tokenizer_path": tokenizer.name,
        "tokenizer_sha256": sha256_file(tokenizer),
        "steps": arguments.steps,
        "warmups": arguments.warmups,
        "repetitions": arguments.repetitions,
        "threads": arguments.threads,
        "schedules": arguments.schedules,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, metadata: dict[str, object], rows: list[dict[str, object]]) -> None:
    best = max(rows, key=lambda row: float(row["median_tokens_per_second"]))
    lines = [
        "# Benchmark summary",
        "",
        f"- CPU: {metadata['cpu']}",
        f"- Compiler: {metadata['compiler']}",
        f"- Model SHA-256: `{metadata['model_sha256']}` ({int(metadata['model_bytes']) / 1_000_000:.1f} MB)",
        f"- Method: {metadata['warmups']} warmup(s), {metadata['repetitions']} measured trials, {metadata['steps']} generated tokens",
        f"- Best observed configuration: {best['variant']} / {best['threads']} threads / {best['schedule']} at "
        f"{float(best['median_tokens_per_second']):.2f} tok/s ({float(best['speedup_vs_baseline']):.2f}x baseline)",
        "",
        "| Variant | Threads | Schedule | Median tok/s | Mean ± SD | Speedup | Same output |",
        "|---|---:|---|---:|---:|---:|:---:|",
    ]
    baseline_hash = next(str(row["stdout_sha256"]) for row in rows if row["variant"] == "baseline")
    for row in rows:
        same = "yes" if row["stdout_sha256"] == baseline_hash else "no"
        lines.append(
            f"| {row['variant']} | {row['threads']} | {row['schedule']} | "
            f"{float(row['median_tokens_per_second']):.2f} | "
            f"{float(row['mean_tokens_per_second']):.2f} ± {float(row['stdev_tokens_per_second']):.2f} | "
            f"{float(row['speedup_vs_baseline']):.2f}x | {same} |"
        )
    lines.extend(
        [
            "",
            "`baseline` is `-O3`; `fast` adds `-ffast-math`; `omp` adds OpenMP; "
            "`neon` replaces the hot dot-product loop with explicit ARM NEON intrinsics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_chart(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["schedule"] == "static" and row["variant"] in {"omp", "neon"}]
    if not selected:
        return
    width, height = 820, 460
    left, top, plot_width, plot_height = 75, 35, 700, 340
    maximum = max(float(row["median_tokens_per_second"]) for row in selected) * 1.12
    threads = sorted({int(row["threads"]) for row in selected})
    x_for = lambda thread: left + threads.index(thread) * (plot_width / max(1, len(threads) - 1))
    y_for = lambda value: top + plot_height - value / maximum * plot_height
    colors = {"omp": "#2563eb", "neon": "#dc2626"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#111827}.axis{stroke:#9ca3af}.grid{stroke:#e5e7eb}</style>',
        '<text x="410" y="22" text-anchor="middle" font-size="18" font-weight="600">M1 Max thread scaling (static schedule)</text>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = y_for(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.0f}</text>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    elements.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    for thread in threads:
        x = x_for(thread)
        elements.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12">{thread}</text>')
    for variant in ("omp", "neon"):
        points = sorted((row for row in selected if row["variant"] == variant), key=lambda row: int(row["threads"]))
        if not points:
            continue
        coordinates = " ".join(f"{x_for(int(row['threads'])):.1f},{y_for(float(row['median_tokens_per_second'])):.1f}" for row in points)
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[variant]}" stroke-width="3"/>')
        for row in points:
            x, y = x_for(int(row["threads"])), y_for(float(row["median_tokens_per_second"]))
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[variant]}"/>')
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-size="13">OpenMP threads</text>',
            f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-size="13">tokens / second</text>',
            '<circle cx="105" cy="58" r="5" fill="#2563eb"/><text x="117" y="62" font-size="12">Clang auto-vectorized</text>',
            '<circle cx="105" cy="80" r="5" fill="#dc2626"/><text x="117" y="84" font-size="12">Explicit NEON</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_csv_list(value: str, converter=str) -> list:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a llama2.c float32 checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.bin")
    parser.add_argument("--cc", help="Clang binary; defaults to Homebrew LLVM when available")
    parser.add_argument("--threads", default="1,2,4,8,10")
    parser.add_argument("--schedules", default="static,dynamic,guided")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results" / "local")
    parser.add_argument("--skip-neon", action="store_true")
    args = parser.parse_args()

    args.model = args.model.expanduser().resolve()
    args.tokenizer = args.tokenizer.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.threads = parse_csv_list(args.threads, int)
    args.schedules = parse_csv_list(args.schedules)
    if not args.model.is_file() or not args.tokenizer.is_file():
        parser.error("model and tokenizer paths must exist")
    if args.steps < 2 or args.repetitions < 1 or args.warmups < 0:
        parser.error("steps must be >= 2, repetitions >= 1, and warmups >= 0")

    include_neon = not args.skip_neon and platform.machine().lower() in {"arm64", "aarch64"}
    cc = find_clang(args.cc)
    configs = configurations(args.threads, args.schedules, include_neon)
    metadata = machine_metadata(cc, args.model, args.tokenizer, args)

    samples: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="llama2c-m1max-bench-") as temporary:
        binaries, commands = compile_variants(cc, Path(temporary), include_neon)
        metadata["compile_commands"] = {
            name: [
                item.replace(str(ROOT), ".").replace(temporary, "<build>")
                for item in command
            ]
            for name, command in commands.items()
        }
        print("warming each configuration...", flush=True)
        for config in configs:
            for _ in range(args.warmups):
                run_once(binaries[config.variant], config, args.model, args.tokenizer, args.steps)

        # Round-robin trials so temperature/background drift is distributed
        # across configurations instead of favoring one contiguous block.
        for trial in range(1, args.repetitions + 1):
            print(f"measured round {trial}/{args.repetitions}", flush=True)
            for config in configs:
                result = run_once(binaries[config.variant], config, args.model, args.tokenizer, args.steps)
                sample = {**asdict(config), "trial": trial, **result}
                samples.append(sample)
                print(f"  {config.label}: {float(result['tokens_per_second']):.2f} tok/s", flush=True)

    rows = summarize(samples)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps({"metadata": metadata, "samples": samples}, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "summary.csv", rows)
    write_markdown(args.output / "README.md", metadata, rows)
    svg_chart(args.output / "thread-scaling.svg", rows)

    hashes = {str(row["stdout_sha256"]) for row in rows}
    print(f"wrote results to {args.output}")
    print(f"generated output hashes: {len(hashes)} ({'consistent' if len(hashes) == 1 else 'review required'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
