from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_COLORS = ["#274690", "#2C7A7B", "#8A5A44", "#7D4EAC", "#C05621", "#4A5568"]


def _prepare_survival_frame(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    work = df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")

    work = work.dropna(subset=[time_col, event_col]).copy()
    work = work[work[time_col] >= 0].copy()
    work = work[work[event_col].isin([0, 1])].copy()
    work[event_col] = work[event_col].astype(int)
    work = work.sort_values(time_col).reset_index(drop=True)
    return work


def km_event_table(df: pd.DataFrame, time_col: str = "time_to_event", event_col: str = "event") -> pd.DataFrame:
    """Compute a simple Kaplan–Meier event table.

    The table includes both event and censor counts at each observed time.
    Survival is updated only when one or more events occur at that time.
    """
    work = _prepare_survival_frame(df, time_col=time_col, event_col=event_col)
    if work.empty:
        raise ValueError("No valid rows available for Kaplan–Meier estimation.")

    unique_times = np.sort(work[time_col].unique())
    records: list[dict] = []
    survival = 1.0

    for observed_time in unique_times:
        at_risk = int((work[time_col] >= observed_time).sum())
        subset = work[work[time_col] == observed_time]
        n_events = int(subset[event_col].sum())
        n_censored = int(len(subset) - n_events)

        if at_risk > 0 and n_events > 0:
            survival *= (1.0 - n_events / at_risk)

        records.append(
            {
                "time": float(observed_time),
                "at_risk": at_risk,
                "n_events": n_events,
                "n_censored": n_censored,
                "survival": float(survival),
            }
        )

    return pd.DataFrame(records)


def survival_at_times(event_table: pd.DataFrame, times: Iterable[float]) -> list[float]:
    """Return KM survival values at arbitrary times."""
    result: list[float] = []
    sorted_table = event_table.sort_values("time")
    event_times = sorted_table["time"].to_numpy()
    survival_values = sorted_table["survival"].to_numpy()

    for time in times:
        mask = event_times <= time
        if not mask.any():
            result.append(1.0)
        else:
            result.append(float(survival_values[mask][-1]))
    return result


def censor_mark_coordinates(
    df: pd.DataFrame,
    event_table: pd.DataFrame,
    time_col: str = "time_to_event",
    event_col: str = "event",
) -> pd.DataFrame:
    """Return x/y coordinates for censor ticks."""
    work = _prepare_survival_frame(df, time_col=time_col, event_col=event_col)
    censors = work[work[event_col] == 0].copy()
    if censors.empty:
        return pd.DataFrame(columns=["time", "survival"])

    censors["survival"] = survival_at_times(event_table, censors[time_col].tolist())
    return censors[[time_col, "survival"]].rename(columns={time_col: "time"})


def at_risk_counts(df: pd.DataFrame, evaluation_times: Iterable[float], time_col: str = "time_to_event") -> list[int]:
    """Return simple at-risk counts for display in a risk table."""
    times = pd.to_numeric(df[time_col], errors="coerce")
    counts = [int((times >= time_point).sum()) for time_point in evaluation_times]
    return counts


def default_time_grid(df: pd.DataFrame, time_col: str = "time_to_event", n_points: int = 5) -> list[int]:
    """Create a readable display grid for at-risk counts."""
    times = pd.to_numeric(df[time_col], errors="coerce").dropna()
    if times.empty:
        return [0]

    max_time = float(times.max())
    if max_time == 0:
        return [0]

    grid = np.linspace(0, max_time, n_points)
    rounded = np.unique(np.round(grid).astype(int))
    return rounded.tolist()


def median_survival(event_table: pd.DataFrame) -> float | None:
    """Return the median survival time if the KM curve crosses 0.5."""
    mask = event_table["survival"] <= 0.5
    if not mask.any():
        return None
    return float(event_table.loc[mask, "time"].iloc[0])


def step_plot_arrays(event_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Construct x/y arrays for a post-step Kaplan–Meier plot."""
    x_values = np.concatenate(([0.0], event_table["time"].to_numpy(dtype=float)))
    y_values = np.concatenate(([1.0], event_table["survival"].to_numpy(dtype=float)))
    return x_values, y_values


def km_summary_payload(df: pd.DataFrame, event_table: pd.DataFrame, label: str) -> dict:
    """Create a compact JSON-friendly summary for a KM fit."""
    work = _prepare_survival_frame(df, time_col="time_to_event", event_col="event")
    return {
        "label": label,
        "n_rows": int(len(work)),
        "n_events": int(work["event"].sum()),
        "n_censored": int((work["event"] == 0).sum()),
        "time_min": float(work["time_to_event"].min()),
        "time_max": float(work["time_to_event"].max()),
        "median_survival": median_survival(event_table),
        "curve_crosses_0_5": median_survival(event_table) is not None,
    }


def plot_km_with_at_risk(
    df: pd.DataFrame,
    output_path: Path,
    *,
    time_col: str = "time_to_event",
    event_col: str = "event",
    group_col: str | None = None,
    title: str = "Kaplan–Meier Estimate",
    time_unit: str = "days",
    group_order: list[str] | None = None,
) -> dict[str, dict]:
    """Plot Kaplan–Meier curve(s) with censor ticks and an at-risk table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group_col is None:
        groups = {"All": df.copy()}
    else:
        work = df.dropna(subset=[group_col]).copy()
        if group_order is None:
            group_order = sorted(work[group_col].astype(str).unique().tolist())
        groups = {group: work[work[group_col].astype(str) == str(group)].copy() for group in group_order}

    fig = plt.figure(figsize=(10, 6.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.4], hspace=0.12)
    ax = fig.add_subplot(grid[0])
    ax_table = fig.add_subplot(grid[1])

    summaries: dict[str, dict] = {}
    risk_rows: list[list[int]] = []
    row_labels: list[str] = []
    row_colors: list[str] = []

    combined_df = pd.concat(groups.values(), axis=0, ignore_index=True)
    time_grid = default_time_grid(combined_df, time_col=time_col, n_points=5)

    for idx, (label, group_df) in enumerate(groups.items()):
        prepared = _prepare_survival_frame(group_df, time_col=time_col, event_col=event_col)
        if prepared.empty:
            continue

        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        event_table = km_event_table(prepared, time_col=time_col, event_col=event_col)
        x_values, y_values = step_plot_arrays(event_table)
        ax.step(x_values, y_values, where="post", linewidth=2.2, color=color, label=label)

        censors = censor_mark_coordinates(prepared, event_table, time_col=time_col, event_col=event_col)
        if not censors.empty:
            ax.scatter(
                censors["time"],
                censors["survival"],
                marker="|",
                s=80,
                linewidths=1.2,
                color=color,
            )

        risk_rows.append(at_risk_counts(prepared, time_grid, time_col=time_col))
        row_labels.append(label if group_col is not None else "At risk")
        row_colors.append(color)
        summaries[label] = km_summary_payload(prepared, event_table, label=label)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Survival probability")
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, linestyle="--")
    if group_col is not None and len(groups) > 1:
        ax.legend(frameon=False)

    ax_table.axis("off")
    table = ax_table.table(
        cellText=risk_rows,
        rowLabels=row_labels,
        colLabels=[str(t) for t in time_grid],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for row_index, color in enumerate(row_colors, start=1):
        if (row_index, -1) in table.get_celld():
            table[(row_index, -1)].get_text().set_color(color)
            table[(row_index, -1)].get_text().set_fontweight("bold")

    ax_table.set_title("At risk", fontsize=11, pad=4, loc="left")

    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.18)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summaries["time_grid"] = {"display_times": time_grid}
    return summaries
