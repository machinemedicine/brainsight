from typing import Tuple, Dict
from datetime import datetime, timedelta

import matplotlib.legend
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def str_to_ms(string: str) -> int:
    """Convert a string timestamp into miliseconds."""
    start = datetime.strptime("", "")
    delta = datetime.strptime(string, "%H:%M:%S") - start
    return int(delta.total_seconds() * 1000)


def ms_to_str(ms: int) -> str:
    """Convert a string timestamp into miliseconds."""
    return str(timedelta(milliseconds=int(ms))).split(".")[0]


def nanpow2db(y):
    """Convert Power to dB."""
    y[y == 0] = np.nan
    return 10 * np.log10(y)


def legend_no_handles(ax: plt.Axes, **kwargs) -> matplotlib.legend.Legend:
    "Draw a legend on the Ax with no handles, jsut labels."
    legend = ax.legend(handlelength=0, handletextpad=0, **kwargs)
    for item in legend.legendHandles:
        item.set_visible(False)

    return legend


def draw_activity(
    activity_dict: Dict[str, Tuple[int, int]],
    ax: plt.Axes,
    roi_ms: Tuple[int, int],
    ax_i: int,
    **axvspan_kwargs,
) -> plt.Axes:
    """Draw activities found fithin the specified ROI on a new twin axis."""
    ax_ticks = ax.twiny()
    xmin, xmax = ax.get_xlim()
    roi_range = set(range(*roi_ms))

    activities = dict(
        sorted(activity_dict.items(), key=lambda item: item[1][0])
    )
    i = 1
    ticks = dict()
    for name, (a_s, a_e) in activities.items():
        overlap = set(range(a_s, a_e)).intersection(roi_range)
        if overlap:
            ax.axvspan(
                xmin=a_s, xmax=a_e, label=f"{i}: {name}", **axvspan_kwargs
            )
            ticks[i] = sum(overlap) / len(overlap)
            i += 1

    if ticks:
        ax_ticks.spines["top"].set_visible(False)
        ax_ticks.spines["right"].set_visible(False)

        ax_ticks.set_xlim(xmin, xmax)
        ax_ticks.set_xticks(
            ticks=list(ticks.values()), labels=list(ticks.keys()), fontsize=8
        )
        if ax_i:
            ax_ticks.tick_params(top=False, labeltop=False)
        else:
            legend_no_handles(
                ax=ax,
                ncols=4,
                fontsize=8,
                loc="lower right",
                bbox_to_anchor=(1, 1.1),
                borderaxespad=0.0,
            )
    else:
        ax_ticks.set_visible(False)

    return ax_ticks
