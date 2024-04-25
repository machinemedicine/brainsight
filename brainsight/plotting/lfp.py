from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter
from brainsight.plotting.utils import ms_to_str


class LFP(BasePlotter):
    _single_ax = False
    _horizontal = False
    _base_wh = (12, 4)
    _use_cache = True

    def __init__(
        self,
        dataset: Dataset,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> None:
        super().__init__(dataset)
        self.low_freq = low_freq
        self.high_freq = high_freq

    def _get_data(self, signal: Signal):
        values = signal.values
        # Apply filtering if either threshold is set
        if self.low_freq or self.high_freq:
            values = filter_data(
                data=values,
                sfreq=signal.sampling_rate,
                l_freq=self.low_freq,
                h_freq=self.high_freq,
                verbose=False,
            )
        return values

    def _draw_activity(self, ax: plt.Axes, roi: Tuple[int, int]):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        roi_range = set(range(*roi))

        activities = dict(
            sorted(self.dataset.ACTIVITY.items(), key=lambda item: item[1][0])
        )
        i = 1
        for name, (a_s, a_e) in activities.items():
            overlap = set(range(a_s, a_e)).intersection(roi_range)
            if overlap:
                ax.axvspan(
                    xmin=a_s,
                    xmax=a_e,
                    zorder=1,
                    alpha=0.2,
                    color="purple",
                    label=f"{i}: {name}",
                )
                ax.vlines(
                    [a_s, a_e],
                    ymin,
                    ymax,
                    zorder=3,
                    color="purple",
                    alpha=0.3,
                    ls=":",
                )
                alpha = 1.0 if (i % 2) else 0.8
                ax.text(
                    sum(overlap) / len(overlap),
                    ymax,
                    s=i,
                    ha="center",
                    va="bottom",
                    alpha=alpha,
                )

                i += 1

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        show_activity: bool = True,
    ):
        values = self.get_data(channel=channel, signal=signal)

        ax.plot(signal.ts, values, color="steelblue", lw=0.8)

        ax.grid(color="black", alpha=0.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(f"Channel: {channel}", ha="left", x=0, fontsize=8)

        ax.set_ylabel("LFP [µV]")
        ax.set_xlim(*roi)

        xticks = ax.get_xticks().astype(int)
        labels = [ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*roi)

        if show_activity:
            self._draw_activity(ax=ax, roi=roi)

        if ax_i:
            ax.set_xlabel("Time [HH:MM:SS]")
            ax.tick_params(labeltop=False)

        else:
            ax.tick_params(labelbottom=False)
            if show_activity:
                ax.legend(
                    ncols=4,
                    fontsize=8,
                    handlelength=0,
                    handletextpad=0,
                    loc="lower right",
                    bbox_to_anchor=(1, 1.05),
                )

        return None

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        lims = sum([[*ax.get_ylim()] for ax in axs], [])

        for ax in axs:
            ax.set_ylim(min(lims), max(lims))
