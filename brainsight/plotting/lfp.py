from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter
from brainsight.plotting.utils import ms_to_str, draw_activity


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

        ax.annotate(
            f"Channel:\n{channel}",
            xy=(0, 1),
            xytext=(3, -3),
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
            color="black",
            fontweight="medium",
            fontsize=9,
        )

        ax.set_ylabel("LFP [µV]")
        ax.set_xlim(*roi)

        xticks = ax.get_xticks().astype(int)
        labels = [ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*roi)

        if show_activity:
            draw_activity(
                activity_dict=self.dataset.ACTIVITY,
                ax=ax,
                roi=roi,
                ax_i=ax_i,
                alpha=0.1,
                color="purple",
                zorder=1,
            )

        if ax_i:
            ax.set_xlabel("Time [HH:MM:SS]")
            ax.tick_params(labeltop=False)

        else:
            ax.tick_params(labelbottom=False)

        return None

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        lims = sum([[*ax.get_ylim()] for ax in axs], [])

        for ax in axs:
            ax.set_ylim(min(lims), max(lims))
