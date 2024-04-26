from typing import Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data

from brainsight import Dataset, Signal
from brainsight.modules.base_module import BaseModule
from brainsight.modules.utils import ms_to_str, draw_activity


class LFP(BaseModule):
    _single_ax = False
    _horizontal = False
    _base_wh = (12, 4)

    def __init__(
        self,
        dataset: Dataset,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset)
        self.low_freq = low_freq
        self.high_freq = high_freq

    def get_data(self, channel: str, use_cache: bool = True, **kwargs):

        return super().get_data(channel=channel, use_cache=use_cache, **kwargs)

    def _get_data(self, signal: Signal, **kwargs):
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

    def _draw_accelerometer(self, ax: plt.Axes) -> plt.Axes:
        """Draws the Accelerometer signal on a new twin Ax"""
        ax_acc = ax.twinx()

        # Plot the accelerometer signal.
        ax_acc.plot(
            self.dataset.ACCELEROMETER.ts,
            self.dataset.ACCELEROMETER.values,
            color="orange",
            lw=0.8,
        )
        ax_acc.set_ylabel("Acceleration [g]")

        ax_acc.spines["top"].set_visible(False)

        return ax_acc

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        show_activity: bool = True,
        show_accelerometer: bool = True,
        **kwargs,
    ):
        values = self.get_data(channel=channel, signal=signal)

        ax.plot(signal.ts, values, color="steelblue", lw=0.8)

        ax_acc = self._draw_accelerometer(ax=ax)
        if not show_accelerometer:
            ax_acc.set_visible(False)

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

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax_acc

    def _plot_fig(
        self,
        fig: plt.Figure,
        axs: np.ndarray,
        rets: list,
        **kwargs,
    ):
        lims = sum([[*ax.get_ylim()] for ax in axs], [])
        ymin, ymax = min(lims), max(lims)
        for ax in axs:
            ax.set_ylim(ymin, ymax)

        # Align y=0 of the new axis with the old's one.
        # NOTE: Assumes both axes cover y=0. It should normally be the case.
        lims = sum([[*ax_acc.get_ylim()] for ax_acc in rets], [])
        ymax_acc = max(lims)
        ymin_acc = ymax_acc * (ymin / ymax)
        for ax_acc in rets:
            ax_acc.set_ylim(ymin_acc, ymax_acc)
