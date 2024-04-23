from typing import Tuple, Optional

import numpy as np
import colorcet
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter


class Spectrogram(BasePlotter):
    _single_ax = False
    _horizontal = False
    _base_wh = (10, 3)
    _use_cache = True

    def __init__(
        self,
        dataset: Dataset,
        window_sec: float,
        frequency_step: float,
        frequency_band: Optional[Tuple[float, float]],
    ) -> None:
        super().__init__(dataset)
        self.window_sec = window_sec
        self.frequency_step = frequency_step
        self.frequency_band = frequency_band

    def _get_data(self, signal: Signal):
        sfreq = signal.sampling_rate
        start, stop = self.frequency_band or (self.frequency_step, sfreq / 2)

        freqs = np.arange(start=start, stop=stop, step=self.frequency_step)
        n_cycles = freqs * self.window_sec

        data = signal.values[None, None, ...]
        result = tf.tfr_array_multitaper(
            data=data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose=False,
        )
        return self.nanpow2db(result.squeeze()), freqs

    def _draw_activity(self, ax: plt.Axes, roi: Tuple[int, int]):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        roi_range = set(range(*roi))

        for a_s, a_e in self.dataset.ACTIVITY.values():
            overlap = set(range(a_s, a_e)).intersection(roi_range)

            if overlap:
                ax.axvspan(
                    xmin=a_s,
                    xmax=a_e,
                    zorder=1,
                    alpha=0.2,
                    color="white",
                )
                ax.vlines(
                    [a_s, a_e],
                    ymin,
                    ymax,
                    zorder=3,
                    color="w",
                    alpha=0.3,
                    ls=":",
                )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        show_activity: bool,
    ):
        spec, freqs = self.get_data(channel=channel, signal=signal)

        extent = [*signal.roi, *freqs[[-1, 0]]]

        im = ax.imshow(spec, extent=extent, aspect="auto")
        im.set_cmap(colormaps["cet_rainbow4"])

        ax.set_title(f"Channel: {channel}", ha="left", x=0, fontsize=8)

        ax.set_ylabel("Frequency [Hz]")
        ax.invert_yaxis()
        ax.set_xlim(*roi)

        xticks = ax.get_xticks().astype(int)
        labels = [self.ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*roi)

        if show_activity:
            self._draw_activity(ax=ax, roi=roi)

        if ax_i:
            ax.set_xlabel("Time [HH:MM:SS]")
            ax.tick_params(labeltop=False)

        else:
            ax.tick_params(labelbottom=False)

        return im

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        vals = sum([[im.norm.vmin, im.norm.vmax] for im in rets], [])

        for ax, im in zip(axs, rets):
            im.set_clim(vmin=min(vals), vmax=max(vals))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.02)
            cax.set_visible(False)

        cax.set_visible(True)
        fig.colorbar(
            im,
            cax=cax,
            label="PSD (dB)",
        )

    def nanpow2db(self, y):
        """Power to dB conversion"""
        y[y == 0] = np.nan
        return 10 * np.log10(y)
