from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter

BRAINWAVE_BANDS = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 32.0),
    "gamma": (32.0, 120.0),
}


class Periodogram(BasePlotter):
    _single_ax = False
    _horizontal = True
    _base_wh = (6, 4)
    _use_cache = False

    def __init__(
        self,
        dataset: Dataset,
        frequency_band: Optional[Tuple[float, float]] = None,
        bandwidth: Optional[float] = None,
        adaptive: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
        self.adaptive = adaptive

    def _get_data(self, signal: Signal, roi: Tuple[int, int]):
        sfreq = signal.sampling_rate
        fmin, fmax = self.frequency_band

        data = signal[slice(*roi)].values[None, ...]
        psds, freqs = tf.psd_array_multitaper(
            x=data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            bandwidth=self.bandwidth,
            adaptive=self.adaptive,
            normalization="length",
            verbose=False,
        )
        return psds.squeeze(), freqs

    def _draw_bands(self, ax: plt.Axes, roi: Tuple[int, int]):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()

        for i, (name, (start, stop)) in enumerate(BRAINWAVE_BANDS.items()):
            ax.axvspan(
                xmin=start,
                xmax=stop,
                zorder=1,
                alpha=0.8,
                color=colormaps["Pastel1"](i),
                label=name,
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
        norm: str,
    ):
        psds, freqs = self.get_data(channel=channel, signal=signal, roi=roi)

        if norm == "dB":
            psds = self.nanpow2db(psds)
            ylabel = "Power [$\mathrm{dB}\ $$\mathrm{{µV²/Hz}}$)]"
        elif norm == "density":
            psds /= psds.sum()
            ylabel = "PSD"
        elif norm == "power":
            ylabel = "Power [$\mathrm{{µV²/Hz}}$)]"
        else:
            raise ValueError()

        line = ax.plot(freqs, psds)

        ax.set_title(f"Channel: {channel}", ha="left", x=0, fontsize=8)

        ax.set_xlabel("Frequency [Hz]")

        self._draw_bands(ax, roi=roi)

        ax.grid(color="white", alpha=0.5)

        ax.set_xlim(self.frequency_band)

        if ax_i:
            ax.tick_params(labelleft=False)
            ax.legend()
        else:
            ax.set_ylabel(ylabel)

        # if show_activity:
        #     self._draw_activity(ax=ax, roi=roi)

        return line

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        lims = sum([[*ax.get_ylim()] for ax in axs], [])

        for ax in axs:
            ax.set_ylim(min(lims), max(lims))

    def nanpow2db(self, y):
        """Power to dB conversion"""
        y[y == 0] = np.nan
        return 10 * np.log10(y)
