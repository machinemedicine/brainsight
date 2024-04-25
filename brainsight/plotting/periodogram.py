from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter
from brainsight.plotting.utils import nanpow2db

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

    def _format_psds(
        self, psds: np.ndarray, norm: str
    ) -> Tuple[np.ndarray, str]:
        "Format the PSDs and create a correspondig label"
        if norm == "dB":
            psds = nanpow2db(psds)
            ylabel = "PSD [$\mathrm{dB}\ $$\mathrm{{µV²/Hz}}$]"
        elif norm == "density":
            psds /= psds.sum()
            ylabel = "PSD"
        elif norm == "power":
            ylabel = "PSD [$\mathrm{{µV²/Hz}}$]"
        else:
            raise ValueError()
        return psds, ylabel

    def _draw_bands(
        self, ax: plt.Axes, freqs: np.ndarray, psds: np.ndarray
    ) -> Dict[str, float]:
        """Draw frequency bands and calcualte their area under the curve"""
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()

        band_areas = dict()
        for i, (name, (start, stop)) in enumerate(BRAINWAVE_BANDS.items()):
            (overlap,) = np.where((start <= freqs) & (freqs < stop))
            color = colormaps["Pastel1"](i)

            if overlap.size:
                # area = np.trapz(y=psds[overlap], x=freqs[overlap])
                area = np.sum(psds[overlap])
                label = f"{name}: {area:.2f}"

                # Interpolate PSDs for band ends for plotting
                start_v, stop_v = np.interp([start, stop], xp=freqs, fp=psds)
                x = np.concatenate(([start], freqs[overlap], [stop]))
                y1 = np.concatenate(([start_v], psds[overlap], [stop_v]))

                ax.axvspan(
                    xmin=start,
                    xmax=stop,
                    zorder=1,
                    alpha=0.2,
                    color=color,
                )

                ax.fill_between(
                    x=x,
                    y1=y1,
                    y2=0,
                    color=color,
                    alpha=0.8,
                    label=label,
                    zorder=2,
                )

                band_areas[name] = area

        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return band_areas

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        norm: str = "density",
    ):
        psds, freqs = self.get_data(channel=channel, signal=signal, roi=roi)

        psds, ylabel = self._format_psds(psds=psds, norm=norm)

        ax.plot(freqs, psds, color="black", lw=0.8, zorder=3)

        ax.set_title(f"Channel: {channel}", ha="left", x=0, fontsize=8)

        ax.set_xlabel("Frequency [Hz]")

        areas = self._draw_bands(ax, freqs=freqs, psds=psds)

        ax.grid(color="black", alpha=0.1, zorder=1)

        ax.set_xlim(self.frequency_band)

        if ax_i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(ylabel)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return areas

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        lims = sum([[*ax.get_ylim()] for ax in axs], [])

        for ax in axs:
            ax.set_ylim(0, max(lims))
