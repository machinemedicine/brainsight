from typing import Tuple, Optional, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.modules.base_module import BaseModule
from brainsight.modules.utils import nanpow2db
import brainsight.modules.defaults as defaults


class Periodogram(BaseModule):
    r"""Periodogram module allows for calculation and plotting
    of multitapered periodograms of the LFP signals

    Parameters
    ----------
    dataset : Dataset
        Dataset instance containing the LFP signals and other
        data modalities.
    frequency_band : Tuple[float, float] or None, optional
        Interval of frequencies for which to compute the PSD.
        If `None`, the band is set to (0, Nyquist), by default None.
    bandwidth : float or None, optional
        Frequency bandwidth of the multi-taper window function in Hz.
        For a given frequency, frequencies at ± bandwidth / 2 are smoothed together.
        If `None`, the value is set to `8 * (signal.samplig_rate / len(signal))`, by default None
    adaptive : bool, optional
        Use adaptive weights to combine the tapered spectra into PSD
        (might be slower), by default False.
    brainwave_bands : Dict[ str, Tuple[float, float] ], optional
        Dictionary of LFP frequency bands used for plotting.
        The specified bands and their corresponding power are highlighted.
        By default the bands are set to:
        ``{"delta": (0.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 32.0), "gamma": (32.0, 120.0)}``

    """

    _single_ax = False
    _horizontal = True
    _base_wh = (6, 4)

    def __init__(
        self,
        dataset: Dataset,
        frequency_band: Optional[Tuple[float, float]] = None,
        bandwidth: Optional[float] = None,
        adaptive: bool = False,
        brainwave_bands: Dict[
            str, Tuple[float, float]
        ] = defaults.BRAINWAVE_BANDS,
        **kwargs,
    ) -> None:
        super().__init__(dataset)
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.brainwave_bands = brainwave_bands

    def get_data(
        self,
        channel: str,
        roi: Optional[Union[Tuple[int, int], str]],
        **kwargs,
    ):
        """Compute a multitaper PSD for the selected LFP channel
        within the specified ROI.

        Parameters
        ----------
        channel : str
            Channel of the LFP for which to calculate the PSD.
        roi : Tuple[int, int] or str, or None
            Region of interest for which to calculate the PSD.
            Can be specified as:



        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the the
            `mne.time_frequency.psd_array_multitaper` function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated power and corresponding frequencies arrays
        """

        return super().get_data(
            channel=channel,
            roi=roi,
            frequency_band=self.frequency_band,
            bandwidth=self.bandwidth,
            adaptive=self.adaptive,
            **kwargs,
        )

    @staticmethod
    def _get_data(
        signal: Signal,
        roi: Tuple[int, int],
        frequency_band: Tuple[float, float],
        bandwidth: Optional[float],
        adaptive: bool,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a multitaper PSD for the given Signal
        within the specified ROI.

        Parameters
        ----------
        signal : Signal
            Input signal for which to compute the PSD.
        roi : Tuple[int, int]
            Region of interest [in miliseconds] for which
            to compute the PSD.
        frequency_band : Tuple[float, float]
            Interval of frequencies for which to compute
            the PSD. If `None`, the band is set to (0, Nyquist).
        bandwidth : float | None
            Frequency bandwidth of the multi-taper window function in Hz.
            For a given frequency, frequencies at ± bandwidth / 2 are smoothed together.
            If `None`, the value is set to `8 * (sfreq / len(signal))`.
        adaptive : bool
            Use adaptive weights to combine the tapered spectra into PSD
            (might be slower).

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the the
            `mne.time_frequency.psd_array_multitaper` function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated power and corresponding frequency arrays
        """
        sfreq = signal.sampling_rate
        fmin, fmax = frequency_band

        data = signal[slice(*roi)].values
        psds, freqs = tf.psd_array_multitaper(
            x=data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            bandwidth=bandwidth,
            adaptive=adaptive,
            verbose=False,
            **kwargs,
        )
        return psds, freqs

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
        for i, (name, (start, stop)) in enumerate(
            self.brainwave_bands.items()
        ):
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

        ax.legend(
            fontsize=8,
            loc="lower right",
            ncols=3,
            bbox_to_anchor=(1, 1.02),
            borderaxespad=0.0,
            # loc="upper right",
            title="Band: Area",
            alignment="left",
            title_fontsize=8,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return band_areas

    def plot(
        self,
        roi: Optional[Union[Tuple[int, int], str]] = None,
        norm: str = "density",
        **kwargs,
    ) -> plt.Figure:
        """Plot the periodogram for all LFP channels within the dataset.

        Parameters
        ----------
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for the PSD calculation. Can be one of:
            tuple of integers (miliseconds); tuple of strings in the "HH:MM:SS" format;
            name of the detected activity, or `None`. If `None`, the LFP signal's ROI is used.
        norm : {"density", "power", "dB"}, optional
            Mode of the plotting norm. If "density", the plot is normalised
            per channel so that the area sums up to 1.0

        Returns
        -------
        plt.Figure
            Periodogram figure.
        """
        return super().plot(roi=roi, norm=norm, **kwargs)

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        norm: str = "density",
        **kwargs,
    ):
        psds, freqs = self.get_data(channel=channel, signal=signal, roi=roi)

        psds, ylabel = self._format_psds(psds=psds, norm=norm)

        ax.plot(freqs, psds, color="black", lw=0.8, zorder=3)

        ax.set_title(f"Channel:\n{channel}", ha="left", x=0, fontsize=8, y=1)

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
