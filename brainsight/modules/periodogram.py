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
        If `None`, the band is set to (0, Nyquist), by default `None`.
    bandwidth : float or None, optional
        Frequency bandwidth of the multi-taper window function in Hz.
        For a given frequency, frequencies at ± bandwidth / 2 are smoothed together.
        If `None`, the value is set to `8 * (signal.samplig_rate / len(signal))`, by default `None`
    adaptive : bool, optional
        Use adaptive weights to combine the tapered spectra into PSD
        (might be slower), by default False.
    brainwave_bands : Dict[ str, Tuple[float, float] ], optional
        Dictionary of LFP frequency bands used for plotting.
        The specified bands and their corresponding power are highlighted.
        By default the bands are set to:
        ``{"delta": (0.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 32.0), "gamma": (32.0, 120.0)}``
    psd_kwargs : dict or None, optional
        Additional parameters passed to the `mne.time_frequency.psd_array_multitaper` function, by default `None`

    Methods
    -------
    get_data(channel, roi, **kwargs)
        Compute a multitaper PSD for the selected LFP channel within the specified ROI.
    plot(roi, norm, **kwargs)
        Plot the periodogram for all LFP channels within the dataset.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters passed to the the
        parent `BaseModule` class.

    Examples
    --------
    >>> dataset = Dataset("path/to/dataset_file.json")
    >>> periodogram = Periodogram(dataset)
    >>> periodogram.plot()
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
        psd_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.frequency_band = frequency_band
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.brainwave_bands = brainwave_bands
        self.psd_kwargs = psd_kwargs or dict()

    def get_data(
        self,
        channel: str,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        **kwargs,
    ):
        """Compute a multitaper PSD for the selected LFP channel
        within the specified ROI.

        Parameters
        ----------
        channel : str
            Channel of the LFP for which to calculate the PSD.
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to calculate the PSD.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the the
            `mne.time_frequency.psd_array_multitaper` function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated power and corresponding frequencies arrays

        Examples
        --------
        >>> dataset = Dataset("path/to/dataset_file.json")
        >>> dataset.LFP
        Dataset:
        - ZERO_TWO_RIGHT
        - ZERO_TWO_LEFT
        >>> dataset.ACTIVITY
        ACTIVITY:
        - LEG_AGILITY_RIGHT
        - FINGER_TAPPING_LEFT
        - GAIT_TOWARDS_CAMERA
        - HAND_MOVEMENTS_RIGHT
        - ARISING_FROM_CHAIR
        - FINGER_TAPPING_RIGHT
        - GAIT_FROM_CAMERA
        - TOE_TAPPING_LEFT
        - TOE_TAPPING_RIGHT
        - POSTURAL_TREMOR_OF_HANDS_LEFT
        - LEG_AGILITY_LEFT
        - HAND_MOVEMENTS_LEFT
        >>> periodogram = Periodogram(dataset, frequency_band=(5.0, 50.0))
        >>> psds, freqs = periodogram.get_data(channel="ZERO_TWO_RIGHT", roi="ARISING_FROM_CHAIR")
        """

        return super().get_data(
            channel=channel,
            roi=roi,
            frequency_band=self.frequency_band,
            bandwidth=self.bandwidth,
            adaptive=self.adaptive,
            **self.psd_kwargs,
            **kwargs,
        )

    @staticmethod
    def _get_data(
        signal: Signal,
        roi_ms: Tuple[int, int],
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
        roi_ms : Tuple[int, int]
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
        fmin, fmax = frequency_band or (0, sfreq / 2)

        data = signal[slice(*roi_ms)].values
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
            ylabel = "PSD [dB µV²/Hz]"
        elif norm == "density":
            psds /= psds.sum()
            ylabel = "PSD"
        elif norm == "power":
            ylabel = "PSD [µV²/Hz]"
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

        ax.axhline(y=0, xmin=0, xmax=1, ls=":", lw="1.0", color="black")
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
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]] = None,
        norm: str = "density",
        **kwargs,
    ) -> plt.Figure:
        """Plot the periodogram for all LFP channels within the dataset.

        Parameters
        ----------
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to plot the PSD.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.
        norm : {"density", "power", "dB"}, optional
            Mode of the plotting norm. If `"density"`, the plot is normalised
            per channel so that the area sums up to 1.0. If `"power"`, the raw power
            is drawn. If `"dB"`, the power gets converted to dB, by default `"desnity"`.

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
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        norm: str,
        **kwargs,
    ):
        psds, freqs = self.get_data(channel=channel, roi=roi)

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
            ax.set_ylim(min(min(lims), 0), max(lims))
