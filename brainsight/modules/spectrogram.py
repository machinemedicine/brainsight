from typing import Tuple, Optional, Union

import numpy as np
import colorcet
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.modules.base_module import BaseModule
from brainsight.modules.utils import ms_to_str, nanpow2db, draw_activity


class Spectrogram(BaseModule):
    r"""Spectrogram module allows for calculation and plotting
    of multitapered spectrograms of the LFP signals

    Parameters
    ----------
    dataset : Dataset
        Dataset instance containing the LFP signals and other
        data modalities.
    window_sec : float
        Sliding time window size [in seconds] with which to compute the spectrogram.
    frequency_step : float, optional
        Parameter controlling the spectral resolution of the spectrogram. Specifies
        the step size between estimated frequencies, by default `1.0`.
    frequency_band : Tuple[float, float] or None, optional
        Interval of frequencies for which to compute the Spectrogram.
        If `None`, the band is set to (`frequency_step`, Nyquist), by default `None`.
    tfr_kwargs : dict or None, optional
            Additional parameters passed to the `mne.time_frequency.tfr_array_multitaper` function, by default `None`

    Methods
    -------
    get_data(channel, roi, **kwargs)
        Compute a multitaper spectrogram for the selected LFP channel within the specified ROI.
    plot(roi, show_activity, **kwargs)
        Plot the spectrogram for all LFP channels within the dataset.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters passed to the the
        parent `BaseModule` class.

    Examples
    --------
    >>> dataset = Dataset("path/to/dataset_file.json")
    >>> spectrogram = Spectrogram(dataset, window_sec=5.0)
    >>> spectrogram.plot()
    """

    _single_ax = False
    _horizontal = False
    _base_wh = (12, 4)

    def __init__(
        self,
        dataset: Dataset,
        window_sec: float = 10.0,
        frequency_step: float = 1.0,
        frequency_band: Optional[Tuple[float, float]] = None,
        tfr_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.window_sec = window_sec
        self.frequency_step = frequency_step
        self.frequency_band = frequency_band
        self.tfr_kwargs = tfr_kwargs or dict()

    def get_data(
        self,
        channel: str,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a multitaper spectrogram for the selected LFP channel
        within the specified ROI.

        Parameters
        ----------
        channel : str
            Channel of the LFP for which to calculate the spectrogram.
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to calculate the spectrogram.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the the
            `mne.time_frequency.tpr_array_multitaper` function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated spectrogram and corresponding frequencies arrays
        """
        # Compute the full spectrogram (it gets cached, this way we avoid re-computing
        # whenever a new ROI is requested saving time and avoiding edge artefacts)
        spec, freqs = super().get_data(
            channel=channel,
            roi=None,
            window_sec=self.window_sec,
            frequency_step=self.frequency_step,
            frequency_band=self.frequency_band,
            **self.tfr_kwargs,
            **kwargs,
        )

        # Index the result with the given ROI
        signal = self.dataset.LFP[channel]
        roi_ms = self._process_roi(roi=roi, signal=signal)
        mask = signal._mask_slice(s=slice(*roi_ms))
        return spec[:, mask], freqs

    @staticmethod
    def _get_data(
        signal: Signal,
        roi_ms: Tuple[int, int],
        window_sec: float,
        frequency_step: float,
        frequency_band: Optional[Tuple[float, float]],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a multitaper spectrogram for the given Signal
        within the specified ROI.

        Parameters
        ----------
        signal : Signal
            Input signal for which to compute the spectrogram.
        roi_ms : Tuple[int, int]
            Region of interest [in miliseconds] for which
            to compute the spectrogram.
        window_sec : float
            Sliding time window size [in seconds] with which to compute the spectrogram.
        frequency_step : float
            Parameter controlling the spectral resolution of the spectrogram. Specifies
            the step size between estimated frequencies.
        frequency_band : Tuple[float, float] or None
            Interval of frequencies for which to compute the Spectrogram.
            If `None`, the band is set to (`frequency_step`, Nyquist).

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the the
            `mne.time_frequency.tpr_array_multitaper` function.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated spectrogram and corresponding frequency arrays
        """
        sfreq = signal.sampling_rate
        fmin, fmax = frequency_band or (frequency_step, sfreq / 2)

        # Prepare frequencies to estimate
        freqs = np.arange(
            start=fmin,
            stop=fmax + frequency_step,
            step=frequency_step,
        )
        # Make the time window fixed-sized for all frequencies
        n_cycles = freqs * window_sec

        data = signal[slice(*roi_ms)].values[None, None, ...]
        result = tf.tfr_array_multitaper(
            data=data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose=False,
            **kwargs,
        )
        return nanpow2db(result.squeeze()), freqs

    def plot(
        self,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]] = None,
        show_activity: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """Plot the spectrogram for all LFP channels within the dataset.

        Parameters
        ----------
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to plot the spectrogram.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.
        show_activity : bool, optional
            Whether to show activity regions found within the selected ROI, by default True.

        Returns
        -------
        plt.Figure
            Spectrogram figure.
        """
        return super().plot(roi=roi, show_activity=show_activity, **kwargs)

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        channel: str,
        signal: Signal,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        show_activity: bool,
        **kwargs,
    ):
        # Compute the spectrogram
        spec, freqs = self.get_data(channel=channel, roi=roi)

        # Draw the spectrogram
        roi_ms = self._process_roi(roi=roi, signal=signal)
        extent = [*roi_ms, *freqs[[-1, 0]]]
        im = ax.imshow(spec, extent=extent, aspect="auto")
        im.set_cmap(colormaps["cet_rainbow4"])
        ax.invert_yaxis()

        # Annotate the channel name and set y-label
        ax.annotate(
            f"Channel:\n{channel}",
            xy=(0, 1),
            xytext=(3, -3),
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
            color="white",
            fontweight="medium",
            fontsize=9,
        )
        ax.set_ylabel(f"Frequency [Hz]")

        # Set the x-ticks and labels
        xticks = ax.get_xticks().astype(int)
        labels = [ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*roi_ms)

        # Draw activity regions
        if show_activity:
            ax_ticks = draw_activity(
                activity_dict=self.dataset.ACTIVITY,
                ax=ax,
                roi_ms=roi_ms,
                ax_i=ax_i,
                alpha=0.2,
                color="white",
                zorder=1,
            )

        # Axis formatting depending on their placing
        if ax_i:
            ax.set_xlabel("Time [HH:MM:SS]")
            ax.tick_params(labeltop=False)

        else:
            ax.tick_params(labelbottom=False)

        return im, ax_ticks

    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        vals = sum([[im.norm.vmin, im.norm.vmax] for im, _ in rets], [])

        for ax, (im, ax_ticks) in zip(axs, rets):
            im.set_clim(vmin=min(vals), vmax=max(vals))

            divider = make_axes_locatable(ax_ticks)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cax.set_visible(False)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cax.set_visible(False)

        cax.set_visible(True)
        fig.colorbar(
            im,
            cax=cax,
            label="PSD [$\mathrm{dB}\ $$\mathrm{{µV²/Hz}}$]",
        )
