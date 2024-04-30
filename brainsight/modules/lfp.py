from typing import Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data

from brainsight import Dataset, Signal
from brainsight.modules.base_module import BaseModule
from brainsight.modules.utils import ms_to_str, draw_activity


class LFP(BaseModule):
    r"""LFP module allows for filtering and plotting
    of the LFP signals

    Parameters
    ----------
    dataset : Dataset
        Dataset instance containing the LFP signals and other
        data modalities.
    low_freq : float or None, optional
        The lower pass-band edge, by default None
    high_freq : float or None, optional
        The upper pass-band edge, by default None
    filter_kwargs : dict or None, optional
        Additional parameters passed to the `mne.filter.filter_data` function, by default `None`
    Notes
    -----
    The filtering applied depends on the provided `low_freq` and `high_freq` according
    to the documentation of `mne.filter.filter_data`:

    Applies a zero-phase low-pass, high-pass, band-pass, or band-stop filter to the channels selected by picks.
    low_freq and high_freq are the frequencies below which and above which,
    respectively, to filter out of the data. Thus the uses are:
        * ``low_freq < high_freq``: band-pass filter
        * ``low_freq > high_freq``: band-stop filter
        * ``low_freq is not None and high_freq is None``: high-pass filter
        * ``low_freq is None and high_freq is not None``: low-pass filter

    For more, see: https://mne.tools/stable/generated/mne.filter.filter_data.html

    Methods
    -------
    get_data(channel, roi, **kwargs)
        Filter the signal of the selected LFP channel within the specified ROI.
    plot(roi, show_activity, show_accelerometer, **kwargs)
        Plot the LFP signals for all channels within the dataset.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters passed to the the
        parent `BaseModule` class.

    Examples
    --------
    >>> dataset = Dataset("path/to/dataset_file.json")
    >>> lfp = LFP(dataset)
    >>> lfp.plot()
    """

    _single_ax = False
    _horizontal = False
    _base_wh = (12, 4)

    def __init__(
        self,
        dataset: Dataset,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        filter_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:

        super().__init__(dataset, **kwargs)
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_kwargs = filter_kwargs or dict()

    def get_data(
        self,
        channel: str,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        **kwargs,
    ) -> np.ndarray:
        """Filter (if either low_freq or high_freq is specified) the LFP
        signal of the selected channel and cut it to the given ROI.

        Parameters
        ----------
        channel : str
            Channel of the LFP for which to process the signal.
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to process the LFP signal.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the
            `mne.filter.filter_data` function.

        Returns
        -------
        np.ndarray
            Values of the (filtered) LFP signal for the selected channel.
        """
        # Filter the full signal (it gets cached, this way we avoid re-computing
        # whenever a new ROI is requested saving time and avoiding edge artefacts)
        signal = super().get_data(
            channel=channel,
            roi=None,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            **self.filter_kwargs,
            **kwargs,
        )

        # Index the resulting signal with the given ROI
        roi_ms = self._process_roi(roi=roi, signal=signal)
        return signal[slice(*roi_ms)]

    @staticmethod
    def _get_data(
        signal: Signal,
        roi_ms: Tuple[int, int],
        low_freq: Optional[float],
        high_freq: Optional[float],
        **kwargs,
    ) -> Signal:
        """Cut the signal to the selected ROI and filter
        (if either low_freq or high_freq is specified) its values.
        By default, 'FIR' filtering is applied.

        Fior more information, see:
        https://mne.tools/stable/generated/mne.filter.filter_data.html

        Parameters
        ----------
        signal : Signal
            Input LFP signal.
        roi_ms : Tuple[int, int]
            Region of interest [in miliseconds] to which the signal is cut.
        low_freq : float
            The lower pass-band edge
        high_freq : float
            The upper pass-band edge

        Other Parameters
        ----------------
        **kwargs
            Additional parameters passed to the
            `mne.filter.filter_data` function.

        Returns
        -------
        Signal
            (Filtered) signal.
        """
        roi_signal = signal[slice(*roi_ms)]
        # Apply filtering if either threshold is set
        if low_freq or high_freq:
            values = filter_data(
                data=roi_signal.values,
                sfreq=roi_signal.sampling_rate,
                l_freq=low_freq,
                h_freq=high_freq,
                verbose=False,
                **kwargs,
            )
        else:
            values = roi_signal.values

        return Signal(
            values=values,
            timestamps=roi_signal.ts,
            sampling_rate=roi_signal.sampling_rate,
        )

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

    def plot(
        self,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]] = None,
        show_activity: bool = True,
        show_accelerometer: bool = False,
        **kwargs,
    ) -> plt.Figure:
        """Plot the (filtered) LFP signals for all channels within the dataset.

        Parameters
        ----------
        roi : Tuple[int, int] or Tuple[str, str], or str, or None
            Region of interest for which to plot the LFP signals.
            Can be specified as:
            - `Tuple[int, int]`; a tuple of timestamps [miliseconds],
            - `Tuple[str, str]`; a tuple of time strings in the "HH:MM:SS" format,
            - `str`; name of the detected activity class,
            - `None`; the entire signal ROI is used.
        show_activity : bool, optional
            Whether to show activity regions found within the selected ROI, by default True.
        show_accelerometer : bool, optional
            Whether to show the accelerometer signal, by default False.

        Returns
        -------
        plt.Figure
            LFP signal figure.
        """
        return super().plot(
            roi=roi,
            show_activity=show_activity,
            show_accelerometer=show_accelerometer,
            **kwargs,
        )

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        show_activity: bool,
        show_accelerometer: bool,
        **kwargs,
    ) -> plt.Figure:
        # Process the signal
        processed = self.get_data(channel=channel, roi=roi)

        # Plot the LFP (and accelerometer) signals
        ax.plot(processed.ts, processed.values, color="steelblue", lw=0.8)
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

        ax.set_xlim(*processed.roi)
        xticks = ax.get_xticks().astype(int)
        labels = [ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*processed.roi)

        if show_activity:
            draw_activity(
                activity_dict=self.dataset.ACTIVITY,
                ax=ax,
                roi_ms=processed.roi,
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
