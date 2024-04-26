import abc
from typing import Tuple, Union, Optional
from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from brainsight import Dataset, Signal
from brainsight.modules.utils import str_to_ms


class BaseModule(abc.ABC):
    _single_ax: bool = False
    _horizontal: bool = False
    _base_wh: Tuple[int, int] = (10, 3)

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        """Base class for plotting modules.

        Parameters
        ----------
        dataset : Dataset
            Dataset instance containing the LFP signals and other
            data modalities.
        """
        self.dataset = dataset
        self._clear_cache()

    @property
    def data(self) -> dict:
        data = deepcopy(self._data_cache)
        return {ch: next(iter(v.values())) for ch, v in data.items()}

    def _clear_cache(self, channel: Optional[str] = None) -> None:
        """Clears the cached data"""
        if channel is None:
            self._data_cache = defaultdict(dict)
        else:
            self._data_cache[channel] = dict()

    def _setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Prepare the figure and axes of the plot."""
        nsig = len(self.dataset.LFP)
        n = 1 if self._single_ax else nsig
        ncols, nrows = (n, 1) if self._horizontal else (1, n)
        w, h = self._base_wh

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(w * ncols, h * nrows),
        )
        axs = np.repeat(axs, nsig) if n == 1 else axs
        return fig, axs

    def _process_roi(
        self,
        roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]],
        signal: Signal,
    ) -> Optional[Tuple[int, int]]:
        """Process ROI input into a tuple of miliseconds or
        return the signal's ROI if `None`."""
        signal_roi = signal.roi
        # Return the Signal's ROI if a None ROI is not provided
        if roi is None:
            out = signal_roi
        # If a string return the detected activity ROI
        elif isinstance(roi, str) and roi in self.dataset.ACTIVITY.keys():
            out = self.dataset.ACTIVITY[roi]
        # If a tuple of integers, return the ROI itself
        elif isinstance(roi, (tuple, list)):
            if not len(roi) == 2:
                raise ValueError(
                    f"ROI tuple has to have 2 elements, got ({len(roi)})"
                )
            if not len(set(type(r) for r in roi)) == 1:
                raise ValueError(
                    "ROI tuple elements have to be of single type"
                )
            if not all(isinstance(r, (int, str)) for r in roi):
                raise ValueError(
                    "ROI tuple elements have to be both integers (miliseconds)"
                    " or strings in the 'HH:MM:SS' format"
                )
            out = roi
        else:
            raise TypeError(
                "`roi` is expected to be one of: (Tuple[int, int] | Tuple[str, str] | str | None). "
                "For a tuple of strings, specify the times in a `%H:%M:%S` format."
                "For a string, use one of the detected activity labels."
            )
        return tuple(out)

    @abc.abstractmethod
    def get_data(
        self,
        channel: str,
        roi: Optional[Union[Tuple[int, int], str]],
        **kwargs,
    ):
        """Abstract method for the extraction of plotting data.
        Its interface needs to be implemented by the child class."""
        signal = self.dataset.LFP[channel]
        roi_ms = self._process_roi(roi=roi, signal=signal)

        if roi_ms in self._data_cache[channel]:
            data = deepcopy(self._data_cache[channel][roi_ms])

        else:
            self._clear_cache(channel=channel)
            data = self._get_data(signal=signal, roi_ms=roi_ms, **kwargs)
            self._data_cache[channel][roi_ms] = data

        return data

    @staticmethod
    @abc.abstractmethod
    def _get_data(signal: Signal, **kwargs):
        """Static function preparing Signal data for plotting."""

    @abc.abstractmethod
    def plot(
        self, roi: Optional[Union[Tuple[int, int], str]] = None, **kwargs
    ):
        """Abstract method for plotting. Its interface needs
        to be implemented by the child class."""
        fig, axs = self._setup_figure()

        rets = []
        for i, (ax, channel) in enumerate(zip(self.dataset.LFP, axs)):
            ret = self._plot_ax(
                ax=ax,
                ax_i=i,
                channel=channel,
                roi=roi,
                **kwargs,
            )

            rets.append(ret)

        self._plot_fig(fig=fig, axs=axs, rets=rets, **kwargs)

        fig.tight_layout()
        return fig

    @abc.abstractmethod
    def _plot_ax(
        self,
        ax: plt.Axes,
        signal: Signal,
        channel: str,
        roi=Tuple[int, int],
        **kwargs,
    ):
        """Function for plotting on a given axis using the provided Signal"""

    @abc.abstractmethod
    def _plot_fig(
        self, fig: plt.Figure, axs: np.ndarray, rets: list, **kwargs
    ):
        """Function for plotting on all Axes and the Figure with an axes to
        values returned by the `_plot_ax()` method"""
