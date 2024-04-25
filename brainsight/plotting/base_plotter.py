import abc
from typing import Tuple, Union, Optional
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from brainsight import Dataset, Signal
from brainsight.plotting.utils import str_to_ms


class BasePlotter:
    _single_ax: bool = False
    _horizontal: bool = False
    _base_wh: Tuple[int, int] = (10, 3)
    _use_cache: bool = True

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._data_cache = dict()

    @property
    def data(self) -> dict:
        return deepcopy(self._data_cache)

    def get_data(self, channel: str, signal: Signal, **kwargs):
        if channel in self._data_cache and self._use_cache:
            data = deepcopy(self._data_cache[channel])
        else:
            data = self._get_data(signal, **kwargs)
            self._data_cache[channel] = data
        return data

    def _setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        "Prepare the Figure and Axes"
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
        self, roi: Optional[Union[Tuple[int, int], Tuple[str, str], str]]
    ) -> Optional[Tuple[int, int]]:
        """Process ROI input into a tuple of miliseconds or leave as `None`."""

        # Return None if ROI is not provided
        if roi is None:
            out = None
        # If a tuple of integers, return the ROI itself
        elif isinstance(roi, tuple) and all(isinstance(r, int) for r in roi):
            out = roi
        # If a tuple of strings, format them into miliseconds
        elif isinstance(roi, tuple) and all(isinstance(r, str) for r in roi):
            out = [str_to_ms(r) for r in roi]
        # If a string return the detected activity ROI
        elif isinstance(roi, str) and roi in self.dataset.ACTIVITY.keys():
            out = self.dataset.ACTIVITY[roi]
        else:
            raise TypeError(
                "`roi` is expected to be one of: (Tuple[int, int] | Tuple[str, str] | str | None). "
                "For a tuple of strings, specify the times in a `%H:%M:%S` format."
                "For a string, use one of the detected activity labels."
            )
        return deepcopy(out)

    def plot(
        self, roi: Optional[Union[Tuple[int, int], str]] = None, **kwargs
    ):
        """Plot"""
        fig, axs = self._setup_figure()

        roi = self._process_roi(roi=roi)

        rets = []
        for (channel, signal), (i, ax) in zip(
            self.dataset.LFP.items(), enumerate(axs)
        ):
            ret = self._plot_ax(
                ax=ax,
                ax_i=i,
                signal=signal,
                channel=channel,
                roi=roi or signal.roi,
                **kwargs
            )

            rets.append(ret)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self._plot_fig(fig=fig, axs=axs, rets=rets, **kwargs)

        fig.tight_layout()
        return fig

    @abc.abstractmethod
    def _plot_ax(
        self, ax: plt.Axes, signal: Signal, channel: str, roi=Tuple[int, int]
    ):
        """Function for plotting on a given Axis using the provided Signal"""

    @abc.abstractmethod
    def _plot_fig(self, fig: plt.Figure, axs: np.ndarray, rets: list):
        """Function for plotting on all Axes and the Figure with an axes to
        values returned by the `_plot_ax()` method"""

    @abc.abstractmethod
    def _get_data(self, signal: Signal, **kwargs):
        """Prepare Signal data for plotting"""
