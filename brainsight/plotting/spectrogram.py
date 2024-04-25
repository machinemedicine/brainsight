from typing import Tuple, Optional

import numpy as np
import colorcet
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne.time_frequency as tf

from brainsight import Dataset, Signal
from brainsight.plotting.base_plotter import BasePlotter
from brainsight.plotting.utils import ms_to_str, nanpow2db, draw_activity


class Spectrogram(BasePlotter):
    _single_ax = False
    _horizontal = False
    _base_wh = (12, 4)
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
        fmin, fmax = self.frequency_band or (self.frequency_step, sfreq / 2)

        # Prepare frequencies to estimate
        freqs = np.arange(
            start=fmin,
            stop=fmax + self.frequency_step,
            step=self.frequency_step,
        )
        # Make the time window fixed-sized for all frequencies
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
        return nanpow2db(result.squeeze()), freqs

    def _draw_activity(self, ax: plt.Axes, roi: Tuple[int, int]):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        roi_range = set(range(*roi))

        activities = dict(
            sorted(self.dataset.ACTIVITY.items(), key=lambda item: item[1][0])
        )
        i = 1
        for name, (a_s, a_e) in activities.items():
            overlap = set(range(a_s, a_e)).intersection(roi_range)
            if overlap:
                ax.axvspan(
                    xmin=a_s,
                    xmax=a_e,
                    zorder=1,
                    alpha=0.2,
                    color="white",
                    label=f"{i}: {name}",
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
                alpha = 1.0 if (i % 2) else 0.8
                ax.text(
                    sum(overlap) / len(overlap),
                    ymax,
                    s=i,
                    ha="center",
                    va="bottom",
                    alpha=alpha,
                )

                i += 1

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _plot_ax(
        self,
        ax: plt.Axes,
        ax_i: int,
        signal: Signal,
        channel: str,
        roi: Tuple[int, int],
        show_activity: bool = True,
    ):
        spec, freqs = self.get_data(channel=channel, signal=signal)

        extent = [*signal.roi, *freqs[[-1, 0]]]

        im = ax.imshow(spec, extent=extent, aspect="auto")
        im.set_cmap(colormaps["cet_rainbow4"])

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
        ax.invert_yaxis()
        ax.set_xlim(*roi)

        xticks = ax.get_xticks().astype(int)
        labels = [ms_to_str(t) for t in xticks]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim(*roi)

        if show_activity:
            ax_ticks = draw_activity(
                activity_dict=self.dataset.ACTIVITY,
                ax=ax,
                roi=roi,
                ax_i=ax_i,
                alpha=0.2,
                color="white",
                zorder=1,
            )

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
