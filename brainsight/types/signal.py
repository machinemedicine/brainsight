from typing import Tuple, List, Dict, Union, Optional

import numpy as np

from brainsight.types.utils import Alias


class Signal:
    r"""A class for convenient handling of dataset's signals.
    Each signal object contains values and timestamps of the signal
    ensuring accurate calculations and plotting.

    Parameters
    ----------
    values : List[float]
        List of signal values.
    timestamps : List[float]
        List of timestamps corresponding to the signal values.
    sampling_rate : Optional[Union[float, int]], optional
        Sampling rate of the signal [in Hz]. If ``None``, the sampling
        rate will be inferred as from the median timestamp difference,
        by default ``None``.

    Attributes
    ----------
    values : np.ndarray
        Numpy array containing signal values.
    timestamps : np.ndarray
        Numpy array containing signal timestamps.
    ts : np.ndarray
        Alias for ``timestamps``.
    sampling_rate : float
        Sampling rate of the signal.
    SamplingRate : float
        Alias for ``sampling_rate``.
    roi : Tuple[int, int]
        Tuple indicating the first and last timestamp of the signal.
    ROI : Tuple[int, int]
        Alias for ``roi``.

    Raises
    ------
    ValueError
        ``values`` and ``timestamps`` are of different length.
    """

    def __init__(
        self,
        values: List[float],
        timestamps: List[float],
        sampling_rate: Optional[Union[float, int]] = None,
    ) -> None:
        if len(values) != len(timestamps):
            raise ValueError(
                "`values` and `timestamps` of a `Signal` have to be of equal length."
            )
        self._values = np.array(values)
        self._timestamps = np.array(timestamps)

        # Infers the median sampling_rate from timestamps if it's not
        # explicitly specified.
        if sampling_rate is None:
            sampling_rate = np.median(1000 / np.diff(self.timestamps))

        self._sampling_rate = float(sampling_rate)

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps.copy()

    ts = Alias("timestamps")

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    SamplingRate = Alias("sampling_rate")

    @property
    def roi(self) -> Tuple[int, int]:
        return (self.ts.min(), self.ts.max())

    ROI = Alias("roi")

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Signal(N: {}, ROI: {}, SamplingRate: {}Hz)".format(
            len(self), self.roi, round(self.sampling_rate, ndigits=2)
        )

    def __getitem__(self, key: slice):
        if isinstance(key, slice):
            return self._index_slice(index=key)
        else:
            raise TypeError("Singal can be indexed with one of `(slice)`")

    def _mask_slice(self, s: slice) -> np.ndarray:
        """Return a binary mask indicating where the signal overlaps the time slice."""
        start_mask = self.timestamps >= s.start
        stop_mask = self.timestamps <= s.stop
        mask = start_mask * stop_mask

        if not mask.sum():
            raise IndexError(
                "Indexed region ({}) does not overlap the Signal's ROI ({})".format(
                    (s.start, s.stop), self.roi
                )
            )
        return mask

    def _index_slice(self, index: slice):
        """Indexes the signal based on timestamps' slice, returns a new Signal"""
        mask = self._mask_slice(s=index)

        return self.__class__(
            values=self.values[mask],
            timestamps=self.timestamps[mask],
            sampling_rate=self.sampling_rate,
        )

    def shift(self, shift: int):
        """Adds a `shift` to the Signal's timestamps. Returns a new Signal instance."""
        return self.__class__(
            values=self.values,
            timestamps=self.timestamps + shift,
            sampling_rate=self.sampling_rate,
        )

    def to_dict(self) -> Dict[str, list]:
        """Converts the signal to a dictionary with serialisable typing."""
        return {
            "values": self.values.tolist(),
            "timestamps": self.timestamps.tolist(),
            "sampling_rate": self.sampling_rate,
        }
