from typing import Tuple, List, Dict, Union, Optional

import numpy as np


class Signal:
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

    @property
    def ts(self) -> np.ndarray:
        return self.timestamps

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def roi(self) -> Tuple[int, int]:
        return (self.ts.min(), self.ts.max())

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

    def _index_slice(self, index: slice):
        start_mask = self.timestamps >= index.start
        stop_mask = self.timestamps <= index.stop
        mask = start_mask * stop_mask

        if not mask.sum():
            raise IndexError(
                "Indexed region ({}) does not overlap the Signal's ROI ({})".format(
                    (index.start, index.stop), self.roi
                )
            )

        return self.__class__(
            values=self.values[mask],
            timestamps=self.timestamps[mask],
            sampling_rate=self.sampling_rate,
        )

    def shift(self, shift: int):
        """Adds `shift` to the Signal's timestamps. Returns a new Signal instance."""
        return self.__class__(
            values=self.values,
            timestamps=self.timestamps + shift,
            sampling_rate=self.sampling_rate,
        )

    def to_dict(self) -> Dict[str, list]:
        """Convert to a dictionary with serialisable typing."""
        return {
            "values": self.values.tolist(),
            "timestamps": self.timestamps.tolist(),
            "sampling_rate": self.sampling_rate,
        }
