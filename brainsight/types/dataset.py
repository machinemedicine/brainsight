from typing import Union
import json

from brainsight.types.signal import Signal
from brainsight.types.utils import VIDEO_VERBOSE_MAPPING


class _Dataset:
    r"""Nested Dataset class, ought to be used as a parent class
    to avoid level-specific properties to be copied for all nested instances.

    Parameters
    ----------
    file_or_dict : Union[str, dict]
        File path to a json file of the dataset downloaded from Kelvin,
        or the already loaded dataset dictionary.
    name : str
        Name of the nested level.

    Raises
    ------
    TypeError
        Provided ``file_or_dict`` is of the wrong type.
    """

    def __init__(self, file_or_dict: Union[str, dict], name: str) -> None:
        if isinstance(file_or_dict, str):
            dataset = self.__load_json(path=file_or_dict)
        elif isinstance(file_or_dict, dict):
            dataset = file_or_dict
        else:
            raise TypeError(
                "Dataset input is expected to be a file path (str) or a (dict)"
            )

        # Iterate over the dataset dictionary and create nested instances
        # of the dataset. Construct a Signal instance if correct keys are found.
        keys = list()
        for k, v in dataset.items():
            key = self.__format_key(k)

            if not isinstance(v, dict):
                nested = v
            elif set(v.keys()).intersection({"values", "timestamps"}):
                nested = Signal(**v)
            else:
                nested = _Dataset(file_or_dict=v, name=key)

            self.__setattr__(key, nested)
            keys.append(key)

        self._keys = set(keys) if keys else None
        self._name = name

    @staticmethod
    def __format_key(key: str) -> str:
        """Formats the dataset keys into valid attribute names."""
        k = VIDEO_VERBOSE_MAPPING.get(key, key)
        return k.split(".").pop()

    @staticmethod
    def __load_json(path: str) -> dict:
        """Loads the dataset json file."""
        if not path.endswith(".json"):
            raise ValueError("The file path is expected to end with `.json`")
        with open(file=path, mode="r") as fp:
            obj = json.load(fp=fp)
        return obj

    def keys(self) -> set:
        return self._keys

    def values(self) -> list:
        return [self[k] for k in self._keys]

    def items(self) -> list:
        return [(k, self[k]) for k in self._keys]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        sep = "\n- "
        keys = (
            "".join([sep + k for k in self._keys]) if self._keys else "Empty"
        )
        return "{}: {}".format(self._name, keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str):
        if isinstance(key, str):
            return self.__getattribute__(key)
        else:
            raise TypeError("Dataset can only be indexed with a (str) key")


class Dataset(_Dataset):
    r"""A class for convenient handling of the dataset obtained
    from the Kelvin platform. It unpacks and formats the json file,
    and integrates with ``brainsight``'s plotting functionality.
    All levels of the dataset are easily accessible as attributes
    of the initialised Dataset instance.


    Parameters
    ----------
    file_or_dict : Union[str, dict]
        File path to a json file of the dataset downloaded from Kelvin,
        or the already loaded dataset dictionary.

    Attributes
    ----------
    lfp_shift : int
        Additional time shift [in miliseconds] settable by the user
        to manually adjust the timestamps of the LFP signals.

    Other Parameters
    ----------------
    name : str, optional
        Name of the dataset, by default "Dataset"

    Examples
    --------
    >>> dataset = Dataset("path/to/dataset_file.json")
    >>> dataset
    Dataset:
    - LFP
    - MDS_UPDRS
    - ACTIVITY
    - ACCELEROMETER
    - VIDEO_METADATA
    - ASSESSMENT_INFO
    - POSE
    Additional LFP shift: 0[ms]
    >>> dataset.LFP
    LFP:
    - ZERO_TWO_LEFT
    - ZERO_TWO_RIGHT
    >>> dataset.LFP.ZERO_TWO_LEFT
    Signal(N: 113661, ROI: (0, 454600), SamplingRate: 250.0Hz)
    """

    def __init__(
        self, file_or_dict: Union[str, dict], name: str = "Dataset"
    ) -> None:
        super().__init__(file_or_dict=file_or_dict, name=name)
        self.lfp_shift = 0

    @property
    def lfp_shift(self) -> int:
        """Additional time shift of the LFP signals. Assign it an integer value
        [miliseconds] to shift all LFP Signals returned by the Dataset by that value.

        Examples
        --------
        >>> dataset = Dataset("path/to/dataset_file.json")
        >>> dataset
        Dataset:
        - LFP
        - MDS_UPDRS
        - ACTIVITY
        - ACCELEROMETER
        - VIDEO_METADATA
        - ASSESSMENT_INFO
        - POSE
        Additional LFP shift: 0[ms]
        >>> dataset.LFP.ZERO_TWO_LEFT
        Signal(N: 113661, ROI: (0, 454600), SamplingRate: 250.0Hz)
        >>> dataset.lfp_shift = 800
        >>> dataset
        Dataset:
        - LFP
        - MDS_UPDRS
        - ACTIVITY
        - ACCELEROMETER
        - VIDEO_METADATA
        - ASSESSMENT_INFO
        - POSE
        Additional LFP shift: 800[ms]
        >>> dataset.LFP.ZERO_TWO_LEFT
        Signal(N: 113661, ROI: (800, 455400), SamplingRate: 250.0Hz)
        """
        return self._lfp_shift

    @lfp_shift.setter
    def lfp_shift(self, shift: int) -> None:
        if "LFP" not in self.keys():
            raise KeyError("The Dataset does not contain LFP signals.")
        elif not isinstance(shift, int):
            raise TypeError(
                "Provided `shift` needs to be an integer [miliseconds]."
            )
        else:
            self._lfp_shift = shift

    def __getattribute__(self, name: str):
        if name == "LFP" and self.lfp_shift:
            shifted = dict()
            for channel, signal in self.__dict__["LFP"].items():
                shifted[channel] = signal.shift(self.lfp_shift)
            return _Dataset(shifted, name=self._name)
        else:
            return super().__getattribute__(name)

    def __str__(self) -> str:
        return super().__str__() + "\nAdditional LFP shift: {}[ms]".format(
            self.lfp_shift
        )
