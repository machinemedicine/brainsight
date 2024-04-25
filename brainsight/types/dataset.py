from typing import Union, Optional
import json

from brainsight.types.signal import Signal
from brainsight.utils.mappings import VIDEO_VERBOSE_MAPPING


class _Dataset:
    def __init__(
        self, file_or_dict: Union[str, dict], name: str = "Dataset"
    ) -> None:
        if isinstance(file_or_dict, str):
            dataset = self.__load_json(path=file_or_dict)
        elif isinstance(file_or_dict, dict):
            dataset = file_or_dict
        else:
            raise TypeError(
                "Dataset input is expected to be a file path (str) or a (dict)."
            )

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
        k = VIDEO_VERBOSE_MAPPING.get(key, key)
        return k.split(".").pop()

    @staticmethod
    def __load_json(path: str) -> dict:
        if not path.endswith(".json"):
            raise ValueError("The file path is expected to end with `.json`")
        with open(file=path, mode="r") as fp:
            obj = json.load(fp=fp)
        return obj

    def keys(self) -> set:
        return self._keys

    def values(self):
        return [self[k] for k in self._keys]

    def items(self):
        return [(k, self[k]) for k in self._keys]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        sep = "\n  - "
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
    def __init__(self, file_or_dict: Union[str, dict]) -> None:
        super().__init__(file_or_dict)
        self._lfp_shift = 0

    @property
    def lfp_shift(self) -> int:
        """Additional shift of the LFP signals."""
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
            return _Dataset(shifted)
        else:
            return super().__getattribute__(name)

    def __str__(self) -> str:
        return super().__str__() + "\nAdditional LFP shift: {}[ms]".format(
            self.lfp_shift
        )
