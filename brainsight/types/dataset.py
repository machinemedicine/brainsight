from typing import Union
import json

from brainsight.types.signal import Signal
from brainsight.utils.mappings import VIDEO_VERBOSE_MAPPING


class Dataset:
    def __init__(self, file_or_dict: Union[str, dict]) -> None:
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
            if not isinstance(v, dict):
                nested = v
            elif set(v.keys()).intersection({"values", "timestamps"}):
                nested = Signal(**v)
            else:
                nested = self.__class__(file_or_dict=v)

            key = self.__format_key(k)
            self.__setattr__(key, nested)
            keys.append(key)

        self._keys = set(keys) if keys else None

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
        return "Dataset (\n  keys: {}\n)".format(
            "".join([sep + k for k in self._keys])
        )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str):
        if isinstance(key, str):
            return self.__getattribute__(key)
        else:
            raise TypeError("Dataset can only be indexed with a (str) key")
