r"""BrainSight package accommodates easy and intuitive analysis of local field potential (LFP)
data captured by the Percept PC device and saved as part of a multimodal data exported through
the KELVIN platform.

Examples
--------
>>> import brainsight as brain

"""

from brainsight.types import Signal, Dataset
from brainsight.modules import LFP, Periodogram, Spectrogram
