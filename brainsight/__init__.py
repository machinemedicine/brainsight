r"""**BrainSight** accommodates easy and intuitive analysis of local field potential (LFP)
data captured by the Percept PC device and saved as part of a multimodal data exported through
the KELVIN platform.

[`brainsight.types`](types/__init__.md) contain classes designed to simplify the handling of
multimodal datasets and timestamp-synchronised signals.

[`brainsight.modules`](modules/__init__.md) allow for processing and plotting of the data using
the custom typing, ensuring that the analysis is straightforward and robust.

The most common `modules` and `types` can be directly accessed from `brainsight`.

Examples
--------
>>> import brainsight as brain
>>> dataset = brain.Dataset("path/to/dataset_file.json")
>>> dataset
Dataset: 
- ACTIVITY
- MDS_UPDRS
- ACCELEROMETER
- LFP
- POSE
- ASSESSMENT_INFO
- VIDEO_METADATA
Additional LFP shift: 0[ms]
"""

from brainsight.types import Signal, Dataset
from brainsight.modules import LFP, Periodogram, Spectrogram
