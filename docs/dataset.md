# Multimodal Dataset Creation

The true potential of BrainSight is realized when analyzing data within multimodal datasets created exclusively by the [KELVIN](https://machinemedicine.com/kelvin/) platform.

Users who capture MDS-UPDRS Part III assessments using the Kelvin Clinic application can attach to them the JSON session report exported from Medtronic's Percept PC/RC neurostimulator and benefit from the automatic processing and synchronization of LFP signals with rich kinematic video data.

Dataset creation begins when the Percept report is uploaded by the user. Once processed, it is available for download in JSON format. This file can then be easily read and explored using the [Dataset](../reference/brainsight/types/dataset/) class.

## Modalities

Here is a list of all modalities contained within the dataset:

| Modality        | Description                                                                  |
| :-------------- | :--------------------------------------------------------------------------- |
| LFP             | Local field potential signals extracted from the Percept JSON session report |
| POSE            | X- and Y-coordinate signals for 75 automatically detected body key-points    |
| ACTIVITY        | Regions of assessment-specific activities recognized within the videos       |
| MDS_UPDRS       | List of user-provided MDS-UPDRS Part III ratings                             |
| ACCELEROMETER   | Acceleration signal recorded by the assessment-capture device                |
| ASSESSMENT_INFO | Reference, date, and user email attached to the KELVIN assessment            |
| VIDEO_METADATA  | Metadata extracted from the assessment items' video files                    |

When successfully processed, all modalities ought to be synchronised in time, enabling the simultaneous analysis of the brain and motor activity! Find out how to perform simple analysis by viewing our [Tutorials](.) pages.