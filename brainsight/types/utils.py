VIDEO_VERBOSE_MAPPING = {
    "SPEECH": "SPEECH",
    "FE": "FACIAL_EXPRESSION",
    "RIGIDITY": "RIGIDITY",
    "FT": "FINGER_TAPPING",
    "HM": "HAND_MOVEMENT",
    "PS": "PRONATION_SUPINATION",
    "TT": "TOE_TAPPING",
    "LA": "LEG_AGILITY",
    "RFC": "ARISING_FROM_CHAIR",
    "GAIT": "GAIT",
    "FOG": "FREEZING_OF_GAIT",
    "POSSTAB": "POSTURAL_STABILITY",
    "POSTURE": "POSTURE",
    "PTH": "POSTURAL_TREMOR_OF_HANDS",
    "KTH": "KINETIC_TREMOR_OF_HANDS",
    "RT": "REST_TREMOR",
}


class _Alias:
    """Used for creating attribute aliases"""

    def __init__(self, source_name):
        self.source_name = source_name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.source_name)

    def __set__(self, obj, value):
        setattr(obj, self.source_name, value)
