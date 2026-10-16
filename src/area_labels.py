"""
DLD records use official land-registry area names, which often differ from the
names people use day to day. This adds the common name for well-known areas.
Only areas where the match is well established are listed; others keep the DLD name.
"""
COMMON_NAMES = {
    "Al Barsha South Fourth": "Jumeirah Village Circle",
    "Marsa Dubai": "Dubai Marina",
    "Burj Khalifa": "Downtown Dubai",
    "Al Thanyah Fifth": "Jumeirah Lake Towers",
    "Hadaeq Sheikh Mohammed Bin Rashid": "Dubai Hills Estate",
    "Al Merkadh": "MBR City / Sobha Hartland",
    "Al Khairan First": "Dubai Creek Harbour",
    "Al Warsan First": "International City",
    "Nadd Hessa": "Dubai Silicon Oasis",
    "Madinat Al Mataar": "Dubai South",
}


def label(area: str) -> str:
    """'Marsa Dubai' -> 'Dubai Marina (Marsa Dubai)'."""
    common = COMMON_NAMES.get(area)
    return f"{common} ({area})" if common else area


def short(area: str) -> str:
    """Common name if known, otherwise the DLD name (for chart labels)."""
    return COMMON_NAMES.get(area, area)
