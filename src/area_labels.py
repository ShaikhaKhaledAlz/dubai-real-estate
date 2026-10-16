"""
Area names: official DLD name -> the name people actually use.

The Dubai Land Department records every sale under an official land-registry
area name (for example "Marsa Dubai"). Most people know these places by their
community name instead (for example "Dubai Marina").

How this list was made (so you can trust it):
    For every area below, I counted the "master_project_en" column of all
    residential sales since 2022 in the official DLD file and kept the
    communities that make up most of the sales in that area.
    Example: 100% of sales in "Al Barsha South Fourth" are in the master
    project "Jumeirah Village Circle".

Areas that are not in this list simply keep their DLD name.
"""

COMMON_NAMES = {
    "Al Barsha South Fourth": "Jumeirah Village Circle (JVC)",
    "Al Barsha South Fifth": "Jumeirah Village Triangle (JVT)",
    "Al Barshaa South Third": "Arjan",
    "Al Barshaa South Second": "Dubai Science Park",
    "Marsa Dubai": "Dubai Marina, JBR & Dubai Harbour",
    "Burj Khalifa": "Downtown Dubai",
    "Al Thanyah Fifth": "Jumeirah Lake Towers (JLT)",
    "Al Thanyah Third": "The Greens",
    "Hadaeq Sheikh Mohammed Bin Rashid": "Dubai Hills Estate",
    "Al Merkadh": "Sobha Hartland & Meydan One",
    "Al Khairan First": "Dubai Creek Harbour",
    "Al Warsan First": "International City",
    "Warsan Fourth": "International City Phase 3",
    "Nadd Hessa": "Dubai Silicon Oasis",
    "Madinat Al Mataar": "Dubai South",
    "Jabal Ali First": "Al Furjan & Discovery Gardens",
    "Wadi Al Safa 2": "Liwan & Falcon City",
    "Wadi Al Safa 3": "Majan",
    "Wadi Al Safa 5": "Dubailand Residence Complex & Arabian Ranches 3",
    "Al Hebiah First": "Motor City",
    "Al Hebiah Third": "DAMAC Hills",
    "Al Hebiah Fourth": "Dubai Sports City",
    "Al Hebiah Fifth": "Remraam & DAMAC Lagoons",
    "Me'aisem First": "Dubai Production City (IMPZ)",
    "Madinat Dubai Almelaheyah": "Dubai Maritime City & Mina Rashid",
    "Al Yelayiss 2": "Town Square",
}


def short(area):
    """Return the common name if we know it, otherwise the DLD name.
    Example: short("Marsa Dubai") -> "Dubai Marina, JBR & Dubai Harbour"
    """
    return COMMON_NAMES.get(area, area)


def label(area):
    """Return 'Common name (DLD name)' so both names are visible.
    Example: label("Nadd Hessa") -> "Dubai Silicon Oasis (Nadd Hessa)"
    """
    if area in COMMON_NAMES:
        return f"{COMMON_NAMES[area]} ({area})"
    return area
