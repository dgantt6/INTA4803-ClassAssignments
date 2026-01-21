# Import packages and libraries
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import datetime as dt
from time import sleep
import pandas as pd
import os
from pathlib import Path
import math
import json

# Prompt for space-track.org credentials
print("Log in to using your Space-Track.org account credentials.\n")
st_email = input("Email: ")
st_pass = input("Password: ")

# Logging in using credentials
st = SpaceTrackClient(identity=st_email, password=st_pass)


def country_to_flagEmoji(country_code: str | None) -> str:
    # Converting 2 letter ISO country code to emoji flag (ISO-2 standard)
    if not country_code:
        return ""
    cc = country_code.strip().upper()
    if len(cc) != 2 or not cc.isalpha():
        return "" # not known or not ISO-2 format
    return chr(ord(cc[0])+127397) + chr(ord(cc[1])+127397)

def spaceTrackApiCall(stq: SpaceTrackClient, rows: list[dict]) -> None:
    # Helper function to fill in the satellite name, object type and operating country
    # using Space-Track

    # making country codes ISO-2 standard for emoji flags
    country_map  = {
        "US": "US", "USA": "US",
        "JPN": "JP",
        "PRC": "CN",
        "CIS": "RU", "RUS": "RU",
        "UK": "GB",
    }

    # Collect NORAD IDs from TLEs.txt file
    norad_ids = sorted({r["Satellite catalog ID"] for r in rows})
    if not norad_ids:
        return

    # Single batch SATCAT query from space-track.org
    records = stq.satcat(norad_cat_id=norad_ids, format="json")

    # Normalize Space-Track response to a list[dict].
    # Depending on spacetrack package version, json format may return a JSON string. --> processed differently
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except json.JSONDecodeError as e:
            raise ValueError(f"Space-Track returned non-JSON text: {e}")

    # If version returns a list of JSON strings (one per record)
    if isinstance(records, list) and records and isinstance(records[0], str):
        records = [json.loads(r) for r in records]

    if not isinstance(records, list):
        raise TypeError(f"Unexpected SATCAT response type: {type(records)}")

    # Lookup satellite data based on the NORAD ID
    lookup = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        try:
            lookup[int(rec.get("NORAD_CAT_ID"))] = rec
        except (TypeError, ValueError):
            continue

    # Update rows in place with name, type, and flag emoji
    for r in rows:
        rec = lookup.get(r["Satellite catalog ID"])
        if not rec:
            continue

        r["Satellite name"] = rec.get("OBJECT_NAME")
        r["Object type"] = rec.get("OBJECT_TYPE")

        raw_country = rec.get("COUNTRY")
        iso2 = country_map.get(raw_country)

        if iso2:
            r["Operating country"] = chr(ord(iso2[0]) + 127397) + chr(ord(iso2[1]) + 127397)
        else:
            r["Operating country"] = raw_country or ""


def parse_tle(line1, line2, name=None):
    # This helper takes a single TLE and returns a dictionary of required orbital
    # and identification data of the satellite
    assert line1.startswith("1 ")  # Ensure that the line actually begins with 1
    assert line2.startswith("2 ")  # Ensure that the line actually begins with 2

    # Identification Fields
    norad_id = int(line1[2:7])
    designator = line1[9:17].strip()

    # Epoch and Conversion to Calendar Date
    epoch_str = line1[18:32].strip()
    yy = int(epoch_str[0:2])
    ddd = float(epoch_str[2:])
    year = 2000 + yy if yy < 57 else 1900 + yy # First man made launch in 1957 for logic
    calendar_date = (dt.datetime(year, 1, 1) + dt.timedelta(days=ddd - 1)).date()

    # Orbital elements (from line 2)
    inclination_deg = float(line2[8:16])
    eccentricity  =  float("0." + line2[26:33].strip())
    mean_motion_rev_day = float(line2[52:63])

    # Calculate semi-major axis
    mu_earth = 398600 # km^3/s^2
    nu_rad_s = mean_motion_rev_day * 2 * math.pi / 86400
    semi_major_axis = (mu_earth / (nu_rad_s ** 2)) ** (1/3) # a = (mu/n^2) ^ (1/3) in km

    # Create dictionary for later population into a table
    sat_dict = {
        "Satellite catalog ID": norad_id,
        "COSPAR ID": designator,
        "Satellite name": name, # could be given in a TLE with header line
        "Object type": None, # will use space track later
        "Operating country": None, # will use space track later
        "Epoch (calendar date)": calendar_date,
        "Semi-major axis, a (km)": semi_major_axis,
        "Eccentricity, e": eccentricity,
        "Inclination, i (deg)": inclination_deg
    }

    return sat_dict



# Open TLE file using relative path
tle_path = Path("../Assignment1/TLEs.txt")
with tle_path.open("r") as f:
    lines =  [line.strip() for line in f if line.strip()]

# Extract every two lines of the TLE to parse them for required data
sat_rows = []
for i in range(0, len(lines), 2):
    line1 = lines[i]
    line2 = lines[i+1]

    # Run the TLE dict and return the data
    sat_data = parse_tle(line1, line2)
    sat_rows.append(sat_data)

# Call the helper functions and create the table to be saved as CSV with emojis
spaceTrackApiCall(st, sat_rows)
df = pd.DataFrame(sat_rows)
df.to_csv("tle_summary.csv", index=False, encoding="utf-8-sig")