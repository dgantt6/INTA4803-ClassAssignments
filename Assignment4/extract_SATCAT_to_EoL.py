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
import matplotlib.pyplot as plt

# Prompt for space-track.org credentials
print("Log in to using your Space-Track.org account credentials.\n")
st_email = input("Email: ")
st_pass = input("Password: ")

# Logging in using credentials
st = SpaceTrackClient(identity=st_email, password=st_pass)

# turns SATCAT data to type float for consistency
def safe_float(value):
    # SATCAT to float helper
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_decay_status(row):
    """ Classify each Starlink satellite using the logic:
    - Deorbited: decay date exists on SATCAT DECAY column
    - Starting decay: perigee <= 530 km while apogee still within normal Starlink shell
    - Decaying: perigee < 300 km and apogee < 300
    - Overdue: no decay date and both apogee/perigee still within normal orbital shell
    - Unclear: any issues with the data being present for apogee or perigee
    """
    apogee = row["Apogee (km)"]
    perigee = row["Perigee (km)"]
    decay_date = row["Decay date"]

    if pd.notna(decay_date):
        return "Deorbited"

    if apogee is None or perigee is None:
        return "Unclear"

    normal_apogee_low = 540
    normal_apogee_high = 650
    normal_perigee_low = 540

    if perigee <= normal_perigee_low - 10 and (normal_apogee_low - 10 <= apogee <= normal_apogee_high):
        return "Overdue - Attempting Decay"

    if perigee < 300 and apogee < 300:
        return "Decaying"

    if perigee >= normal_perigee_low-10 and apogee >= normal_apogee_low-10:
        return "Overdue"

    if 300 < perigee < normal_perigee_low and 300 < apogee < normal_apogee_low:
        return "Starting decay"

    elif (perigee <= 300 and apogee >= 300) or (perigee >= 300 and apogee <= 300):
        return "Decaying"

    return "Unclear"


# turn SATCAT launch date into age of satellite, and classify disposal status
def fetch_starlink_satcat(space_track_client, age_years=5):
    today = dt.date.today()
    cutoff_date = today - dt.timedelta(days=365 * age_years)

    records = space_track_client.satcat(
        object_name=op.like("STARLINK%"), # pull STARLINK satellite based on name
        format="json"
    )

    if isinstance(records, str):
        records = json.loads(records)
    elif records is None:
        records = []
    else:
        records = list(records)

    df = pd.DataFrame.from_records(records)

    if df.empty:
        return df

    required_columns = {"NORAD_CAT_ID", "OBJECT_NAME", "INTLDES", "LAUNCH", "DECAY", "APOGEE", "PERIGEE"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns: # in case there is a problem with the SATCAT data
        raise KeyError(f"SATCAT response is missing expected columns: {sorted(missing_columns)}")

    # turn data into python data frames
    df["Launch date"] = pd.to_datetime(df["LAUNCH"], errors="coerce").dt.date
    df["Decay date"] = pd.to_datetime(df["DECAY"], errors="coerce").dt.date
    df["Apogee (km)"] = df["APOGEE"].apply(safe_float)
    df["Perigee (km)"] = df["PERIGEE"].apply(safe_float)

    df = df[df["Launch date"].notna()].copy()
    df = df[df["Launch date"] <= cutoff_date].copy()

    df["Age (years)"] = ((today - df["Launch date"]).apply(lambda x: x.days)) / 365.25
    df["Disposal status"] = df.apply(classify_decay_status, axis=1)

    keep_columns = [
        "NORAD_CAT_ID",
        "OBJECT_NAME",
        "INTLDES",
        "Launch date",
        "Decay date",
        "Apogee (km)",
        "Perigee (km)",
        "Age (years)",
        "Disposal status",
    ]

    df = df[keep_columns].rename(
        columns={
            "NORAD_CAT_ID": "NORAD CAT ID",
            "OBJECT_NAME": "Satellite name",
            "INTLDES": "COSPAR ID",
        }
    )

    df = df.sort_values(by=["Disposal status", "Launch date", "NORAD CAT ID"]).reset_index(drop=True)
    return df


def main():
    try:
        starlink_df = fetch_starlink_satcat(st, age_years=5)
    except Exception as exc:
        print(f"Failed to fetch or parse Starlink SATCAT data: {exc}")
        return

    if starlink_df.empty:
        print("No matching Starlink satellites were returned from Space-Track.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nStarlink satellites that are 5 years old or older:\n")
    print(starlink_df)

    print("\nCounts by disposal status:\n")
    print(starlink_df["Disposal status"].value_counts(dropna=False))

    output_path = Path("starlink_5yr_decay_status.csv")
    starlink_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path.resolve()}")

    # Create bar chart of disposal categories
    counts = starlink_df["Disposal status"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Starlink End-of-Life Disposal Status for 5+ Year Old Satellites")
    plt.xlabel("Disposal Category")
    plt.ylabel("Number of Satellites")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("starlink_decay_status_bar.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

