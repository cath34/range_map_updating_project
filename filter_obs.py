import pandas as pd


def apply_effort_filters(df):
    """
    Applies effort filters to a species observation dataset
    The filters used in this function are the same as the one used to train the Status&Trends model from eBird.
    For more info, see : https://science.ebird.org/en/status-and-trends/faq#general2
    """

    mandatory_cols = [
        "TIME OBSERVATIONS STARTED",
        "EFFORT DISTANCE KM",
        "PROTOCOL NAME",
        "NUMBER OBSERVERS",
        "DURATION MINUTES",
        "OBSERVATION COUNT",
    ]

    df = df.copy()

    # Make sure numeric cols are numeric, and replace invalid values with NaN
    df["DURATION MINUTES"] = pd.to_numeric(df["DURATION MINUTES"], errors="coerce")
    df["EFFORT DISTANCE KM"] = pd.to_numeric(df["EFFORT DISTANCE KM"], errors="coerce")
    df["NUMBER OBSERVERS"] = pd.to_numeric(df["NUMBER OBSERVERS"], errors="coerce")
    df["OBSERVATION COUNT"] = pd.to_numeric(df["OBSERVATION COUNT"], errors="coerce")

    # Mask 1 : Observations associated with complete checklists
    all_species = pd.to_numeric(df["ALL SPECIES REPORTED"], errors="coerce").eq(1)

    # Mask 2 : Mandatory information columns
    base_notna = df[mandatory_cols].notna().all(axis=1)

    # Mask 3 : The sampling protocol is either "Stationary" or "Traveling"
    protocol_ok = df["PROTOCOL NAME"].isin(["Stationary", "Traveling"])

    # Mask 4 : Sampling trip duration is less than equal to 24 hours
    duration_ok = df["DURATION MINUTES"].le(1440)

    # Mask 5 : If sampling protocol == "Traveling", distance traveled must be less than equal to 10km
    traveling = df["PROTOCOL NAME"].eq("Traveling")
    distance_ok = (~traveling) | df["EFFORT DISTANCE KM"].le(10)

    # Combine all filters
    mask = base_notna & all_species & protocol_ok & duration_ok & distance_ok
    return df.loc[mask].copy()


df = pd.read_csv("Dataset/observational_data/full/bird_3.csv")
df_filtered = apply_effort_filters(df)
df_filtered.to_csv("Dataset/observational_data/filtered/bird_3.csv", index=False)
