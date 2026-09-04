import pandas as pd
import os

def clean_exposure_data():
    cols = ["dot_number", "nbr_power_unit", "mcs150_mileage", "phy_state"]
    folder = os.path.join("Datasets", "Exposure Bases")
    files = ["exp_apr2025.csv", "exp_may2025.csv", "exp_jun2025.csv",
            "exp_jul2025.csv", "exp_aug2025.csv", "exp_sep2025.csv",
            "exp_oct2025.csv", "exp_nov2025.csv", "exp_dec2025.csv",
            "exp_jan2026.csv", "exp_feb2026.csv", "exp_mar2026.csv",
            "exp_apr2026.csv", "exp_may2026.csv", "exp_jun2026.csv",
            "exp_jul2026.csv", "exp_aug2026.csv"]

    for f in files:
        df = pd.read_csv(os.path.join(folder, f))
        df = df[df["phy_country"] == "US"]
        df = df[cols]
        df.to_csv(os.path.join("Datasets", "Cleaned Exposures", f), index=False)
        print(f + " done")


def clean_crash_data():
    crash_df = pd.read_csv(os.path.join("Datasets", "crash_data_202609.csv"))
    exp_df = pd.read_csv(os.path.join("Datasets", "cleaned_exp_202609.csv"))

    crash_cols = ["Report_number", "DOT_Number", "Report_Date",
                  "Fatalities", "Injuries", "Tow_Away", "Hazmat_Released",
                  "Not_Preventable", "Vehicle_ID_Number"]
    exp_cols = ["DOT_NUMBER", "MCS150_MILEAGE", "CARRIER_OPERATION", "NBR_POWER_UNIT", "PHY_STATE"]

    crash_df = crash_df[crash_cols]
    exp_df = exp_df[exp_cols]
    #print(crash_df)

    crash_agg = (crash_df.groupby("DOT_Number").agg(
        crashes=("Report_number", "nunique"),
        fatalities=("Fatalities", "sum"),
        injuries=("Injuries", "sum"),
        tow_aways=("Tow_Away", "sum"),
        hazmat_releases=("Hazmat_Released", lambda x: (x=="TRUE").sum())
        )
        .reset_index()
    )

    merged_df = exp_df.merge(crash_agg, left_on="DOT_NUMBER", right_on="DOT_Number", how="left").drop(
        columns="DOT_Number"
    )

    fill_cols = ["crashes", "fatalities", "injuries", "tow_aways", "hazmat_releases"]
    merged_df[fill_cols] = merged_df[fill_cols].fillna(0)
    print(merged_df["crashes"].sum())

    merged_df.to_csv(os.path.join("Datasets", "merged_data.csv"), index=False)
    print("Done")


# === run functions ===
#clean_exposure_data()
clean_crash_data()
