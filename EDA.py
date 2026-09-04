import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv(r"Datasets\merged_data.csv")
df["MCS150_MILEAGE"] = pd.to_numeric(df["MCS150_MILEAGE"], errors="coerce")
df["NBR_POWER_UNIT"] = pd.to_numeric(df["NBR_POWER_UNIT"], errors="coerce")
df = df[(df["MCS150_MILEAGE"] > 0) &
        (df["MCS150_MILEAGE"] <= 500_000*df["NBR_POWER_UNIT"]) &
        (df["MCS150_MILEAGE"] > 1) &
        (df["NBR_POWER_UNIT"] > 0) &
        (df["crashes"] >= 0)
       ]

def freq_unit_plot():

    plot_df = df.copy()
    plot_df["crashes per 100k miles"] = plot_df["crashes"] / plot_df["MCS150_MILEAGE"] * 100000

    # crashes v mileage
    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_df["MCS150_MILEAGE"],
        plot_df["crashes"],
        alpha=0.2,
        s=10
    )
    plt.xlabel("MCS-150 Mileage")
    plt.xscale("log")
    plt.ylabel("Number of Crashes")
    plt.title("Crashes vs. MCS-150 Mileage")
    plt.grid(True, alpha=0.3)
    plt.show()

    # crashes v power units
    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_df["NBR_POWER_UNIT"],
        plot_df["crashes per 100k miles"],
        alpha=0.2,
        s=10
    )
    plt.xlabel("Number of Power Units")
    plt.ylabel("Number of Crashes")
    plt.title("Crashes vs. Number of Power Units")
    plt.grid(True, alpha=0.3)
    plt.show()     

def get_outliers():
    df["miles_per_unit"] = df["MCS150_MILEAGE"] / df["NBR_POWER_UNIT"]
    df["crashes_per_mile"] = df["crashes"] / df["MCS150_MILEAGE"]

    printcols = ["DOT_NUMBER", "MCS150_MILEAGE", "NBR_POWER_UNIT",
                 "miles_per_unit", "crashes_per_mile", "crashes"]
    
    print(df.nlargest(20, "miles_per_unit")[printcols])
    print(df.nlargest(20, "crashes_per_mile")[printcols])
    print(df.groupby("NBR_POWER_UNIT")["crashes"].agg(["mean", "median", "count"]))  
    print(df.groupby("NBR_POWER_UNIT")["miles_per_unit"].agg(["mean", "median", "count"]))
    print(df.groupby("NBR_POWER_UNIT")["crashes_per_mile"].agg(["mean", "median", "count"]))

    df["bad_mileage_large"] = df["miles_per_unit"] > 500_000
    print(df["bad_mileage_large"].sum())
    print(df["bad_mileage_large"].mean())

    df["bad_mileage_small"] = df["miles_per_unit"] == 1
    print(df["bad_mileage_small"].sum())
    print(df["bad_mileage_small"].mean())

    X = sm.add_constant(df["NBR_POWER_UNIT"])
    y = df["crashes_per_mile"]*100000

    model = sm.OLS(y, X).fit()

    print(model.summary())
    print("p-value:", model.pvalues["NBR_POWER_UNIT"])

# === run functions ===
#freq_unit_plot()
get_outliers()
