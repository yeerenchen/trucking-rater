import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_freq = pd.read_csv(r"Datasets\merged_data.csv")
df_freq = df_freq[(df_freq["MCS150_MILEAGE"] <= 500_000*df_freq["POWER_UNITS"]) &
        (df_freq["MCS150_MILEAGE"] > 1) &
        (df_freq["POWER_UNITS"] > 0) &
        (df_freq["crashes"] >= 0) &
        (df_freq["CARRIER_OPERATION"].notna())
       ]

print(df_freq.isna().sum()[df_freq.isna().sum() > 0])


def fit_freq_models():

    train, temp = train_test_split(df_freq, test_size=0.2, random_state=20)
    valid, test = train_test_split(temp, test_size=0.5, random_state=20)

    '''# === Neg-Bin GLM ===
    nb = smf.glm(
        "crashes ~ C(POWER_UNITS) * CARRIER_OPERATION",
        data=train,
        family=sm.families.NegativeBinomial(),
        offset=np.log(train["MCS150_MILEAGE"])
    ).fit()

    valid["predicted_crashes_nb"] = nb.predict(valid, offset=np.log(valid["MCS150_MILEAGE"]))
    print(nb.summary())

    mae_nb = mean_absolute_error(
        valid["crashes"],
        valid["predicted_crashes_nb"]
    )

    rmse_nb = np.sqrt(mean_squared_error(
        valid["crashes"],
        valid["predicted_crashes_nb"]
    ))'''

    # === Poisson GLM ===
    pois = smf.glm(
        "crashes ~ C(POWER_UNITS) * CARRIER_OPERATION",
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["MCS150_MILEAGE"])
    ).fit()

    valid["predicted_crashes_pois"] = pois.predict(valid, offset=np.log(valid["MCS150_MILEAGE"]))
    print(pois.summary())

    mae_pois = mean_absolute_error(
        valid["crashes"],
        valid["predicted_crashes_pois"]
    )

    rmse_pois = np.sqrt(mean_squared_error(
        valid["crashes"],
        valid["predicted_crashes_pois"]
    ))

    '''print("=====NEGATIVE-BINOMIAL GLM=====")
    print(f"Validation MAE NB:  {mae_nb:.4f}")
    print(f"Validation RMSE NB: {rmse_nb:.4f}")

    actual = valid["crashes"].sum()
    predicted_nb = valid["predicted_crashes_nb"].sum()

    print(f"Actual crashes:    {actual:.0f}")
    print(f"Predicted crashes: {predicted_nb:.0f}")
    print(f"Actual/Predicted:  {actual / predicted_nb:.3f}")

    calibration_nb = valid.groupby("POWER_UNITS").agg(
        actual=("crashes", "sum"),
        predicted=("predicted_crashes_nb", "sum"),
        exposure=("MCS150_MILEAGE", "sum")
    )

    calibration_nb["A_E"] = (
        calibration_nb["actual"] / calibration_nb["predicted"]
    )

    calibration_nb["actual_rate"] = (
        calibration_nb["actual"] / calibration_nb["exposure"] * 100_000
    )

    calibration_nb["predicted_rate"] = (
        calibration_nb["predicted"] / calibration_nb["exposure"] * 100_000
    )

    print(calibration_nb)'''

    print(f"=====POISSON GLM=====")
    print(f"Validation MAE:  {mae_pois:.4f}")
    print(f"Validation RMSE: {rmse_pois:.4f}")

    actual = valid["crashes"].sum()
    predicted_pois = valid["predicted_crashes_pois"].sum()

    print(f"Actual crashes:    {actual:.0f}")
    print(f"Predicted crashes: {predicted_pois:.0f}")
    print(f"Actual/Predicted:  {actual / predicted_pois:.3f}")

    group_cols = ["POWER_UNITS", "CARRIER_OPERATION"]
    for s in group_cols:
        calibration_pois = valid.groupby(s).agg(
            actual=("crashes", "sum"),
            predicted=("predicted_crashes_pois", "sum"),
            exposure=("MCS150_MILEAGE", "sum")
        )

        calibration_pois["A_E"] = (
            calibration_pois["actual"] / calibration_pois["predicted"]
        )

        calibration_pois["actual_rate"] = (
            calibration_pois["actual"] / calibration_pois["exposure"] * 100_000
        )

        calibration_pois["predicted_rate"] = (
            calibration_pois["predicted"] / calibration_pois["exposure"] * 100_000
        )

        print(calibration_pois)

    # === POISSON GLM TEST ===
    print("===== TEST SET METRICS =====")
    test['predicted_crashes_pois'] = pois.predict(test, offset=np.log(test["MCS150_MILEAGE"]))
    mae_pois = mean_absolute_error(
        test["crashes"],
        test["predicted_crashes_pois"]
    )

    rmse_pois = np.sqrt(mean_squared_error(
        test["crashes"],
        test["predicted_crashes_pois"]
    ))

    print(f"Test MAE:  {mae_pois:.4f}")
    print(f"Test RMSE: {rmse_pois:.4f}")

    actual = test["crashes"].sum()
    predicted_pois = test["predicted_crashes_pois"].sum()

    print(f"Actual crashes:    {actual:.0f}")
    print(f"Predicted crashes: {predicted_pois:.0f}")
    print(f"Actual/Predicted:  {actual / predicted_pois:.3f}")

    for s in group_cols:
        calibration_pois = test.groupby(s).agg(
            actual=("crashes", "sum"),
            predicted=("predicted_crashes_pois", "sum"),
            exposure=("MCS150_MILEAGE", "sum")
        )

        calibration_pois["A_E"] = (
            calibration_pois["actual"] / calibration_pois["predicted"]
        )

        calibration_pois["actual_rate"] = (
            calibration_pois["actual"] / calibration_pois["exposure"] * 100_000
        )

        calibration_pois["predicted_rate"] = (
            calibration_pois["predicted"] / calibration_pois["exposure"] * 100_000
        )

        print(calibration_pois)
    

# === run ===
fit_freq_models()