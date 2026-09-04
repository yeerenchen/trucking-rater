import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from xgboost import XGBRegressor

df_freq = pd.read_csv(r"Datasets\Frequency Model Data\merged_data.csv")
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

    # All cargo columns
    cargo_cols = [c for c in train.columns if c.startswith("CRGO_")]

    # Convert X / blank -> 1 / 0
    train[cargo_cols] = train[cargo_cols].eq("X").astype(int)
    valid[cargo_cols] = valid[cargo_cols].eq("X").astype(int)

    features = [
    "POWER_UNITS",
    "MCS150_MILEAGE",
    *cargo_cols
    ]

    X_train = train[features]
    X_valid = valid[features]

    y_train = train["crashes"]
    y_valid = valid["crashes"]

    # XGBoost Poisson model
    xgb = XGBRegressor(
    objective="count:poisson",
    random_state=20
    )

    param_grid = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 2, 5, 10]
    }

    mae_scorer = make_scorer(
        mean_absolute_error,
        greater_is_better=False
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=4,
        scoring=mae_scorer,
        cv=5,
        random_state=20,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)

    print("Best CV MAE:")
    print(-search.best_score_)

    xgb_best = search.best_estimator_

    valid["predicted_crashes_xgb"] = xgb_best.predict(X_valid)

    mae_xgb = mean_absolute_error(
        valid["crashes"],
        valid["predicted_crashes_xgb"]
    )

    rmse_xgb = np.sqrt(mean_squared_error(
        valid["crashes"],
        valid["predicted_crashes_xgb"]
    ))

    print(f"Validation MAE:  {mae_xgb:.4f}")
    print(f"Validation RMSE: {rmse_xgb:.4f}")
    

# === run ===
fit_freq_models()