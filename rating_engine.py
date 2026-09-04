from dataclasses import dataclass
import numpy as np
import requests
import re
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===== MODEL AND PROBABILITY SETUPS =====
# state mapping
state_to_region = {
    # 1 = Northeast
    "PA": 1, "NJ": 1, "NY": 1, "NH": 1, "VT": 1,
    "RI": 1, "MA": 1, "ME": 1, "CT": 1,

    # 2 = Midwest
    "OH": 2, "IN": 2, "IL": 2, "MI": 2, "WI": 2,
    "MN": 2, "ND": 2, "SD": 2, "NE": 2, "IA": 2,
    "MO": 2, "KS": 2,

    # 3 = South
    "MD": 3, "DE": 3, "DC": 3, "WV": 3, "VA": 3,
    "KY": 3, "TN": 3, "NC": 3, "SC": 3, "GA": 3,
    "FL": 3, "AL": 3, "MS": 3, "LA": 3, "AR": 3,
    "OK": 3, "TX": 3,

    # 4 = West
    "MT": 4, "ID": 4, "WA": 4, "OR": 4, "CA": 4,
    "NV": 4, "NM": 4, "AZ": 4, "UT": 4, "CO": 4,
    "WY": 4, "AK": 4, "HI": 4,
}

# BI sev
df_sev = pd.read_csv(r"Datasets\Severity Model Data\combined.csv")
bi_sev_probs = (
    df_sev.groupby("REGION")["MAX_VSEV"]
      .value_counts(normalize=True)
      .unstack(fill_value=0)
      .sort_index(axis=1)
)
bi_sev_grouped = df_sev["MAX_VSEV"].value_counts(normalize=True)

# freq model
df_freq = pd.read_csv(r"Datasets\Frequency Model Data\merged_data.csv")
df_freq = df_freq[(df_freq["MCS150_MILEAGE"] <= 500_000*df_freq["POWER_UNITS"]) &
                (df_freq["MCS150_MILEAGE"] > 1) &
                (df_freq["POWER_UNITS"] > 0) &
                (df_freq["crashes"] >= 0) &
                (df_freq["CARRIER_OPERATION"].notna())
                ]

# ========================================

def decode_vin(vin):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}"
    
    r = requests.get(url, params={"format": "json"})
    r.raise_for_status()
    
    result = r.json()["Results"][0]
    
    return {
        "VIN": vin,
        "Make": result.get("Make"),
        "Model": result.get("Model"),
        "ModelYear": result.get("ModelYear"),
        "GVWR": result.get("GVWR"),
        "GCWR": result.get("GCWR"),
        "CurbWeightLB": result.get("CurbWeightLB"),
        "VehicleType": result.get("VehicleType"),
        "BodyClass": result.get("BodyClass")
    }

def parse_gvwr(x):
    if not x:
        return None
    
    nums = re.findall(r"[\d,]+", x)
    
    if len(nums) >= 2:
        lo = int(nums[0].replace(",", ""))
        hi = int(nums[1].replace(",", ""))
        return (lo + hi) / 2
    
    return None

def fit_freq_glm():

    train, temp = train_test_split(df_freq, test_size=0.2, random_state=20)
    valid, test = train_test_split(temp, test_size=0.5, random_state=20)

    pois = smf.glm(
        "crashes ~ C(POWER_UNITS) * CARRIER_OPERATION",
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["MCS150_MILEAGE"])
    ).fit()
    
    valid["predicted_crashes_pois"] = pois.predict(valid, offset=np.log(valid["MCS150_MILEAGE"]))
    #print(pois.summary())
    
    mae_pois = mean_absolute_error(
        valid["crashes"],
        valid["predicted_crashes_pois"]
    )
    
    rmse_pois = np.sqrt(mean_squared_error(
        valid["crashes"],
        valid["predicted_crashes_pois"]
    ))

    #print(f"=====POISSON GLM=====")
    #print(f"Validation MAE:  {mae_pois:.4f}")
    #print(f"Validation RMSE: {rmse_pois:.4f}")

    actual = valid["crashes"].sum()
    predicted_pois = valid["predicted_crashes_pois"].sum()

    #print(f"Actual crashes:    {actual:.0f}")
    #print(f"Predicted crashes: {predicted_pois:.0f}")
    #print(f"Actual/Predicted:  {actual / predicted_pois:.3f}")

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

        #print(calibration_pois)

    # === POISSON GLM TEST ===
    #print("===== TEST SET METRICS =====")
    test['predicted_crashes_pois'] = pois.predict(test, offset=np.log(test["MCS150_MILEAGE"]))
    mae_pois = mean_absolute_error(
        test["crashes"],
        test["predicted_crashes_pois"]
    )

    rmse_pois = np.sqrt(mean_squared_error(
        test["crashes"],
        test["predicted_crashes_pois"]
    ))

    #print(f"Test MAE:  {mae_pois:.4f}")
    #print(f"Test RMSE: {rmse_pois:.4f}")

    actual = test["crashes"].sum()
    predicted_pois = test["predicted_crashes_pois"].sum()

    #print(f"Actual crashes:    {actual:.0f}")
    #print(f"Predicted crashes: {predicted_pois:.0f}")
    #print(f"Actual/Predicted:  {actual / predicted_pois:.3f}")

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

        #print(calibration_pois)

    return pois


class PricingEngine:
    def __init__(self, usdot, drivers, pu, operating_radius, hazmat, state=""):
        self.usdot = usdot
        self.drivers = drivers
        self.pu = pu  # list of VINs
        for vin in self.pu:
            if len(vin) != 17: raise ValueError("Invalid VIN")
        self.operating_radius = operating_radius
        self.hazmat = hazmat
        if hazmat not in ["A","B","C"]: raise ValueError("Invalid Carrier Operating Code")
        self.state = state
        if state not in state_to_region: raise ValueError("Invalid State Abbreviation")

        self.trucks = len(pu)
        self.weights = self.get_powerunit_weight()

        est_miles = np.minimum(250*min(self.drivers, self.trucks)*1.1*self.operating_radius, 100_000)
        self.mileage = est_miles * self.trucks

        self.quote = True
        self.premCutoff = 0

        if (self.mileage > 50000) and self.hazmat == "B": self.quote = False

    def get_powerunit_weight(self):
        weights = []
        for vin in self.pu:
            w = decode_vin(vin)["GVWR"]
            weights.append(parse_gvwr(w))
        return weights

    def get_PD_sev(self):
        PD_BASE = 9500
        lo, hi = 0.7, 1.5
        
        sum_pd = 0
        for w in self.weights:
            if w < 10000: sum_pd += PD_BASE*lo
            else:
                factor = hi*(w-10000)/70000 + lo*(1-(w-10000)/70000)
                sum_pd += PD_BASE*factor

        return sum_pd / self.trucks

    def get_BI_sev(self):
        BI_PROXY = 45600

        # factors
        K = 11_295_400/11_900
        A = 655_000/11_900
        B = 198_500/11_900
        C = 125_600/11_900
        O = 11_900/11_900

        BASE_SEV = BI_PROXY/(K*bi_sev_grouped[4]+
                             A*bi_sev_grouped[3]+
                             B*bi_sev_grouped[2]+
                             C*bi_sev_grouped[1]+
                             O*bi_sev_grouped[0])

        if self.state == "": return BI_PROXY
        else:
            region = state_to_region[self.state]
            factor = (K*bi_sev_probs.loc[region, 4]+
                      A*bi_sev_probs.loc[region, 3]+
                      B*bi_sev_probs.loc[region, 2]+
                      C*bi_sev_probs.loc[region, 1]+
                      O*bi_sev_probs.loc[region, 0])
            
            return factor*BASE_SEV

    def get_freq(self):
        model = fit_freq_glm()

        data = pd.DataFrame({
            "POWER_UNITS": [self.trucks],
            "CARRIER_OPERATION": [self.hazmat],
            "MCS150_MILEAGE": [self.mileage]
        })

        baseline_freq = model.predict(data, offset=np.log(data["MCS150_MILEAGE"])).iloc[0]
        self.premCutoff = baseline_freq * (9500+45600) * 3

        # credibility with df_freq
        FULL_CRED = 1_000_000  # max possible mileage (500,000) gets ~70% weight

        if self.usdot in df_freq["DOT_NUMBER"]:
            row = df_freq.loc[df_freq["DOT_NUMBER"] == self.usdot].iloc[0]
            Z = min(1,np.sqrt(row["MCS150_MILEAGE"] / FULL_CRED))  # prior info factor
            hist_freq = row["crashes"] / row["MCS150_MILEAGE"] * self.mileage

            if hist_freq > 2*baseline_freq: self.quote = False

            return Z*hist_freq + (1-Z)*baseline_freq
        
        else: return baseline_freq

    def get_prem(self):
        EXPENSE = 0.3  # CAS
        PROFIT = 0.1  # safety

        freq = self.get_freq()
        bi_sev = self.get_BI_sev()
        pd_sev = self.get_PD_sev()

        prem = freq*(bi_sev+pd_sev)/(1-EXPENSE-PROFIT)
  
        if prem > self.premCutoff: self.quote = False

        return prem if self.quote else "Decline"
    
# hazmat input: A - Interstate, B - Intrastate+Hazmat, C - Intrastate+NoHazmat
c1 = PricingEngine(285841,3,["1FVACWDB2GH123456"],500,"A","OK")
c2 = PricingEngine(894785,5,["4V4NC9EH0JN886331","1XKYD49X6LJ956830","3AKJHHDR0PSNY9089"], 30, "B")
c3 = PricingEngine(2833560,2,["4V4NC9EH0JN886331","1XKYD49X6LJ956830","3AKJHHDR0PSNY9089"], 50, "B")
c4 = PricingEngine(2835739,10,["1XPHD49X1AD796233","1XPHD49X1AD796233","1XPHD49X1AD796233","1XPHD49X1AD796233","1XPHD49X1AD796233"], 100, "C", "AL")
c5 = PricingEngine(285841,3,["1FVACWDB2GH123456"],500,"A","KY")
c6 = PricingEngine(285841,3,["1FVACWDB2GH123456"],500,"A","VT")
c7 = PricingEngine(285841,3,["1FVACWDB2GH123456"],300,"A")

cs = [c1,c2,c3,c4,c5,c6,c7]
for c in cs: print(c.get_prem())