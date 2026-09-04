import pandas as pd

# DATA MAPPINGS
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

"""
VSEV:
0 - No Apparent Injury (O)
1 - Possible Injury (C)
2 - Suspected Minor Injury (B)
3 - Suspected Serious Injury (A)
4 - Fatal Injury (K)
"""

df = pd.read_csv(r"Datasets\Severity Model Data\combined.csv")

# === BI SEV ===
# KABCO FACTORS
K = 11_295_400/11_900
A = 655_000/11_900
B = 198_500/11_900
C = 125_600/11_900
O = 1
PD_BASE = 9500  # trended/approximated using TAIPA Filing
BI_PROXY = 45600  # ''

sev_probs = (
    df.groupby("REGION")["MAX_VSEV"]
      .value_counts(normalize=True)
      .unstack(fill_value=0)
      .sort_index(axis=1)
)
sev_grouped = df["MAX_VSEV"].value_counts(normalize=True)

print(sev_probs)
print(sev_grouped.index)

BASE_BI_SEV = BI_PROXY/(K*sev_grouped[4]+A*sev_grouped[3]+B*sev_grouped[2]+C*sev_grouped[1]+O*sev_grouped[0])
print(BASE_BI_SEV)

# === PD SEV ===
# BASE: 9500 (from texas)
# <10000 - 0.7
# 80 - 1.5
# between is linear