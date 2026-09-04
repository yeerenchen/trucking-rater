import pandas as pd
import os

files = ['vehicle2020.csv','vehicle2021.csv','vehicle2022.csv','vehicle2023.csv','vehicle2024.csv']

def clean():

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join("Datasets","Severity Model Data","raw",f), encoding="cp1252")

        cols = ['CASENUM','VEH_NO','REGION','MONTH','HOUR','MAX_VSEV','TOWED','VSURCOND',
                'VTRAFWAY','HAZ_INV']
        df = df[cols]

        df = df.sort_values("MAX_VSEV", ascending=False).drop_duplicates("CASENUM",keep='first').sort_index()

        df.to_csv(os.path.join("Datasets","Severity Model Data",f), index=False)
        print(f + " done")

        dfs.append(df)
    big_df = pd.concat(dfs, ignore_index=True)

    big_df = big_df[big_df["MAX_VSEV"].isin([0,1,2,3,4]) &
                    big_df["VSURCOND"].isin(range(1,12)) &
                    big_df["VTRAFWAY"].isin([1,2,3,4]) &
                    big_df["TOWED"].isin([2,3,5,7])]

    big_df.to_csv(os.path.join("Datasets","Severity Model Data", "combined.csv"), index=False)



clean()