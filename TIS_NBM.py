'''
File name: TIS_NBM.py 
Author: George Papagiannopoulos 
Email Address: george.papagiannopoulos@rssb.co.uk
Description: A time series interrupted negative binomial model to check whether changes in the trend/ count 
of Safe Insights entries are statistically significant. 
Last Edited: George Papagiannopoulos 16/11/2025
'''

import pandas as pd 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import numpy as np
from safe_insights_event_counts import group_by_period 
from tqdm import tqdm 

df = group_by_period(pd.read_csv("safe_insights_extract.csv"))

results = []

for prec_id, df_prec in tqdm(df.groupby("PREC_ID")):
    
    df_prec = df_prec.sort_values("End Date").copy()

    # Create time index per category
    df_prec["time"] = range(len(df_prec))
    
    # Find intervention time for this category

    intervention_time = df_prec.index[df_prec["End Date"] >= "2025-07-01"].min()
    if pd.isna(intervention_time):
        # No post-intervention data => cannot fit ITS
        continue
    intervention_time = df_prec.loc[intervention_time, "time"]
    
    df_prec["intervention"] = (df_prec["time"] >= intervention_time).astype(int)
    df_prec["post_trend"] = (df_prec["time"] - intervention_time).clip(lower=0)

    # Check for numeric model
    if df_prec[["time","intervention","post_trend"]].nunique().min() <= 1:
        # Too few unique values to fit a model
        continue

    # Fit Negative Binomial ITS
    model = smf.glm(
        formula="smis_reference ~ time + intervention + post_trend",
        data=df_prec,
        family=sm.families.NegativeBinomial()
    ).fit()

    results.append({
        "PREC_ID": prec_id,
        "beta_intervention": model.params["intervention"],
        "beta_posttrend": model.params["post_trend"],
        "p_intervention": model.pvalues["intervention"],
        "p_posttrend": model.pvalues["post_trend"],
        "RR_intervention": np.exp(model.params["intervention"]),
        "RR_posttrend": np.exp(model.params["post_trend"])
    })

results_df = pd.DataFrame(results)
results_df.to_csv("tis_nbm_results.csv", index=False)
print(results_df)
