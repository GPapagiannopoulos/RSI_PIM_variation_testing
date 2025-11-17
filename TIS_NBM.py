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

# Configuration 
intervention_date = pd.Timestamp("2025-08-13")

# Load and prepare data 
df = group_by_period(pd.read_csv("safe_insights_extract.csv"))
df["fiscal_year"] = df["Period"].str.extract(r"^(\d{4})/", expand=False).astype(int)
df["period_num"]  = df["Period"].str.extract(r"P(\d+)", expand=False).astype(int)
df["End Date"] = pd.to_datetime(df["End Date"])

def determine_period_days(data:pd.DataFrame) -> pd.DataFrame:
    period_end_dates = pd.read_csv("period_end_dates.csv")
    period_end_dates["end_date"] = pd.to_datetime(period_end_dates["mmm yyyy"])
    period_end_dates["fiscal_year"] = period_end_dates["Period"].str.extract(r"^(\d{4})/", expand=False).astype(int)
    period_end_dates["period_num"]  = period_end_dates["Period"].str.extract(r"P(\d+)", expand=False).astype(int)
    
    period_end_dates = period_end_dates.sort_values(["fiscal_year", "period_num"])

    period_end_dates["start_date"] = (
        period_end_dates["end_date"].shift(1) + pd.Timedelta(days=1)
    )

    # Merge to get the start dates for each period
    data = data.merge(
        period_end_dates[["Period", "start_date", "end_date"]],
        on="Period",
        how="left"
    )    

    # Compute the offset so that the model knows to account for it
    data["period_days"] = (
        data["end_date"] - data["start_date"]
    ).dt.days + 1
    data["log_offset"] = np.log(data["period_days"])

    return data[["year", "period_num", "start_date", "end_date", "period_days", "log_offset", "PREC_ID", "smis_reference"]] 

df = determine_period_days(df)


results = []
for prec_id, group_data in df.groupby("PREC_ID"):

    group_data = group_data.sort_values("end_date").copy()
    group_data["t"] = np.arange(len(group_data))

    transition_mask = (
        (group_data["start_date"] < intervention_date) &
        (group_data["end_date"]   >= intervention_date)
    )
    group_data["intervention"] = (group_data["start_date"] >= intervention_date).astype(int)

    model_data = group_data[~transition_mask].copy()

    if model_data["intervention"].nunique() < 2:
        print(f"PREC_ID {prec_id} does not have both pre and post intervention data - skipping")
        continue
    model_data = model_data.sort_values("end_date").copy()
    model_data["t"] = np.arange(len(model_data))

    first_post_t = model_data.loc[model_data["intervention"] == 1, "t"].min()
    model_data["post_trend"] = np.where(model_data["t"] >= first_post_t,
                                        model_data["t"] - first_post_t,
                                        0)

    # Seasonal term based on true fiscal period number
    if model_data["period_num"].nunique() < 3:
        formula = "smis_reference ~ t + intervention + post_trend"
    else:
        formula = "smis_reference ~ t + C(period_num) + intervention + post_trend"


    try:
        nb_family = sm.families.NegativeBinomial(alpha=0.1)

        model = smf.glm(
            formula=formula,
            data=model_data,
            family=nb_family,
            offset=model_data["log_offset"]
        ).fit()

        model2 = smf.glm(
            formula=formula,
            data=model_data,
            family=sm.families.Poisson(),
            offset=model_data["log_offset"]
        ).fit(cov_type="HC3")

        # Dispersion diagnostic: Pearson chi2 / df_resid
        if model.df_resid > 0:
            pearson_ratio = model.pearson_chi2 / model.df_resid
        else:
            pearson_ratio = np.nan  # saturated or nearly so
        if model2.df_resid >0:
            pearson_ratio_poisson = model2.pearson_chi2 / model2.df_resid
        else:
            pearson_ratio_poisson = np.nan  # saturated or nearly so
        results.append({
            "PREC_ID": prec_id,
            "Intervention_Date": intervention_date,
            "P_Value_NB": model.pvalues.get("intervention", np.nan),
            "P_Value_Poisson": model2.pvalues.get("intervention", np.nan),
            "Pearson_ratio_NB": pearson_ratio,
            "Pearson_ratio_Poisson": pearson_ratio_poisson,
            "Coef_NB": model.params.get("intervention", np.nan),
            "Significant": bool(model.pvalues.get("intervention", 1.0) < 0.05),
            "Significant_Poisson": bool(model2.pvalues.get("intervention", 1.0) < 0.05)
        })

    except Exception as e:
        print(f"Model for PREC_ID {prec_id} did not converge: {e}")


results = pd.DataFrame(results)

overdispersed_mask = (results["Pearson_ratio_Poisson"] > 1.5)

NB_cols = ["PREC_ID", "P_Value_NB", "Pearson_ratio_NB", "Significant"]
NB_precursors = results.loc[overdispersed_mask, NB_cols].rename(
    columns={
        "P_Value_NB": "P_value",
        "Pearson_ratio_NB": "Pearson_ratio"
    }
)

Poisson_cols = ["PREC_ID", "P_Value_Poisson", "Pearson_ratio_Poisson", "Significant_Poisson"]
Poisson_precursors = results.loc[~overdispersed_mask, Poisson_cols].rename(
    columns={
        "P_Value_Poisson": "P_value",
        "Pearson_ratio_Poisson": "Pearson_ratio",
        "Significant_Poisson": "Significant"
    }
)


results = pd.concat([NB_precursors, Poisson_precursors], ignore_index=True).sort_values("PREC_ID")
results.to_csv("results.csv", index=False)

print(results[results["Significant"]]["PREC_ID"])

