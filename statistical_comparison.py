'''
File name: statistical_comparison.py 
Author: George Papagiannopoulos 
Email Address: george.papagiannopoulos@rssb.co.uk
Description: This function tests whether recent deviations in counts are attributable to random chance.
Last Edited: George Papagiannopoulos 13/11/2025
'''
import pandas as pd 
import numpy as np 
from scipy import stats 
import safe_insights_event_counts as siec 

# Defining the cutoff date
safe_insights_extract = siec.group_by_period(pd.read_csv("safe_insights_extract.csv"))
mask = (safe_insights_extract["year"] == 2025) & ((safe_insights_extract["annual_period"] >= 5) & (safe_insights_extract["annual_period"] < 10))

historical_data = safe_insights_extract[~mask]
recent_data = safe_insights_extract[mask]

def kruskal_wallis(df: pd.DataFrame, prec_id_column: str, period_column: str, event_count_col: str) -> pd.DataFrame:
    '''
    This function performs the Kruskal-Wallis H-test in a vectorized manner 
    It is used to check whether the interperiod differences in event counts for each precursor are statistically significant. 
    If not, then the data from the different periods can be pooled together which will allow more robust statistical analysis. 
    '''
    df = df.copy()
    df["rank"] = df.groupby(prec_id_column)[event_count_col].rank(method="average")

    # Grouping by precursor and period to get sum and count of ranks 
    period_stats = df.groupby([prec_id_column, period_column])["rank"].agg(["sum", "count"])
    #Calculate the inner term R_j^2/n_j
    period_stats["term"] = period_stats["sum"]**2/period_stats["count"]

    # Sum terms up to precursor level 
    cat_stats = period_stats.groupby(prec_id_column).agg({"term": "sum", "count": "sum"})
    cat_stats.rename(columns = {"count": "N"}, inplace=True)

    # Compute Raw H statistic 
    N = cat_stats["N"]
    cat_stats["H_raw"] = (12/(N*(N+1))) * cat_stats["term"] - 3 * (N  + 1)

    # We have integer data so we need to apply tie correction 
    value_counts = df.groupby([prec_id_column, event_count_col]).size()
    ties = value_counts[value_counts > 1]

    if not ties.empty:
        tie_term = (ties**3 - ties).groupby(prec_id_column).sum()
        cat_stats = cat_stats.join(tie_term.rename("tie_sum"), how="left").fillna(0)
    else:
        cat_stats["tie_sum"] = 0
    
    # Calculate and apply correction factor D = 1 - Sum(t^3 - t)/(N^3 - N)
    denominator = (N**3 - N).replace(0,1) #safety for division by zero 
    cat_stats["D"] = 1 - cat_stats["tie_sum"]/denominator

    # Final H statistic and P value
    cat_stats["H"] = cat_stats["H_raw"]/ cat_stats["D"]

    # Degrees of freedom is number of groups - 1
    k = df.groupby(prec_id_column)[period_column].nunique()
    cat_stats["dof"] = k - 1

    cat_stats['p_value'] = stats.chi2.sf(cat_stats['H'], cat_stats['dof'])
    # If the correction factor is zero we get an H of infinity
    # This happens if all the values are tied , in which case we assume that we cannot reject the null hypothesis  
    cat_stats.loc[cat_stats["D"] <= 1e-9, "p_value"] = 1.0 
    cat_stats.loc[cat_stats["D"] <= 1e-9, "H"] = 0.0
    # Our null hypothesis is that different period have the same distribution 
    # if p_value < 0.05 then we reject it, and we cannot group the data together 
    cat_stats["reject_null"] = cat_stats["p_value"] < 0.05
    
    # Filter out invalid results (e.g. N < 2 or dof < 1)
    return cat_stats[['H', 'p_value', 'N', 'dof', "reject_null"]]    

def analyze_pooled_data(historical_df: pd.DataFrame, recent_df: pd.DataFrame, prec_to_include: list, prec_id_column: str, period_column: str, event_count_col: str) -> pd.DataFrame:
    '''
    The Kruskal-Wallis test indicates that only a subset of precursors shows seasonal variation (i.e. Periods 1 - 13 come from the same distribution)
    For the precursors that do not show seasonality we can pool together the data from all the periods 
    Because we cannot assume normality of the data we will use a non-parametric test (Mann-Whitney U test) to compare our recent data 
    to the historical baseline. A Poisson test would be a better option if we could assume normality. 
    '''
    historical_df = historical_df[(historical_df[prec_id_column].isin(prec_to_include))].copy()
    recent_df = recent_df[recent_df[prec_id_column].isin(prec_to_include)].copy()

    def perform_mwu_test(group):
        prec_id = group.name

        historical_sample = historical_df[historical_df[prec_id_column] == prec_id][event_count_col]
        recent_sample = group[event_count_col]

        statistic, p_value = stats.mannwhitneyu(
            historical_sample, 
            recent_sample, 
            alternative="two-sided"
            )
        
        return pd.Series({
            "statistic": statistic,
            "p_value": p_value, 
            "n_historical": len(historical_sample),
            "n_recent": len(recent_sample)
            }
        )

    mw_result = (
        recent_df.groupby(prec_id_column)
        .apply(perform_mwu_test, include_groups= False)
    )
    mw_result["Significant"] = mw_result["p_value"] < 0.05

    return mw_result

def analyze_stratified_data(
            historical_df: pd.DataFrame, 
            recent_df: pd.DataFrame, 
            non_poolable_ids: list, 
            prec_id_col: str, 
            period_col: str, 
            event_col: str
        ) -> pd.DataFrame:
    
    # 1. Filter Data to "Non-Poolable" IDs
    hist_subset = historical_df[historical_df[prec_id_col].isin(non_poolable_ids)].copy()
    recent_subset = recent_df[recent_df[prec_id_col].isin(non_poolable_ids)].copy()
    
    # 2. Identify Active Periods
    active_periods = recent_subset[period_col].unique()
    hist_subset = hist_subset[hist_subset[period_col].isin(active_periods)]
    
    # 3. Prepare Historical Baselines (CORRECTED)
    # STEP A: Sum everything to the [Year, Period, ID] level first. 
    # This collapses any regional/fragmented lines into a single 'Total' for that moment in time.
    hist_period_totals = (
        hist_subset
        .groupby([prec_id_col, period_col, "year"])[event_col]
        .sum()
        .reset_index()
    )
    
    # STEP B: Now calculate Mean/Std across the Years
    stats_df = (
        hist_period_totals
        .groupby([prec_id_col, period_col])[event_col]
        .agg(lambda_val="mean", sigma_val="std")
        .reset_index()
    )
    
    # 4. Prepare Recent Data
    # Sum to [ID, Period] level
    recent_grouped = (
        recent_subset
        .groupby([prec_id_col, period_col], as_index=False)[event_col]
        .sum()
    )
    
    # 5. Merge on [ID, Period]
    merged = pd.merge(recent_grouped, stats_df, on=[prec_id_col, period_col], how="left")
    
    k = merged[event_col]
    lam = merged["lambda_val"]
    sig = merged["sigma_val"]

    merged.dropna(subset=["lambda_val"], inplace=True)
    
    # USE Z-TEST (Respects Overdispersion)
    # Since we have a calculated sigma from history, use it!
    # Avoid division by zero if history had 0 variance
    z_scores = (k - lam) / sig.replace(0, np.nan)
    
    # Two-tailed P-value from Normal Distribution
    p_values = stats.norm.sf(np.abs(z_scores)) * 2
    
    # Fallback: If sigma is 0 or NaN (single data point in history), 
    # you might default to Poisson, but for now Z-score is safer for overdispersed data.
    
    # 7. Finalize
    merged["Method"] = "Stratified Z-Score"
    merged["P_Value"] = p_values
    merged["Significant"] = merged["P_Value"] < 0.05

    return merged

kruskal_wallis_results = kruskal_wallis(historical_data, "PREC_ID", "annual_period", "smis_reference")

precursors_to_pool = kruskal_wallis_results[kruskal_wallis_results["reject_null"] == False].index.tolist()
precursors_not_to_pool = kruskal_wallis_results[kruskal_wallis_results["reject_null"]].index.tolist()
pooled_analysis_results = analyze_pooled_data(historical_data, recent_data, precursors_to_pool,"PREC_ID", "annual_period", "smis_reference")

pooled_analysis_results.to_csv("pooled_analysis_results.csv")
'''
non_poolable_analysis_results = analyze_stratified_data(historical_data, recent_data, precursors_not_to_pool, "PREC_ID", "annual_period", "smis_reference")
pooled_analysis_results.to_csv("pooled_analysis_results.csv")
non_poolable_analysis_results.to_csv("non_poolable_analysis_results.csv")
print(pooled_analysis_results[pooled_analysis_results["Significant"]]["PREC_ID"].unique())
print(non_poolable_analysis_results[non_poolable_analysis_results["Significant"]]["PREC_ID"].unique())
'''