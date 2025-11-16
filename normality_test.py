
import pandas as pd 
import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt
import safe_insights_event_counts as siec 

# Defining the cutoff date
safe_insights_extract = siec.group_by_period(pd.read_csv("safe_insights_extract.csv"))
mask = (safe_insights_extract["year"] == 2025) & ((safe_insights_extract["annual_period"] >= 5) & (safe_insights_extract["annual_period"] <= 10))

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

kruskal_wallis_results = kruskal_wallis(historical_data, "PREC_ID", "annual_period", "smis_reference")

precursors_to_pool = kruskal_wallis_results[kruskal_wallis_results["reject_null"] == False].index.tolist()
precursors_not_to_pool = kruskal_wallis_results[kruskal_wallis_results["reject_null"]].index.tolist()

historical_df = historical_data[(historical_data["PREC_ID"].isin(precursors_to_pool))].copy()
recent_df = recent_data[recent_data["PREC_ID"].isin(precursors_to_pool)].copy()
grouped = recent_df.groupby("PREC_ID")
for name, group in historical_df.groupby("PREC_ID"):
    plt.figure() # Create a new figure for each plot
    
    # Generate the probability plot
    stats.probplot(group["smis_reference"], dist="norm", plot=plt)
    
    plt.show()