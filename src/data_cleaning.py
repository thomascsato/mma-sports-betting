"""
data_cleaning.py

This script reads in data from mma_data_scraper.py and convert_txt_to_csv.py in order to
1) Remove columns that are incomplete for some reason and
2) Join with fighter data in lieu of the missing columns for fighter stats.

Author: Thomas Sato
Date: 11/10/2024
"""

import pandas as pd

COLUMNS_TO_DROP = ["r_SLpM_total", "r_SApM_total", "r_sig_str_acc_total", "r_td_acc_total",	"r_str_def_total",
                   "r_td_def_total", "r_sub_avg", "r_td_avg", "b_SLpM_total", "b_SApM_total", "b_sig_str_acc_total",
                   "b_td_acc_total", "b_str_def_total",	"b_td_def_total", "b_sub_avg", "b_td_avg", "SLpM_total_diff",
                   "SApM_total_diff", "sig_str_acc_total_diff",	"td_acc_total_diff", "str_def_total_diff",
                   "td_def_total_diff",	"sub_avg_diff",	"td_avg_diff", "wins_total_diff", "losses_total_diff",
                   "age_diff", "height_diff", "weight_diff", "reach_diff"
]

fighters = pd.read_csv("MMA_Fighter_Compare\\data\\fighter_stats.csv")
completed_fights = pd.read_csv("MMA_Fighter_Compare\\data\\completed_events_large.csv")

completed_fights_new = completed_fights.drop(columns=COLUMNS_TO_DROP)

red_fighters = fighters.add_prefix("r_").add_suffix("_agg")
blue_fighters = fighters.add_prefix("b_").add_suffix("_agg")

result = pd.merge(completed_fights_new, red_fighters, left_on="r_fighter", right_on="r_name_agg", how="left")
result = pd.merge(result, blue_fighters, left_on="b_fighter", right_on="b_name_agg", how="left")

# Adding difference columns (red minus blue)
result["wins_diff"] = result["r_wins_agg"] - result["b_wins_agg"]
result["losses_diff"] = result["r_losses_agg"] - result["b_losses_agg"]
result["height_diff"] = result["r_height_agg"] - result["b_height_agg"]
result["weight_diff"] = result["r_weight_agg"] - result["b_weight_agg"]
result["reach_diff"] = result["r_reach_agg"] - result["b_reach_agg"]
result["age_diff"] = result["r_age_agg"] - result["b_age_agg"]
result["SLpM_diff"] = result["r_SLpM_agg"] - result["b_SLpM_agg"]
result["SApM_diff"] = result["r_SApM_agg"] - result["b_SApM_agg"]
result["sig_str_acc_agg_diff"] = result["r_sig_str_acc_agg"] - result["b_sig_str_acc_agg"]
result["str_def_agg_diff"] = result["r_str_def_agg"] - result["b_str_def_agg"]
result["td_avg_agg_diff"] = result["r_td_avg_agg"] - result["b_td_avg_agg"]
result["td_acc_agg_diff"] = result["r_td_acc_agg"] - result["b_td_acc_agg"]
result["td_def_agg_diff"] = result["r_td_def_agg"] - result["b_td_def_agg"]
result["sub_avg_agg_diff"] = result["r_sub_avg_agg"] - result["b_sub_avg_agg"]

result['winner'] = result['winner'].apply(lambda x: 1 if x == 'Red' else 0)

result.to_csv("MMA_Fighter_Compare\\data\\all_fights_final.csv", index=False)