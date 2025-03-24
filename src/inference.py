import joblib
import pandas as pd
"""
print("Reading in models")
post_fight_model = joblib.load('models\\post_fight_model.joblib')
win_pred_model = joblib.load('models\\win_pred_model.joblib')
print("Models read in successfully.")
"""
# Column names
pre_features_numeric = ["r_wins_agg",
    "r_losses_agg", "r_height_agg", "r_weight_agg", "r_reach_agg", "r_age_agg",
    "r_SLpM_agg", "r_sig_str_acc_agg", "r_SApM_agg", "r_str_def_agg",
    "r_td_avg_agg", "r_td_acc_agg", "r_td_def_agg", "r_sub_avg_agg", "b_wins_agg",
    "b_losses_agg", "b_height_agg", "b_weight_agg", "b_reach_agg", "b_age_agg",
    "b_SLpM_agg", "b_sig_str_acc_agg", "b_SApM_agg", "b_str_def_agg",
    "b_td_avg_agg", "b_td_acc_agg", "b_td_def_agg", "b_sub_avg_agg"]
pre_features_categorical = ["weight_class", "is_title_bout", "gender", "r_stance_agg", "b_stance_agg"]

post_features_numeric = ["finish_round", "total_rounds", "time_sec", 
        "r_kd", "r_sig_str", "r_sig_str_att", "r_sig_str_acc", "r_str",
        "r_str_att", "r_str_acc", "r_td", "r_td_att", "r_td_acc",
        "r_sub_att", "r_rev", "r_ctrl_sec", "b_kd", "b_sig_str",
        "b_sig_str_att", "b_sig_str_acc", "b_str", "b_str_att", "b_str_acc",
        "b_td", "b_td_att", "b_td_acc", "b_sub_att", "b_rev", "b_ctrl_sec"]

# Read in data for automation
fighters = pd.read_csv("data\\fighter_stats.csv")

# Read in user input
fighter_1 = input("Fighter 1: ")
fighter_2 = input("Fighter 2: ")
weight = input("Weight (Fl, Bw, Ft, Lw, Ww, Mw, Lh, Hw, Ws, Wf, Wb): ")
is_title = [int(input("Title Fight? (1/0): "))]
gender = [input("Gender (Male/Female): ")]

fighter_1_stats = fighters[fighters["name"] == fighter_1]
fighter_2_stats = fighters[fighters["name"] == fighter_2]

col_ordering = ["wins", "losses", "height", "weight", "reach", "age", 
                "SLpM", "sig_str_acc", "SApM", "str_def", 
                "td_avg", "td_acc", "td_def", "sub_avg"]

weight_classes = {
    "Fl" : "Flyweight",
    "Bw" : "Bantamweight",
    "Ft" : "Featherweight",
    "Lw" : "Lightweight",
    "Ww" : "Welterweight",
    "Mw" : "Middleweight",
    "Lh" : "Light Heavyweight",
    "Hw" : "Heavyweight",
    "Ws" : "Women's Strawweight",
    "Wf" : "Women's Flyweight",
    "Wb" : "Women's Bantamweight"
}

weight = [weight_classes[weight]]



print(weight)

fighters_concat = fighter_1_stats[col_ordering].iloc[0].tolist() + fighter_2_stats[col_ordering].iloc[0].tolist() + weight + is_title + gender + [fighter_1_stats["stance"].iloc[0]] + [fighter_2_stats["stance"].iloc[0]]

print(fighters_concat)

"""# Train first model to predict 
jon_jones_vs_stipe_miocic = pd.DataFrame([
    [27, 1, 193.04, 112.49, 213.36, 37, 4.29, 0.57, 2.22, 0.64, 1.93, 0.45, 0.95, 0.5,
    20, 4, 193.04, 108.86, 203.2, 42, 4.82, 0.53, 3.82, 0.54, 1.86, 0.34, 0.68, 0,
    "UFC Heavyweight Title", 1, "Men", "Orthodox", "Orthodox"]
], columns=pre_features_numeric + pre_features_categorical)

print("Calculating post fight stats predictions")
post_fight_predictions = post_fight_model.predict(jon_jones_vs_stipe_miocic)
print(post_fight_predictions)

post_fight_predictions = pd.DataFrame(post_fight_predictions, columns=post_features_numeric)
combined_input = pd.concat([jon_jones_vs_stipe_miocic, post_fight_predictions], axis=1)

print("Calculating win probabilities")
win_probabilities = win_pred_model.predict_proba(combined_input)
print(f"Fighter 1 Win Probability: {win_probabilities[0][0]}\nFighter 2 Win Probability: {win_probabilities[0][1]}")"""