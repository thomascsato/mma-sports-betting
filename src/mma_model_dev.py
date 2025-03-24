import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
import joblib

def duplicate_and_switch(df):
    """This function will double the dataset, switching the red and blue fighter in order
    to not fit the model based on order of fighter in the data."""

    # Create a copy of the original DataFrame
    df_switched = df.copy()

    # Define columns that need to be switched
    columns_to_switch = [
        "wins_agg", "losses_agg", "height_agg", "weight_agg", "reach_agg",
        "SLpM_agg", "sig_str_acc_agg", "SApM_agg", "str_def_agg",
        "td_avg_agg", "td_acc_agg", "td_def_agg", "sub_avg_agg", 
        "age_agg", "kd", "sig_str", "sig_str_att", "sig_str_acc",
        "str_att", "str_acc", "td", "td_att", "td_acc", "str",
        "sub_att", "rev", "ctrl_sec", "stance_agg"
    ]

    # Swap the corresponding 'r_' and 'b_' columns
    for col in columns_to_switch:
        df_switched['b_' + col] = df['r_' + col]  # Assign red fighter stats to blue fighter
        df_switched['r_' + col] = df['b_' + col]  # Assign blue fighter stats to red fighter

    # Adjust winner column based on new roles
    df_switched['winner'] = df['winner'].apply(lambda x: 1 if x == 0 else 0)  # Assuming 1 for red win, 0 for blue win

    return pd.concat([df, df_switched], ignore_index=True)

def prepare_data(df):

    global pre_features_numeric, pre_features_categorical, post_features_numeric

    # Create pre-fight features for both fighters
    pre_features_numeric = ["r_wins_agg",
        "r_losses_agg", "r_height_agg", "r_weight_agg", "r_reach_agg", "r_age_agg",
        "r_SLpM_agg", "r_sig_str_acc_agg", "r_SApM_agg", "r_str_def_agg",
        "r_td_avg_agg", "r_td_acc_agg", "r_td_def_agg", "r_sub_avg_agg", "b_wins_agg",
        "b_losses_agg", "b_height_agg", "b_weight_agg", "b_reach_agg", "b_age_agg",
        "b_SLpM_agg", "b_sig_str_acc_agg", "b_SApM_agg", "b_str_def_agg",
        "b_td_avg_agg", "b_td_acc_agg", "b_td_def_agg", "b_sub_avg_agg"]
    pre_features_categorical = ["weight_class", "is_title_bout", "gender", "r_stance_agg", "b_stance_agg"]

    # Create post-fight features for both fighters
    post_features_numeric = ["finish_round", "total_rounds", "time_sec", 
        "r_kd", "r_sig_str", "r_sig_str_att", "r_sig_str_acc", "r_str",
        "r_str_att", "r_str_acc", "r_td", "r_td_att", "r_td_acc",
        "r_sub_att", "r_rev", "r_ctrl_sec", "b_kd", "b_sig_str",
        "b_sig_str_att", "b_sig_str_acc", "b_str", "b_str_att", "b_str_acc",
        "b_td", "b_td_att", "b_td_acc", "b_sub_att", "b_rev", "b_ctrl_sec"]

    # Combining pre fight features with post fight features
    all_features_numeric = pre_features_numeric + post_features_numeric
    all_features_categorical = pre_features_categorical

    # Combining pre-fight features and post-fight features
    all_pre_features = pre_features_numeric + pre_features_categorical
    all_post_features = post_features_numeric
    all_features = pre_features_numeric + pre_features_categorical + post_features_numeric
    
    # Splitting predictors and targets
    X_pre = df[all_pre_features]
    X_all = df[all_features]
    y_post = df[all_post_features]
    y_win = df["winner"]  # Assuming 1 for r_fighter win, 0 for b_fighter win
    
    return X_pre, X_all, y_post, y_win, all_features_numeric, all_features_categorical

def win_prob_single_vector(post_fight_pred, pre_fight_data):
    """This function will combine the original data and the predicted data into one vector
    for input into the train_win_prob_model function."""

    #post_fight = pd.DataFrame(post_fight_pred)
    #pre_fight = pd.DataFrame(pre_fight_data)

    combined_input = pd.concat([pre_fight_data, post_fight_pred], axis=1)

    return combined_input

def train_post_fight_model(fighter_input_data, X_pre, y_post, pre_features_numeric, pre_features_categorical):
    """This function aims to predict post-fight statistics from two fighters.
    This will then be fed into a second model that predicts win probabilities.
    """

    # Split data into training and testing
    # X_train, X_test, y_train, y_test = train_test_split(X_pre, y_post, test_size=0.2, random_state=42)

    # Training on all data available
    X_train = X_pre
    y_train = y_post

    # Imputing target data
    imputer = SimpleImputer(strategy='mean')
    y_train_imputed = pd.DataFrame(imputer.fit_transform(y_train), columns=y_train.columns)

    # Transform numeric variables with scaling and imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Transform categorical variables with one hot encoding and imputation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    # Use transformers to preprocess columns
    preprocessor = ColumnTransformer(
        transformers=[
            # Transforming only on X_train columns (pre-fight features)
            ('num', numeric_transformer, pre_features_numeric),
            ('cat', categorical_transformer, pre_features_categorical)
        ])
    
    # Multi-output regressor handles multiple target variables
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('multi_output_regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=5, verbose=2, random_state=42)))
    ])

    # Fit the pipeline
    print("Fitting model.")
    model.fit(X_train, y_train_imputed)
    print("Model fit successfully.")
    
    # Evaluate model
    print("Evaluating model.")
    y_pred = model.predict(fighter_input_data)
    print("Model evaluated successfully.")
    
    return model, y_pred

def train_win_prob_model(post_pred, X_all, y_win, all_features_numeric, all_features_categorical):
    """This model aims to predict win probabilities based on predicted post-fight stats."""

    # Split the data into training and testing
    # X_train, X_test, y_train, y_test = train_test_split(X_all, y_win, test_size=0.2, random_state=42)

    # Training on all data available
    X_train = X_all
    y_train = y_win

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_features_numeric),
            ('cat', categorical_transformer, all_features_categorical)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, verbose=2))
    ])
    
    # Fit the pipeline
    model.fit(X_train, y_train)
    
    # Evaluate model
    #y_pred = model.predict(post_pred)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"Model Accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    y_prob = model.predict_proba(post_pred)
    
    return model, y_prob

def save_model(post_model, win_model):
    joblib.dump(post_model, 'post_fight_model.joblib')
    joblib.dump(win_model, 'win_pred_model.joblib')

if __name__ == "__main__":
    
    # Load your fight data
    df = pd.read_csv('data\\all_fights_final.csv')

    # Duplicate values and concatenate
    df_full = duplicate_and_switch(df)
    
    # Prepare data
    X_pre, X_all, y_post, y_win, all_numeric, all_categorical = prepare_data(df_full)
    
    # Train first model to predict 
    jon_jones_vs_stipe_miocic = pd.DataFrame([
        [27, 1, 193.04, 112.49, 213.36, 37, 4.29, 0.57, 2.22, 0.64, 1.93, 0.45, 0.95, 0.5,
        20, 4, 193.04, 108.86, 203.2, 42, 4.82, 0.53, 3.82, 0.54, 1.86, 0.34, 0.68, 0,
        "UFC Heavyweight Title", 1, "Men", "Orthodox", "Orthodox"]
    ], columns=pre_features_numeric + pre_features_categorical)

    post_fight_model, prediction = train_post_fight_model(jon_jones_vs_stipe_miocic, X_pre, y_post, pre_features_numeric, pre_features_categorical)
    
    prediction = pd.DataFrame([
        [3.6667, 5.0000, 268.2000,
        0.7000, 66.4000, 163.8000, 0.5693, 121.9333, 200.1333, 0.6260, 2.3000, 4.4000, 0.6800, 0.8000, 0.0000, 119.6000,
        0.0000, 35.8000, 117.7333, 0.4600, 47.6000, 103.6000, 0.4400, 0.0000, 7.4000, 0.0160, 0.0000, 0.0000, 5.0000]
    ], columns=post_features_numeric)

    prediction_vector = win_prob_single_vector(prediction, jon_jones_vs_stipe_miocic)
    print(prediction_vector)

    win_prob_model, win_prob = train_win_prob_model(prediction_vector, X_all, y_win, all_numeric, all_categorical)
    print(win_prob)