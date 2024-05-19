import pickle

with open('pickle_test.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

import os
import pandas as pd

lr_dict = {
    "11": "lr = 0.1", "12": "lr = 0.01", "13": "lr = 0.001", "14": "lr = 0.0001"}

def clean_metrics_csv(df, origin, params_dict):
    # Initialize lists to collect the desired rows
    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    params_list = []

    # Iterate through unique epochs
    for epoch in df['epoch'].unique():
        # Get the rows for the current epoch
        epoch_rows = df[df['epoch'] == epoch]
        
        # Extract the row with train_acc and the row with val_acc
        train_row = epoch_rows.dropna(subset=['train_acc']).iloc[-1]
        val_row = epoch_rows.dropna(subset=['val_acc']).iloc[-1]
        
        # Collect data
        epoch_list.append(epoch)
        train_acc_list.append(train_row['train_acc'])
        train_loss_list.append(train_row['train_loss'])
        val_acc_list.append(val_row['val_acc'])
        val_loss_list.append(val_row['val_loss'])
        params = params_dict[origin]
        params_list.append(params)

    # Create the new DataFrame
    result_df = pd.DataFrame({
        'epoch': epoch_list,
        'train_acc': train_acc_list,
        'train_loss': train_loss_list,
        'val_acc': val_acc_list,
        'val_loss': val_loss_list,
        'params': params_list
    })

    return result_df

# List to store all cleaned DataFrames
all_clean_dfs = []

# Current working directory
cwd = os.getcwd()

# Loop through the versions
for ver_num in ["11", "12", "13", "14"]:
    csv_path = os.path.join(cwd, "lightning_logs", f"version_{ver_num}", "metrics.csv")
    csv_df = pd.read_csv(csv_path)
    clean_df = clean_metrics_csv(csv_df, ver_num, lr_dict)
    all_clean_dfs.append(clean_df)

# Concatenate all the cleaned DataFrames
combined_df = pd.concat(all_clean_dfs, ignore_index=True)
combined_df

