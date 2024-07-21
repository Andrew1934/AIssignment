# 1. Load and explore the dataset.

import pandas as pd
import numpy as np


file_path = "/workspaces/AIssignment/telco_churn_data.csv"
df = pd.read_csv(file_path)

# 2. Review data file: Export .txt file with key attributes

import os # file structure to export data overview to new file

output_folder = "/workspaces/AIssignment/"
data_overview = "original_data_overview.txt"
file_path = os.path.join(output_folder,data_overview)

df_shape = df.shape
with open(file_path,'a') as file:
    file.write(f"Number of rows: {df_shape[0]}\n")
    file.write(f"Number of columns: {df_shape[1]}\n\n\n")

df_head_str = df.head(20).to_string(index=False)

with open(file_path,'a') as file:
    file.write(df_head_str)
    file.write("\n\n\n")

from io import StringIO

buffer = StringIO()
df.info(buf = buffer)
info_str = buffer.getvalue()

with open(file_path,'a') as file:
    file.write(info_str)

# 3. Clean data file

# Remove superfluous and potentially bias fields (eg.inconcistent customer satisfaction score)
df_cleaned = df.drop(columns=['Referred a Friend', 'Number of Referrals', 'Offer', 'Internet Type', 'Avg Monthly Long Distance Charges', 'City', 'Latitude', 'Longitude', 'CLTV', 'Churn Category', 'Churn Reason', 'Customer Satisfaction'])

# No data formatting required.

# Encode category data
df_encoded = pd.get_dummies (df_cleaned, columns=['Contract', 'Payment Method', 'Gender'])

# Save the cleaned data to a new CSV file
df_encoded.to_csv('training_data.csv', index=False)

