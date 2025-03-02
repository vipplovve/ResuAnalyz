import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def process_dataset(file_path):
    df = pd.read_csv(file_path)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df.iloc[:, :-1])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(df.columns[:-1]))

    target_encoder = OneHotEncoder(sparse_output=False)
    encoded_target = target_encoder.fit_transform(df.iloc[:, -1].values.reshape(-1, 1))
    encoded_target_df = pd.DataFrame(encoded_target, columns=target_encoder.get_feature_names_out([df.columns[-1]]))

    final_df = pd.concat([encoded_df, encoded_target_df], axis=1)

    output_dir = "C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets"
    output_file_path = os.path.join(output_dir, "ProcessedProficiencyDataset.csv")
    final_df.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}")

    plt.figure(figsize=(10, 6))
    sns.countplot(x=df.iloc[:, -1])
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()

    numerical_df = final_df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    print("Dataset Insights:")
    print(final_df.describe())

    print("\nClass Distribution:")
    print(df.iloc[:, -1].value_counts())

file_path = "C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\ProficiencyDataset.csv"
process_dataset(file_path)
