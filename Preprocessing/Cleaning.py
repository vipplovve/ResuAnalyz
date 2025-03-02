import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[^a-zA-Z ]', ' ', text).lower().strip()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def extract_primary_job(position):
    if not isinstance(position, str):
        return ""
    
    found_jobs = [job for job in tech_positions if job in position]
    return found_jobs[0] if found_jobs else ""

file_name = "C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\MainDataset.csv"
df = pd.read_csv(file_name)

text_columns = ['career_objective', 'skills', 'positions']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

tech_positions = {
    "software engineer", "data scientist", "machine learning engineer", 
    "data analyst", "sde", "software developer", "full stack engineer intern", "software analyst",
    "data engineer", "research scientist", "data science intern", "software engineering intern"
}

df = df[df['positions'].apply(lambda x: any(job in x for job in tech_positions))]
df['positions'] = df['positions'].apply(extract_primary_job)

df = df[(df['career_objective'] != '') & (df['skills'] != '') & (df['positions'] != '')]

df['Description'] = df['career_objective'] + ' ' + df['skills']

new_df = df[['Description', 'positions']]
new_df.columns = ['Description', 'Positions']

new_file_name = "C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\CleanedDataset.csv"
new_df.to_csv(new_file_name, index=False)

print(f"Cleaned dataset saved as {new_file_name}")

import matplotlib.pyplot as plt

position_counts = new_df['Positions'].value_counts()

plt.figure(figsize=(10, 6))
position_counts.plot(kind='bar')
plt.title('Distribution of Job Positions')
plt.xlabel('Job Positions')
plt.ylabel('Number of Data Points')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Number of data points for each job position:")
print(position_counts)

plt.figure(figsize=(8, 8))
position_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Job Positions Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()