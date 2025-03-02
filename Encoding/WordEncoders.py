import pandas as pd
import numpy as np

def LoadEncoder(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def SentenceEmbeddings(sentence, embeddings, dimensions=300):
    words = str(sentence).split()
    matrix = [embeddings[word] for word in words if word in embeddings]
    return np.mean(matrix, axis=0) if matrix else np.zeros((dimensions,))

# Load embeddings
GloVe = LoadEncoder('C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Encoding\\GloVe6B\\glove.6B.300d.txt')
FastText = LoadEncoder('C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Encoding\\FastText300D\\wiki-news-300d-1M-subword.vec')

# Load dataset
df = pd.read_csv('C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\CleanedDataset.csv')

# Encode Objective and Skillset columns
info = np.array([SentenceEmbeddings(text, GloVe, 300) for text in df['Description']])
final_df_glove = pd.DataFrame(info)
final_df_glove['Positions'] = df['Positions']
final_df_glove.to_csv('C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\EncodedData\\ResumesEncodedGloVe.csv', index=False)

info = np.array([SentenceEmbeddings(text, FastText, 300) for text in df['Description']])
final_df_fasttext = pd.DataFrame(info)
final_df_fasttext['Positions'] = df['Positions']
final_df_fasttext.to_csv('C:\\Users\\viplo\\Desktop\\stuff\\projects\\InterviewPrep\\Datasets\\EncodedData\\ResumesEncodedFastText.csv', index=False)

print('Data Encoded and Saved Successfully!')