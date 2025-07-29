import pdfplumber
import os
import nltk
nltk.data.path.append("C:/Users/hruth/AppData/Roaming/nltk_data")
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()

    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

def rank_resumes(jd_path, resumes_folder):
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd = f.read()

    jd_processed = preprocess(jd)
    resume_texts = []
    resume_names = []

    for file in os.listdir(resumes_folder):
        if file.endswith('.pdf'):
            path = os.path.join(resumes_folder, file)
            text = extract_text_from_pdf(path)
            processed = preprocess(text)
            resume_texts.append(processed)
            resume_names.append(file)

    all_docs = [jd_processed] + resume_texts
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    result_df = pd.DataFrame({'Resume': resume_names, 'Score': scores})
    result_df.sort_values(by='Score', ascending=False, inplace=True)
    return result_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
