import pdfplumber
import os
import nltk
nltk.data.path.append("C:/Users/hruth/AppData/Roaming/nltk_data")
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, render_template, request, send_file
app = Flask(__name__)
UPLOAD_FOLDER = 'resumes'
OUTPUT_CSV = 'output/hr_report.csv'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    job_desc = request.form['jobdesc']
    resumes_folder = UPLOAD_FOLDER

    if not os.path.exists(resumes_folder):
        os.makedirs(resumes_folder)

    # Save job description to a text file
    with open('job_description.txt', 'w', encoding='utf-8') as f:
        f.write(job_desc)

    # Save uploaded resumes
    files = request.files.getlist('resumes')
    for file in files:
        if file.filename.endswith('.pdf'):
            file.save(os.path.join(resumes_folder, file.filename))

    # Rank resumes
    result_df = rank_resumes('job_description.txt', resumes_folder)

    # Save HR report CSV
    if not os.path.exists('output'):
        os.makedirs('output')
    result_df.to_csv(OUTPUT_CSV, index=False)

    # Convert to dictionary for table/chart
    records = result_df.to_dict(orient='records')
    return render_template('results.html', tables=[result_df.to_html(classes='table table-bordered table-hover', index=False)], records=records)

@app.route('/download')
def download():
    return send_file(OUTPUT_CSV, as_attachment=True)


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Extract plain text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Preprocess text: lowercase, remove punctuation, stopwords
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()

    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# Core function to score and rank resumes
def rank_resumes(jd_path, resumes_folder):
    # Read the job description
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

    # TF-IDF vectorization + similarity scoring
    all_docs = [jd_processed] + resume_texts
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    result_df = pd.DataFrame({'Resume': resume_names, 'Score': scores})
    result_df.sort_values(by='Score', ascending=False, inplace=True)
    return result_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
