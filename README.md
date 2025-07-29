Resume Ranker | AI-Based Resume Scoring System:

Introduction:

-Resume Ranker is an AI-based Resume Scoring System that automates the process of ranking resumes based on their relevance to a provided job description. This project uses NLP and machine learning techniques to parse resumes, calculate similarity scores, and rank them accordingly.

Features:

1.Upload resumes in PDF format.

2.Paste job description for comparison.

3.Rank Resumes based on relevance.

4.View results in a ranked table.

5.Downloadable HR report in CSV format.

6.Visualize resume scores with a bar chart (Chart.js).

7.Dark Mode Toggle in the UI for improved usability.

Tools & Technologies:

i.Backend: Python, Flask

ii.NLP: NLTK (Natural Language Processing Toolkit)

iii.PDF Extraction: pdfplumber

iv.Machine Learning: Scikit-learn (TF-IDF, Cosine Similarity)

v.Frontend: HTML, Bootstrap, Chart.js

vi.Data Processing: Pandas

vii.Hosting: Render (for the live demo)

Installation & Setup:-

Clone the repository:

git clone (https://github.com/VeldiHruthika/ResumeRankerA.I.) git
cd ResumeRanker_AI

Set up a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Files Included:

a.app.py: Main file to run the Flask application.

b.resume_ranker.py: Python script to process resumes and rank them.

c.index.html: Homepage template where users input job description and upload resumes.

d.results.html: Displays ranked resumes with scores and a bar chart.

e.requirements.txt: Lists all Python dependencies.

->You can view the live demo of the project here: [Resume Ranker Demo](https://resumeranker-ai-wgo1.onrender.com)
