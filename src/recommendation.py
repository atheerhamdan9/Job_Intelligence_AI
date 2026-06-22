import re
from pathlib import Path
import pdfplumber
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


BASE_DIR = Path(__file__).resolve().parent.parent

jobs_df = pd.read_csv(BASE_DIR / "data" / "final_jobs_cleaned.csv")

job_embeddings = np.load(BASE_DIR / "data" / "job_embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_cv_embedding(cv_text):
    return model.encode(cv_text)


def extract_information(uploaded_file):

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text



def clean_resume_text(text):
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'(\+?\d[\d\s\-\(\)]{7,})', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\b(linkedin|github)\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def build_job_context(row):
    return f"""
Job Title:
{row['title']}

Company:
{row['name']}

Skills:
{row['Skills']}

Experience:
{row['experience']}

Education:
{row['education']}

Job Description:
{row['job_text']}
"""


def gpt_skill_gap_analysis(resume_text, job_context, match_score):
    prompt = f"""
You are an expert technical recruiter.

Analyze the candidate resume against the job requirements.

Resume:
{resume_text}

Job:
{job_context}

Match Score:
{match_score:.2%}

Return valid JSON only:

{{
    "matched_skills": [],
    "missing_skills": [],
    "skill_gap_summary": "",
    "learning_recommendations": [],
    "recommendation_explanation": ""
}}
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )

    return json.loads(response.output_text)


def compute_final_score(cv_emb, job_emb):
    return cosine_similarity([cv_emb], [job_emb])[0][0]


def recommend_jobs(pdf_path, jobs_df, job_embeddings, build_cv_embedding, top_k=10):
    resume_text = extract_information(pdf_path)
    resume_text = clean_resume_text(resume_text)

    cv_emb = build_cv_embedding(resume_text)

    scores = []

    for job_emb in job_embeddings:
        score = compute_final_score(cv_emb, job_emb)
        scores.append(score)

    temp = jobs_df.copy()
    temp["score"] = scores

    temp = temp.sort_values("score", ascending=False)

    temp = temp.drop_duplicates(
        subset=["title", "name"],
        keep="first"
    )

    return temp.head(top_k), resume_text


def analyze_top_jobs(top_df, resume_text):
    results = []

    for _, row in top_df.head(5).iterrows():
        job_context = build_job_context(row)

        analysis = gpt_skill_gap_analysis(
            resume_text=resume_text,
            job_context=job_context,
            match_score=row["score"]
        )

        results.append({
            "Job Title": row["title"],
            "Company": row["name"],
            "Match Score": f"{row['score']:.2%}",
            "Matched Skills": analysis.get("matched_skills", []),
            "Missing Skills": analysis.get("missing_skills", []),
            "Skill Gap Summary": analysis.get("skill_gap_summary", ""),
            "Learning Recommendations": analysis.get("learning_recommendations", []),
            "Explanation": analysis.get("recommendation_explanation", "")
        })

    return results
