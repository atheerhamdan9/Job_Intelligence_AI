import streamlit as st
from pathlib import Path
import tempfile
import requests
import pandas as pd
import io
from src.recommendation import (
    clean_resume_text,
    recommend_jobs,
    analyze_top_jobs,
    jobs_df,
    job_embeddings,
    build_cv_embedding
)


API_URL = "http://127.0.0.1:8000"


def classify_single_text(job_text):
    response = requests.post(
        f"{API_URL}/predict",
        data={"job_description": job_text}
    )

    if response.status_code != 200:
        raise Exception(response.json().get("detail", "Prediction failed"))

    return response.json()


def classify_batch_file(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv"
        )
    }

    response = requests.post(
        f"{API_URL}/predict_batch_csv",
        files=files
    )

    if response.status_code != 200:
        raise Exception(response.json().get("detail", "Batch prediction failed"))

    return pd.read_csv(io.BytesIO(response.content))

st.set_page_config(
    page_title="Job Intelligence AI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

classification_icon = ASSETS_DIR / "classification.svg"
recommendation_icon = ASSETS_DIR / "recommendation.svg"
job_icon = ASSETS_DIR / "job.svg"

if "service" not in st.session_state:
    st.session_state.service = None

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp { background-color: #f8fafc; }

.block-container {
    padding-top: 45px;
    padding-left: 70px;
    padding-right: 70px;
}

.main-title {
    font-size: 56px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 22px;
    color: #64748b;
    margin-bottom: 35px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 18px;
}

.result-card, .job-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    margin-bottom: 18px;
}

.job-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.job-title {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

.score {
    color: #16a34a;
    font-weight: 700;
}

.skill {
    background: #eef2ff;
    color: #2563eb;
    padding: 8px 14px;
    border-radius: 999px;
    display: inline-block;
    margin: 6px;
    font-weight: 600;
}

.missing {
    background: #fff7ed;
    color: #ea580c;
    padding: 8px 14px;
    border-radius: 999px;
    display: inline-block;
    margin: 6px;
    font-weight: 600;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 24px !important;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
header_col1, header_col2 = st.columns([0.5, 12])

with header_col1:
    st.image(str(job_icon), width=70)

with header_col2:
    st.markdown(
        '<div class="main-title">Welcome to Job Intelligence AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-Powered Career Classification & Job Recommendation Platform</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="section-title">Choose a service to get started</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.image(str(classification_icon), width=100)
        st.markdown("<h2 style='color:#2563eb;'>Classification</h2>", unsafe_allow_html=True)
        st.markdown("Upload your file and let AI classify it into the most suitable career category.")

        if st.button("Get Started", key="classification"):
            st.session_state.service = "classification"

with col2:
    with st.container(border=True):
        st.image(str(recommendation_icon), width=100)
        st.markdown("<h2 style='color:#16a34a;'>Recommendation</h2>", unsafe_allow_html=True)
        st.markdown("Upload your resume and get the top 5 most relevant job recommendations.")

        if st.button("Get Started", key="recommendation"):
            st.session_state.service = "recommendation"

st.write("")


# Classification
if st.session_state.service == "classification":
    st.divider()
    st.header("Job Classification")

    # Single Prediction
    st.subheader("Single Prediction")

    job_text = st.text_area(
        "Enter Job Description",
        height=220,
        key="single_job_text"
    )

    if st.button("Single Predict"):

        if job_text.strip():

            result = classify_single_text(job_text)

            st.markdown(f"""
            <div class="result-card">
                <h3>Classification Result</h3>
                <div style="text-align:center;">
                    <p style="color:#64748b;font-size:18px;">Predicted Category</p>
                    <h1 style="color:#2563eb;font-size:42px;">
                        {result["predicted_category"]}
                    </h1>
                    <p style="font-size:18px;">
                        Confidence Score: <b>{result["confidence_score"]:.2%}</b>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(result["confidence_score"])

        else:
            st.warning("Please enter a job description.")

    st.divider()

    # Batch Prediction
    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="classification_upload"
    )

    if uploaded_file is not None:

        if st.button("Batch Predict"):
            try:
                result_df = classify_batch_file(uploaded_file)

                st.success("Batch prediction completed!")

                st.dataframe(result_df.head())

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Classified CSV",
                    data=csv,
                    file_name="classified_jobs.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}")

# Recommendation
if st.session_state.service == "recommendation":
    st.divider()
    st.header("Job Recommendation System")

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        key="recommendation_upload"
    )

    if uploaded_file:
        if st.button("Generate Recommendations"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                cv_path = tmp.name

            with st.spinner("Analyzing resume and generating recommendations..."):


                top_jobs, resume_text = recommend_jobs(

                    cv_path,
                    jobs_df,
                    job_embeddings,
                    build_cv_embedding,
                    top_k=10
                    )

                results = analyze_top_jobs(

                    top_jobs,
                    resume_text
                    )

            st.markdown("### Top 5 Recommended Jobs")

            # for i, job in enumerate(results[:5], 1):


            #     title_col, score_col = st.columns([5, 1])

            #     with title_col:
            #         st.markdown(
            #             f"### {i}. {job.get('Job Title', 'Not Specified')}"
            #         )

            #     with score_col:
            #         st.markdown(
            #             f"""
            #             <div style="
            #             color:#16a34a;
            #             font-size:20px;
            #             font-weight:700;
            #             text-align:right;
            #             margin-top:10px;
            #             ">
            #             {job.get('Match Score', 'N/A')}
            #             </div>
            #             """,
            #             unsafe_allow_html=True
            #         )

            #     st.markdown(
            #         f"**Company:** {job.get('Company', 'Not Specified')}"
            #     )

            #     st.markdown("**Matched Skills:**")

            #     matched_skills = job.get("Matched Skills", [])

            #     if matched_skills:
            #         st.write(", ".join(matched_skills))
            #     else:
            #         st.write("Not specified")

            #     st.markdown("**Missing Skills:**")

            #     missing_skills = job.get("Missing Skills", [])

            #     if missing_skills:
            #         st.write(", ".join(missing_skills))
            #     else:
            #         st.write("Not specified")

            #     st.markdown("**Skill Gap Summary:**")
            #     st.write(
            #         job.get(
            #             "Skill Gap Summary",
            #             "Not specified"
            #         )
            #     )

            #     st.markdown("**Learning Recommendations:**")

            #     learning_recommendations = job.get(
            #         "Learning Recommendations",
            #         []
            #     )

            #     if learning_recommendations:
            #         for rec in learning_recommendations:
            #             st.markdown(f"- {rec}")
            #     else:
            #         st.write("Not specified")

            #     st.markdown("**Explanation:**")
            #     st.write(
            #         job.get(
            #             "Explanation",
            #             "Not specified"
            #         )
            #     )

            #     st.divider()

            #     st.write("")
            for i, job in enumerate(results[:5], 1):
                with st.container(border=True):
                    with st.container(border=True):
                        st.markdown(f"""
                                    <div style=" display:flex;
                                    justify-content:space-between;
                                    align-items:center; ">
                                    <h3 style="margin:0;"> {i}. {job.get("Job Title", "Not Specified")}
                                    </h3> <span style=" color:#16a34a;
                                    font-size:22px; font-weight:700;
                                    "> Match Score: {job.get("Match Score", "N/A")}
                                    </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.write(f"Company: {job.get('Company')}")
                        st.markdown("### Matched Skills")
                        matched_skills = job.get("Matched Skills", [])
                        if matched_skills:
                            for skill in matched_skills:
                                st.success(skill)
                        else:
                            st.write("No matched skills found")
                        st.markdown("### Missing Skills")
                        missing_skills = job.get("Missing Skills", [])
                        if missing_skills:
                            for skill in missing_skills:
                                st.warning(skill)
                        else:
                            st.write("No missing skills")
                        st.markdown("### Skill Gap Summary")
                        st.write( job.get( "Skill Gap Summary", "Not specified" ) )
                        st.markdown("### Learning Recommendations")
                        learning_recommendations = job.get( "Learning Recommendations", [] )
                        if learning_recommendations:
                            for rec in learning_recommendations:
                                st.markdown(f"- {rec}")
                        st.markdown("### 💡 Explanation")
                        st.write( job.get( "Explanation", "Not specified" ) )
                        st.divider()

        if st.button("🗑️ Clear Results"):

            st.session_state.clear()
            st.rerun()
