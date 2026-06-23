# Job_Intelligence_AI 

<!-- 
Job intelligence is an AI-powered career support system that combines job classification, job recommendation including skill gap analysis. GPT was used to generate labels for unlabeled job descriptions and provide personalized skill gap analysis. DistilBERT was trained to automatically classify jobs, while Sentence Transformers and cosine similarity were used to match resumes with relevant job opportunities. The system also generates learning recommendations to help users improve their qualifications. To make the solution accessible to end users, a Streamlit-based web interface was developed, allowing users to classify individual job postings or batches of job descriptions from CSV files, as well as upload resumes to receive personalized job recommendations, skill gap insights, and learning recommendations through an interactive platform. 


team members :
Lamis Mohammed Alzahrani - LamisMoha24@gmail.com
Noura Nawar Alhuthali - nouraalhuthli76@gmail.com
Ather Alwethinani - atheerhamdan9@gmail.com

-->

# Project Folder Structure:

<!-- 

Job_Intelligence_AI /
├── app/
│   └── main.py
├── assets/
│   └── images.svg
├── data/
│   └── final_jobs_cleaned.csv 
├── frontend/
│   └── streamlit_app.py
├── notebooks/
│   └── Job_Classification.ipynb
├── src/
│   ├── recommendation.py
│── README.md
│── requirements.txt 

 -->


# Setup & Installation Instructions

<!--
 1- Clone the Repository
Open your terminal or command prompt, clone the project repository, and navigate into the project directory:

 git clone [https://github.com/atheerhamdan9/Job_Intelligence_AI.git](https://github.com/atheerhamdan9/Job_Intelligence_AI.git)
cd Job_Intelligence_AI


2- Checkout to Main Branch
Ensure you are operating on the primary branch:

git fetch origin
git checkout main


3- Create and Activate a Virtual Environment
On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

On macOS / Linux (Terminal):
python3 -m venv venv
source venv/bin/activate

4- Install Required Dependencies
install all project libraries listed in the requirements file:

pip install -r requirements.txt

5- et OpenAI API Key Environment Variable

On Windows (PowerShell):

$env:OPENAI_API_KEY="openai_key_here"
-> Verification (Should output a string starting with 'sk-'):
echo $env:OPENAI_API_KEY

On macOS / Linux (Terminal):

export OPENAI_API_KEY="your_actual_openai_key_here"
-> Verification (Should output a string starting with 'sk-'):
echo $OPENAI_API_KEY 

-->



# Running the Application

<!-- 
To run the complete pipeline, you must open two separate terminal windows (ensure the virtual environment (venv) is activated in both).

Terminal 1: Launch FastAPI Backend Server

uvicorn api.main:app

Terminal 2: Launch Streamlit Frontend UI

python -m streamlit run frontend/streamlit_app.py 
-->