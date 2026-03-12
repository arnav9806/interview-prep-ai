import streamlit as st
import sys
import os

# Add parent directory to sys.path to avoid 'module not found'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.parsers.resume_parser import parse_resume
from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import ResumeVectorStore
from app.chains.question_chain import generate_questions
from app.services.ats_service import calculate_ats_score
from app.services.resume_analysis import analyze_resume
from app.services.resume_improvement import improve_resume

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="InterviewPrep AI",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# Title
# =====================================
st.title("🤖 InterviewPrep AI")
st.write("AI powered resume analysis and interview preparation")

st.divider()

# =====================================
# Sidebar Settings
# =====================================
st.sidebar.header("Interview Settings")

difficulty = st.sidebar.selectbox(
    "Select Difficulty Level",
    ["Beginner", "Intermediate", "Advanced"]
)

question_type = st.sidebar.selectbox(
    "Select Question Type",
    ["Technical", "Programming", "Scenario Based", "HR"]
)

# =====================================
# Resume Upload
# =====================================
st.header("Upload Resume")

resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"]
)

# =====================================
# Job Description (Optional)
# =====================================
st.header("Job Description (Optional)")

jd_text = st.text_area(
    "Paste job description here",
    height=200
)

st.divider()

# =====================================
# Buttons
# =====================================
col1, col2, col3 = st.columns(3)

generate_questions_btn = col1.button("Generate Interview Questions")
calculate_ats_btn = col2.button("Calculate ATS Score")
improve_resume_btn = col3.button("Improve Resume")

st.divider()

# =====================================
# Output Section
# =====================================
if generate_questions_btn or calculate_ats_btn or improve_resume_btn:

    if resume_file is None:
        st.error("Please upload a resume first.")
    else:
        try:
            # -----------------------
            # Parse Resume
            # -----------------------
            resume_text = parse_resume(resume_file)
            print("Resume text extracted successfully")
            print("Resume text length:", len(resume_text))

            # -----------------------
            # Chunk Resume
            # -----------------------
            chunks = chunk_text(resume_text)
            print("Total chunks created:", len(chunks))
            # print("First chunk preview:\n", chunks[0])

            # -----------------------
            # Create Embeddings
            # -----------------------
            embeddings = create_embeddings(chunks)
            print("Embeddings created")
            print("First embedding vector length:", len(embeddings[0]))
            print("Total embeddings:", len(embeddings))

            # -----------------------
            # FAISS Vector Store
            # -----------------------
            if len(embeddings) > 0:
                vector_store = ResumeVectorStore(embedding_dim=len(embeddings[0]))
                vector_store.add_embeddings(embeddings)
                vector_store.save_index()

                # Optional: Test search
                distances, indices = vector_store.search(embeddings[0], top_k=3)
                print("Search test distances:", distances)
                print("Search test indices:", indices)

            # ==============================
            # Generate Questions
            # ==============================
            if generate_questions_btn:
                st.subheader("Interview Questions")
                questions_list = generate_questions(
                    query_text=resume_text,
                    difficulty=difficulty,
                    question_type=question_type
                )
                for i, q in enumerate(questions_list, 1):
                    st.write(f"{i}. {q}")

            # ==============================
            # ATS Score
            # ==============================
            elif calculate_ats_btn:
                if jd_text.strip():
                    with st.spinner("Calculating ATS Score..."):
                        ats_result = calculate_ats_score(resume_text, jd_text)
                    st.subheader("ATS Score")
                    st.metric("Score", f"{ats_result['score']}%")
                    st.divider()
                    st.write("Matched Skills")

                    if ats_result["matched_skills"]:
                        for skill in ats_result["matched_skills"]:
                            st.write(f"- {skill}")
                    else:
                        st.write("No matched skills found")

                    st.write(" Missing Skills")
                    if ats_result["missing_skills"]:
                        for skill in ats_result["missing_skills"]:
                            st.write(f"- {skill}")
                    else:
                        st.write("No missing skills 🎉")

                else:

                    st.warning("No Job Description provided. Running resume analysis instead.")
                    with st.spinner("Analyzing Resume..."):
                        result = analyze_resume(resume_text)
                    st.subheader("Resume Analysis")

                    st.write(result)
            # elif calculate_ats_btn:
            #     st.subheader("ATS Score")
            #     st.metric("Score", "78 / 100")
            #     st.write("Missing Skills")
            #     st.write("- Docker")
            #     st.write("- Kubernetes")
            #     st.write("- AWS")

            # ==============================
            # Resume Improvements
            # ==============================
            elif improve_resume:
                st.subheader("AI Resume Improvements")
                with st.spinner("Analyzing resume..."):
                    if jd_text.strip():
                        result = improve_resume(resume_text, jd_text)
                    else:
                        result = improve_resume(resume_text)
                st.write(result)
            # elif improve_resume_btn:
            #     st.subheader("Resume Improvements")
            #     suggestions = [
            #         "Add measurable achievements",
            #         "Improve project descriptions",
            #         "Include cloud technologies",
            #         "Add GitHub project links"
            #     ]
            #     for s in suggestions:
            #         st.write(f"- {s}")

        except Exception as e:
            st.error(f"Error processing resume: {e}")
            st.stop()