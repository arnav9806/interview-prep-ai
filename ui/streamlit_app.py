# import streamlit as st
# from app.parsers.resume_parser import parse_resume

import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.rag.chunker import chunk_text
from app.parsers.resume_parser import parse_resume
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
    [
        "Technical",
        "Programming",
        "Scenario Based",
        "HR"
    ]
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

generate_questions = col1.button("Generate Interview Questions")
calculate_ats = col2.button("Calculate ATS Score")
improve_resume = col3.button("Improve Resume")

st.divider()

# =====================================
# Output Section
# =====================================

if generate_questions or calculate_ats or improve_resume:

    if resume_file is None:
        st.error("❌ Please upload a resume first.")
    else:

        try:
            # resume_text = parse_resume(resume_file)
            resume_text = parse_resume(resume_file)

            chunks = chunk_text(resume_text)

            # print("First chunk preview:", chunks[0])
            print("📄 Resume text extracted successfully")
            print("📏 Resume text length:", len(resume_text))

        except Exception as e:
            st.error(f"Error parsing resume: {e}")
            st.stop()

# if generate_questions or calculate_ats or improve_resume:

#     if resume_file is None:
#         st.error("❌ Please upload a resume first.")
#     else:

#         resume_text = resume_file.read()

#         # Generate Questions
#         if generate_questions:

#             st.subheader("Interview Questions")

#             questions = [
#                 "Explain REST API architecture.",
#                 "What is dependency injection?",
#                 "How does Django ORM work?",
#                 "Explain microservices architecture.",
#                 "What is database indexing?"
#             ]

#             for i, q in enumerate(questions, 1):
#                 st.write(f"{i}. {q}")

#         # ATS Score
#         elif calculate_ats:

#             st.subheader("ATS Score")

#             st.metric("Score", "78 / 100")

#             st.write("Missing Skills")

#             st.write("- Docker")
#             st.write("- Kubernetes")
#             st.write("- AWS")

#         # Resume Improvements
#         elif improve_resume:

#             st.subheader("Resume Improvements")

#             suggestions = [
#                 "Add measurable achievements",
#                 "Improve project descriptions",
#                 "Include cloud technologies",
#                 "Add GitHub project links"
#             ]

#             for s in suggestions:
#                 st.write(f"- {s}")