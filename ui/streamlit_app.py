import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.parsers.resume_parser import parse_resume
from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embeddings, create_query_embedding
from app.rag.vector_store import run_debug_pipeline
from app.rag.query_builder import build_dynamic_query
from app.rag.vector_store import create_collection, add_chunks, search
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
            # embeddings = create_embeddings(chunks)
            chunks_with_embeddings, embeddings = create_embeddings(chunks)
            print("Embeddings created")
            # print("First embedding vector length:", len(embeddings[0]))
            if embeddings:
                print("First embedding vector length:", len(embeddings[0]))
            else:
                print("❌ No embeddings created")
            print("Total embeddings:", len(embeddings))
            
            # -----------------------
            # QDRANT INITIALIZATION
            # -----------------------
            from app.rag.vector_store import (
                create_collection,
                add_chunks,
                search,
                check_collection,
                debug_qdrant
            )

            print("\n🚀 INITIALIZING QDRANT PIPELINE")

            # 1. Check connection
            debug_qdrant()
            
            # -----------------------
            # RESET COLLECTION (DEV ONLY)
            # -----------------------
            from app.rag.vector_store import client

            try:
                client.delete_collection("resume_chunks")
                print("🗑️ Old collection deleted")
            except:
                print("No existing collection to delete")

            # 2. Create collection
            create_collection(vector_size=len(embeddings[0]))


            # -----------------------
            # STORE RESUME (ALWAYS)
            # -----------------------
            if len(embeddings) > 0:
                print("\n📄 Storing Resume Chunks...")
                add_chunks(chunks_with_embeddings, embeddings, source="resume")


            # -----------------------
            # PROCESS JOB DESCRIPTION (FIXED)
            # -----------------------
            if jd_text and jd_text.strip():

                print("\n📄 Processing Job Description...")

                # 1. Chunk JD
                jd_chunks = chunk_text(jd_text)
                print("✅ JD chunks created:", len(jd_chunks))

                # 2. Embed JD
                jd_chunks_with_embeddings, jd_embeddings = create_embeddings(jd_chunks)
                print("✅ JD embeddings created:", len(jd_embeddings))

                # 3. Store JD
                if len(jd_embeddings) > 0:
                    print("📦 Storing JD chunks in Qdrant...")
                    add_chunks(jd_chunks_with_embeddings, jd_embeddings, source="jd")
                else:
                    print("❌ No JD embeddings created")


            # -----------------------
            # VERIFY EVERYTHING (IMPORTANT)
            # -----------------------
            check_collection(limit=10)
            
            query_text = build_dynamic_query(chunks=chunks_with_embeddings, jd_text=jd_text)

            query_vector = create_query_embedding(query_text)

            results = search(
                query_vector,
                top_k=5,
                source="resume"   # 🔥 IMPORTANT
            )

                # -----------------------
                # STREAMLIT UI OUTPUT
                # -----------------------
            st.subheader("🔍 Top Matching Resume Sections")

            for res in results:
                    st.write(f"**Section:** {res['section']}")
                    st.write(f"**Score:** {res['score']:.4f}")
                    st.write(res["text"][:300])
                    st.divider()

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


        except Exception as e:
            st.error(f"Error processing resume: {e}")
            st.stop()