from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.services.resume_quality import analyze_resume_quality


def calculate_ats_score(resume_text, jd_text=None):

    print("Starting ATS calculation...")

    resume_chunks = chunk_text(resume_text)
    resume_embeddings = create_embeddings(resume_chunks)

    resume_quality = analyze_resume_quality(resume_text)

    # -----------------------------
    # MODE 1 : JD Provided
    # -----------------------------
    if jd_text and jd_text.strip():

        print("Running semantic comparison with JD")

        jd_embedding = create_embeddings([jd_text])[0]

        similarities = []

        for emb in resume_embeddings:

            sim = cosine_similarity([emb], [jd_embedding])[0][0]

            similarities.append(sim)

        semantic_score = max(similarities) * 100

        final_score = (
            0.8 * semantic_score +
            0.3 * resume_quality
        )

        final_score = round(final_score)

        return {
            "score": final_score,
            "semantic_score": round(semantic_score),
            "quality_score": resume_quality
        }

    # -----------------------------
    # MODE 2 : Resume Only
    # -----------------------------
    else:

        print("No JD provided. Using resume quality score")

        return {
            "score": resume_quality,
            "quality_score": resume_quality
        }


# from app.rag.chunker import chunk_text
# from app.rag.embeddings import create_embeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# from app.services.skill_extractor import extract_skills
# from app.services.resume_quality import analyze_resume_quality


# def calculate_ats_score(resume_text, jd_text=None):

#     print("Starting ATS calculation...")

#     resume_chunks = chunk_text(resume_text)
#     resume_embeddings = create_embeddings(resume_chunks)

#     # -----------------------------------
#     # MODE 1 : Resume vs JD
#     # -----------------------------------

#     if jd_text and jd_text.strip():

#         print("Running Resume vs JD ATS")

#         jd_embedding = create_embeddings([jd_text])[0]

#         similarities = []

#         for emb in resume_embeddings:
#             sim = cosine_similarity([emb], [jd_embedding])[0][0]
#             similarities.append(sim)

#         semantic_score = max(similarities) * 100

#         print("Semantic similarity score:", semantic_score)

#         # -----------------------------
#         # Skill Matching
#         # -----------------------------

#         jd_skills = extract_skills(jd_text)

#         skill_embeddings = create_embeddings(jd_skills)

#         matched = 0

#         for skill_emb in skill_embeddings:

#             scores = cosine_similarity([skill_emb], resume_embeddings)

#             if max(scores[0]) > 0.6:
#                 matched += 1

#         skill_score = (matched / len(jd_skills)) * 100 if jd_skills else 0

#         print("Skill match score:", skill_score)

#         # -----------------------------
#         # Resume Quality
#         # -----------------------------

#         quality_score = analyze_resume_quality(resume_text)

#         # -----------------------------
#         # Final Score
#         # -----------------------------

#         final_score = (
#             0.5 * semantic_score +
#             0.3 * skill_score +
#             0.2 * quality_score
#         )

#         final_score = round(final_score)

#         return {
#             "score": final_score,
#             "semantic_score": round(semantic_score),
#             "skill_score": round(skill_score),
#             "quality_score": quality_score
#         }

#     # -----------------------------------
#     # MODE 2 : Resume Only
#     # -----------------------------------

#     else:

#         print("Running Resume-only ATS")

#         quality_score = analyze_resume_quality(resume_text)

#         return {
#             "score": quality_score,
#             "quality_score": quality_score
#         }

# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# def extract_skills_from_text(text):

#     prompt = f"""
#     Extract all professional skills from the following text.

#     Rules:
#     - Return only comma separated skills
#     - Do not explain anything
#     - Do not add numbering

#     TEXT:
#     {text}
#     """

#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )

#     skills_text = response.choices[0].message.content.strip()
#     print("skills_text=========>>>",skills_text)

#     skills = [s.strip().lower() for s in skills_text.split(",")]
#     print("skills=========>>>",skills)
#     return list(set(skills))


# def calculate_ats_score(resume_text, jd_text):

#     print("Extracting resume skills...")
#     resume_skills = extract_skills_from_text(resume_text)

#     print("Extracting JD skills...")
#     jd_skills = extract_skills_from_text(jd_text)

#     matched = list(set(resume_skills) & set(jd_skills))
#     missing = list(set(jd_skills) - set(resume_skills))

#     print("matched", matched)
#     print("missing", missing)

#     if len(jd_skills) == 0:
#         score = 0
#     else:
#         score = int((len(matched) / len(jd_skills)) * 100)

#     return {
#         "score": score,
#         "resume_skills": resume_skills,
#         "jd_skills": jd_skills,
#         "matched_skills": matched,
#         "missing_skills": missing
#     }