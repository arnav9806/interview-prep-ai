import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_skills_from_text(text):

    prompt = f"""
    Extract all professional skills from the following text.

    Rules:
    - Return only comma separated skills
    - Do not explain anything
    - Do not add numbering

    TEXT:
    {text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    skills_text = response.choices[0].message.content.strip()
    print("skills_text=========>>>",skills_text)

    skills = [s.strip().lower() for s in skills_text.split(",")]
    print("skills=========>>>",skills)
    return list(set(skills))


def calculate_ats_score(resume_text, jd_text):

    print("Extracting resume skills...")
    resume_skills = extract_skills_from_text(resume_text)

    print("Extracting JD skills...")
    jd_skills = extract_skills_from_text(jd_text)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    print("matched", matched)
    print("missing", missing)

    if len(jd_skills) == 0:
        score = 0
    else:
        score = int((len(matched) / len(jd_skills)) * 100)

    return {
        "score": score,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing
    }