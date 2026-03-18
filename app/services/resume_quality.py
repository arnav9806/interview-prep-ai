import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_resume_quality(resume_text):

    print("Analyzing resume quality...")

    prompt = f"""
Analyze this resume and give a score from 0 to 100 based on:

- clarity
- skills
- projects
- measurable achievements

Resume:
{resume_text}

Return only the score number.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    score = int(response.choices[0].message.content.strip())

    print("Resume quality score:", score)

    return score