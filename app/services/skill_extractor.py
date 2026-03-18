import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_skills(text):

    print("Extracting skills using LLM...")

    prompt = f"""
Extract the important skills from the following job description.

Return only a list of skills separated by commas.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    skills_text = response.choices[0].message.content.strip()

    skills = [s.strip() for s in skills_text.split(",")]

    print("Extracted skills:", skills)

    return skills