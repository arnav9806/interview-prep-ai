import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_resume(resume_text):

    prompt = f"""
    Analyze the following resume.

    Provide:

    1. Resume score out of 100
    2. Key skills detected
    
    Note : Nothing else should be return except this two things 
    Resume:
    {resume_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    

    result = response.choices[0].message.content
    print(result,"<<<<<<<<<<result")

    return result