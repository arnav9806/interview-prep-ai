import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def improve_resume(resume_text, jd_text=None):

    if jd_text:

        prompt = f"""
        You are an expert resume reviewer.

        A candidate wants to optimize their resume for a specific job.

        Analyze the resume and job description.

        Provide:

        1. Resume strengths
        2. Skills missing from resume but present in job description
        3. Resume improvements to better match the job
        4. Improved bullet points
        5. Suggestions to increase ATS score

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}
        Note: Only Written 5 points except that don't written any thing
        """

    else:

        prompt = f"""
        You are an expert resume reviewer.

        Analyze the following resume and provide:

        1. Resume strengths
        2. Resume weaknesses
        3. Missing important skills
        4. Improved versions of weak bullet points
        5. Suggestions to improve the resume

        RESUME:
        {resume_text}
        
        Note: Only Written 5 points except that don't written any thing 
        """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content