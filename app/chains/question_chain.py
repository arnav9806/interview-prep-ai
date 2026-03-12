# app/chains/question_chain.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("Loading Groq API key...")

client = Groq(api_key=api_key)

print("Groq client initialized")


def generate_questions(query_text: str, difficulty: str, question_type: str, jd_text: str = None):
    """
    Generate interview questions from resume
    If JD is provided → questions from Resume + JD
    If JD not provided → questions from Resume only
    """

    print("Generating interview questions with Groq...")

    # -------------------------
    # Mode 1 → Resume + JD
    # -------------------------
    if jd_text and jd_text.strip():

        prompt = f"""
You are an expert technical interviewer.

Generate 5 {difficulty} level {question_type} interview questions.

Use BOTH the candidate resume and the job description.

Focus on:
- Candidate experience
- Technologies in job description
- Practical interview scenarios

Resume:
{query_text}

Job Description:
{jd_text}

Rules:
- Only return questions
- Do not include answers
- Each question must be on a new line
"""

    # -------------------------
    # Mode 2 → Resume Only
    # -------------------------
    else:

        prompt = f"""
You are an expert technical interviewer.

Based on the following resume, generate 5 {difficulty} level {question_type} interview questions.

Resume:
{query_text}

Rules:
- Only return questions
- Do not include answers
- Each question must be on a new line
"""

    try:

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )

        questions_text = response.choices[0].message.content.strip()

        print("Raw LLM output received")

        questions = [
            q.strip()
            for q in questions_text.split("\n")
            if q.strip()
        ]

        print("Questions generated successfully")
        print("Total questions:", len(questions))

        return questions

    except Exception as e:

        print("Error generating questions:", e)

        return ["Error generating questions. Please try again."]



# # app/chains/question_chain.py

# import os
# from groq import Groq
# from dotenv import load_dotenv

# # Load .env variables
# load_dotenv()

# api_key = os.getenv("GROQ_API_KEY")

# print("Loading Groq API key...")

# # Initialize Groq client
# client = Groq(api_key=api_key)

# print("Groq client initialized")


# def generate_questions(query_text: str, difficulty: str, question_type: str):
#     """
#     Generate interview questions from resume text
#     """

#     print("Generating interview questions with Groq...")

#     prompt = f"""
# You are an expert technical interviewer.

# Based on the following resume, generate 5 {difficulty} level {question_type} interview questions and one more thing if question is DSA and difficulty is intermediate only then ask questions on array.

# Resume:
# {query_text}

# Rules:
# - Only return questions
# - Do not include answers
# - Each question must be on a new line
# """

#     try:

#         response = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             model="llama-3.3-70b-versatile",
#             temperature=0.7
#         )

#         # Extract response text
#         questions_text = response.choices[0].message.content.strip()

#         print("Raw LLM output:")
#         # print(questions_text)

#         # Convert text to list
#         questions = [
#             q.strip()
#             for q in questions_text.split("\n")
#             if q.strip()
#         ]

#         print("Questions generated successfully")
#         print("Total questions:", len(questions))

#         return questions

#     except Exception as e:

#         print(" Error generating questions:", e)

#         return [
#             "Error generating questions. Please try again."
#         ]

