# app/chains/question_chain.py

import os
from groq import Groq
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("Loading Groq API key...")

# Initialize Groq client
client = Groq(api_key=api_key)

print("Groq client initialized")


def generate_questions(query_text: str, difficulty: str, question_type: str):
    """
    Generate interview questions from resume text
    """

    print("Generating interview questions with Groq...")

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

        # Extract response text
        questions_text = response.choices[0].message.content.strip()

        print("Raw LLM output:")
        # print(questions_text)

        # Convert text to list
        questions = [
            q.strip()
            for q in questions_text.split("\n")
            if q.strip()
        ]

        print("Questions generated successfully")
        print("Total questions:", len(questions))

        return questions

    except Exception as e:

        print(" Error generating questions:", e)

        return [
            "Error generating questions. Please try again."
        ]

