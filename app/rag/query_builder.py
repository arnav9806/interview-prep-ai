# # app/rag/query_builder.py

def build_dynamic_query(chunks, jd_text=None):

    skills = []
    experience = []
    summary = []

    for chunk in chunks:
        section = chunk["metadata"].get("section", "")
        text = chunk["text"]

        if section == "skills":
            skills.append(text[:80])
        elif section == "experience":
            experience.append(text[:80])
        elif section == "summary":
            summary.append(text[:80])

    # ===============================
    # CASE 1: JD PROVIDED
    # ===============================
    if jd_text and jd_text.strip():
        query = (
            jd_text[:300] + " " +      # limit JD
            " ".join(skills[:2])       # add skills
        )

        print("\n🧠 QUERY (JD + SKILLS):\n", query)

    # ===============================
    # CASE 2: NO JD
    # ===============================
    else:
        query = (
            " ".join(skills[:2]) + " " +       # HIGH
            " ".join(experience[:1]) + " " +   # MED
            " ".join(summary[:1])              # LOW
        )

        print("\n🧠 QUERY (RESUME ONLY):\n", query)

    return query.strip()