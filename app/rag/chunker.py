from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_text(text):
    """
    Fix spaced characters like 'P Y T H O N'
    """
    print("\n🧹 Cleaning parsed text...")

    text = re.sub(
        r'(\b[A-Z]\s+){2,}[A-Z]\b',
        lambda x: x.group(0).replace(" ", ""),
        text
    )

    print(" Cleaning completed")

    return text

def normalize_section_name(name):
    """
    Normalize different section names into standard categories
    """

    original_name = name
    name = name.lower().strip()

    # 🔥 Remove special characters
    name = name.replace("&", "and")
    name = name.replace("-", " ")
    name = name.replace("_", " ")

    # 🔥 Remove extra spaces
    name = " ".join(name.split())

    # ==============================
    # 🎯 SECTION MAPPING
    # ==============================

    # EDUCATION
    if any(keyword in name for keyword in [
        "educat", "academic", "qualification", "degree"
    ]):
        return "education"

    # SKILLS FIRST (IMPORTANT ORDER)
    elif any(keyword in name for keyword in [
        "skill", "expertise", "technology", "tech stack", "competenc"
    ]):
        return "skills"

    # EXPERIENCE AFTER
    elif any(keyword in name for keyword in [
        "experience", "work", "employment", "career", "intern"
    ]):
        return "experience"

    # PROJECTS
    elif any(keyword in name for keyword in [
        "project", "portfolio", "case study"
    ]):
        return "projects"

    # SUMMARY / PROFILE
    elif any(keyword in name for keyword in [
        "summary", "profile", "about", "objective"
    ]):
        return "summary"

    # CERTIFICATIONS
    elif any(keyword in name for keyword in [
        "certification", "certificate", "license"
    ]):
        return "certifications"

    # ACHIEVEMENTS
    elif any(keyword in name for keyword in [
        "achievement", "award", "honor"
    ]):
        return "achievements"

    # CONTACT
    elif any(keyword in name for keyword in [
        "contact", "phone", "email"
    ]):
        return "contact"

    # ==============================
    # ⚠️ NOISE FILTER (VERY IMPORTANT)
    # ==============================

    # If looks like address/location → ignore as section
    if "," in original_name:
        return "general"

    # If too long → not a section
    if len(original_name.split()) > 6:
        return "general"

    # Default fallback
    return "general"

def is_section_header(line):
    """
    Improved detection of section headers (strict)
    """

    line = line.strip()

    if not line:
        return False

    # ❌ Ignore obvious non-headers
    if any(char.isdigit() for char in line):
        return False  # phone numbers, years

    if "@" in line:
        return False  # emails

    if line.startswith("•") or line.startswith("-"):
        return False  # bullet points

    if len(line) > 50:
        return False  # too long → likely sentence

    words = line.split()

    # ✅ Condition 1: ALL CAPS (strong signal)
    if line.isupper() and 1 <= len(words) <= 5:
        return True

    # ✅ Condition 2: Title Case (but strict)
    if line.istitle() and 1 <= len(words) <= 5:
        return True

    # ✅ Condition 3: Ends with colon (strong signal)
    if line.endswith(":") and len(words) <= 6:
        return True

    return False


def split_into_sections(text):
    """
    Dynamically split resume into sections
    """

    print("\n" + "="*80)
    print("🔍 STEP 1: DETECTING SECTIONS")
    print("="*80)

    sections = {}
    current_section = "general"
    sections[current_section] = []

    lines = text.split("\n")

    print(f" Total lines in resume: {len(lines)}")

    for idx, line in enumerate(lines):

        if is_section_header(line):
            # current_section = line.strip()
            raw_section = line.strip()
            current_section = normalize_section_name(raw_section)

            print(f"🧩 [Line {idx}] Detected NEW section → '{raw_section}' | Normalized → '{current_section}'")
            # sections[current_section] = []
            if current_section not in sections:
                sections[current_section] = []  
            print(f"\n🧩 [Line {idx}] Detected NEW section → '{current_section}'")

        else:
            # sections[current_section].append(line)
                # If current section is 'general' and we already have a real section,
            # attach content to last valid section
            if current_section == "general" and len(sections) > 1:
                last_section = list(sections.keys())[-1]
                sections[last_section].append(line)
            else:
                sections[current_section].append(line)

    # Convert lists to text
    for key in sections:
        sections[key] = "\n".join(sections[key]).strip()

    print("\n📂 FINAL SECTIONS CREATED:")
    for sec_name, sec_text in sections.items():
        print(f" - {sec_name} (Length: {len(sec_text)} chars)")

    print(f"\n Total sections detected: {len(sections)}")

    return sections


def chunk_text(text):
    """
    Production-grade chunking:
    - Dynamic section detection
    - Semantic chunking
    - Metadata support
    """

    print("\n" + "="*80)
    print("🚀 STEP 2: STARTING CHUNKING PROCESS")
    print("="*80)
    
    text = clean_text(text)

    sections = split_into_sections(text)
    
    sections = split_into_sections(text)

    # ======================================
    # 🔥 MERGE SMALL SECTIONS (ADD HERE)
    # ======================================
    print("\n🔧 Merging small sections...")

    MIN_SECTION_LENGTH = 200

    merged_sections = {}
    buffer_text = ""

    for sec, sec_text in sections.items():

        print(f" Section '{sec}' length: {len(sec_text)}")

        if len(sec_text) < MIN_SECTION_LENGTH:
            print(f"  ➡️ Adding '{sec}' to buffer (too small)")
            buffer_text += "\n" + sec_text
        else:
            if buffer_text:
                print(f"  🔗 Merging buffered content into '{sec}'")
                sec_text = buffer_text + "\n" + sec_text
                buffer_text = ""

            merged_sections[sec] = sec_text

    # If anything left in buffer → attach to last section
    if buffer_text and merged_sections:
        last_key = list(merged_sections.keys())[-1]
        print(f"  🔗 Attaching leftover buffer to '{last_key}'")
        merged_sections[last_key] += "\n" + buffer_text

    sections = merged_sections

    print("\n✅ Sections after merging:")
    for sec, txt in sections.items():
        print(f" - {sec}: {len(txt)} chars")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    total_chunks = 0

    for section_name, section_text in sections.items():

        if not section_text.strip():
            print(f"\n⚠️ Skipping empty section: {section_name}")
            continue

        print("\n" + "-"*80)
        print(f"📌 PROCESSING SECTION: {section_name}")
        print(f" Section length: {len(section_text)} characters")
        print("-"*80)

        chunks = splitter.split_text(section_text)

        print(f" Total chunks created in this section: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "section": section_name,
                    "chunk_id": i,
                    "length": len(chunk)
                }
            }

            print("\n" + "~"*60)
            print(f"🧩 CHUNK #{total_chunks}")
            print(f" Section: {section_name}")
            print(f" Local Chunk ID: {i}")
            print(f" Length: {len(chunk)} characters")
            print(f" Preview:\n{chunk[:300]}")
            print("~"*60)

            all_chunks.append(chunk_data)
            total_chunks += 1

    print("\n" + "="*80)
    print("✅ FINAL CHUNKING SUMMARY")
    print("="*80)
    print(f" Total chunks created: {len(all_chunks)}")

    section_distribution = {}
    for chunk in all_chunks:
        sec = chunk["metadata"]["section"]
        section_distribution[sec] = section_distribution.get(sec, 0) + 1

    print("\n📊 CHUNK DISTRIBUTION BY SECTION:")
    for sec, count in section_distribution.items():
        print(f" - {sec}: {count} chunks")

    print("="*80 + "\n")

    return all_chunks