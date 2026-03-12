# app/parsers/resume_parser.py

import pypdf
import docx2txt


def parse_pdf(file):
    """
    Extract text from PDF resume
    """
    print("Parsing PDF resume...")
    reader = pypdf.PdfReader(file)
    text = ""
    for page_number, page in enumerate(reader.pages):
        extracted = page.extract_text()
        if extracted:
            print(f"Extracted text from page {page_number + 1}")
            text += extracted + "\n"
        else:
            print(f" No text found on page {page_number + 1}")
    print("PDF parsing completed")
    return text


def parse_docx(file):
    """
    Extract text from DOCX resume
    """
    print("Parsing DOCX resume...")
    text = docx2txt.process(file)
    print("DOCX parsing completed")
    return text


def parse_txt(file):
    """
    Extract text from TXT resume
    """
    print("Parsing TXT resume...")
    text = file.read().decode("utf-8")
    print("TXT parsing completed")
    return text


def parse_resume(file):
    """
    Detect file type and parse resume
    """
    filename = file.name.lower()
    print(f"Uploaded file: {filename}")

    if filename.endswith(".pdf"):
        print("Detected file type: PDF")
        return parse_pdf(file)

    elif filename.endswith(".docx"):
        print("Detected file type: DOCX")
        return parse_docx(file)

    elif filename.endswith(".txt"):
        print("Detected file type: TXT")
        return parse_txt(file)

    else:
        print("Unsupported file format")
        raise ValueError("Unsupported file format. Upload PDF, DOCX or TXT.")