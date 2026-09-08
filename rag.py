import os
import uuid

import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Make sure it is present in your .env file."
    )


# ============================================================
# 2. GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# 3. CHROMADB SETUP
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_documents"
)


# ============================================================
# 4. EXTRACT TEXT FROM PDF
# ============================================================

def extract_pages_from_pdf(pdf_file):
    """
    Extract text from every page of a PDF.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:
            text = text.strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages


# ============================================================
# 5. CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):
    """
    Split text into overlapping chunks.

    Example:

        chunk_size = 1000
        overlap = 200

    Chunk 1: characters 0-1000
    Chunk 2: characters 800-1800
    Chunk 3: characters 1600-2600
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 6. PROCESS A PDF
# ============================================================

def process_pdf(pdf_file):
    """
    Extract text page-by-page and create chunks.

    Every chunk keeps its page number.
    """

    pages = extract_pages_from_pdf(
        pdf_file
    )

    # Check whether PDF contains readable text

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be a scanned or image-only PDF."
        )

    all_chunks = []

    for page in pages:

        page_number = page["page_number"]

        page_text = page["text"]

        page_chunks = chunk_text(
            page_text
        )

        for chunk in page_chunks:

            all_chunks.append({
                "text": chunk,
                "page": page_number
            })

    if not all_chunks:

        raise ValueError(
            "No text could be extracted from this PDF."
        )

    return all_chunks


# ============================================================
# 7. STORE CHUNKS IN CHROMADB
# ============================================================

def store_chunks(
    chunks,
    filename
):
    """
    Store chunks in ChromaDB.

    Every chunk gets:
        - unique ID
        - text
        - source filename
        - page number
    """

    ids = []

    documents = []

    metadatas = []

    for chunk in chunks:

        chunk_id = str(
            uuid.uuid4()
        )

        ids.append(chunk_id)

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "source": filename,
            "page": chunk["page"]
        })

    if documents:

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    return len(documents)


# ============================================================
# 8. SEARCH CHROMADB
# ============================================================

def search_chunks(
    question,
    n_results=5
):
    """
    Retrieve the most relevant chunks
    from ChromaDB.
    """

    total_chunks = collection.count()

    if total_chunks == 0:

        return {
            "documents": [[]],
            "metadatas": [[]]
        }

    # Don't request more chunks than exist

    n_results = min(
        n_results,
        total_chunks
    )

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    return results


# ============================================================
# 9. BUILD CONTEXT
# ============================================================

def build_context(results):
    """
    Convert retrieved ChromaDB results
    into context for the LLM.
    """

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    context_parts = []

    sources = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        source = metadata["source"]

        page = metadata["page"]

        context_parts.append(
            f"""
Source: {source}
Page: {page}

{document}
"""
        )

        sources.append({
            "source": source,
            "page": page
        })

    context = "\n\n".join(
        context_parts
    )

    return context, sources


# ============================================================
# 10. GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    context,
    chat_history=None
):
    """
    Send retrieved document context
    + user question to Groq.
    """

    history_text = ""

    if chat_history:

        history_text = (
            "\n\nPrevious conversation:\n"
        )

        for message in chat_history:

            role = message["role"]

            content = message["content"]

            history_text += (
                f"{role}: {content}\n"
            )

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer questions using ONLY
the information contained in the provided
document context.

IMPORTANT RULES:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the answer cannot be found in the
   provided documents, say:
   "I couldn't find that information
   in the uploaded documents."

4. Give clear and useful answers.

5. When appropriate, mention the source
   document and page number.

6. If the user asks a question that is
   unrelated to the uploaded documents,
   politely explain that you are designed
   to answer questions about the uploaded
   documents.

{history_text}

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# 11. COMPLETE RAG PIPELINE
# ============================================================

def answer_question(
    question,
    n_results=5,
    chat_history=None
):
    """
    Complete RAG pipeline:

        Question
             ↓
        ChromaDB
             ↓
       Relevant chunks
             ↓
          Context
             ↓
           Groq
             ↓
          Answer
    """

    results = search_chunks(
        question,
        n_results=n_results
    )

    documents = results["documents"][0]

    if not documents:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents.",
            []
        )

    context, sources = build_context(
        results
    )

    answer = generate_answer(
        question,
        context,
        chat_history
    )

    return answer, sources


# ============================================================
# 12. GET NUMBER OF STORED CHUNKS
# ============================================================

def get_document_count():

    return collection.count()


# ============================================================
# 13. CLEAR DATABASE
# ============================================================

def clear_database():

    global collection

    try:

        chroma_client.delete_collection(
            name="pdf_documents"
        )

    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name="pdf_documents"
    )