

-- Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents, retrieve relevant information using semantic search, and generate grounded answers using a Groq-hosted LLM.

This project implements the core RAG pipeline directly using **ChromaDB, PyPDF, Streamlit, and Groq**, without using high-level frameworks such as LangChain or LlamaIndex.

The goal is to understand how document-based AI systems work internally — from PDF ingestion and chunking to vector retrieval, context construction, and LLM generation.

---

## Architecture Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                       │
│                                                              │
│                     Streamlit Web UI                         │
│                                                              │
│   ┌──────────────────┐          ┌────────────────────────┐   │
│   │   PDF Upload     │          │    Chat Interface      │   │
│   │                  │          │                        │   │
│   │  Single / Multi  │          │  Ask questions about   │   │
│   │      PDF         │          │  uploaded documents    │   │
│   └────────┬─────────┘          └───────────┬────────────┘   │
└────────────┼────────────────────────────────┼────────────────┘
             │                                │
             │                                │ Question
             ▼                                ▼
┌────────────────────────┐        ┌───────────────────────────┐
│      PDF INGESTION     │        │       RETRIEVAL           │
│                        │        │                           │
│        PyPDF           │        │        ChromaDB           │
│                        │        │                           │
│  Extract text/page     │        │  Query → Embedding       │
│  information           │        │        ↓                  │
└───────────┬────────────┘        │  Similarity Search        │
            │                     │        ↓                  │
            ▼                     │  Relevant Chunks          │
┌────────────────────────┐        └─────────────┬─────────────┘
│      CHUNKING          │                      │
│                        │                      │
│  Page-wise text        │                      │
│  chunking               │                      │
│                        │                      │
│  Chunk Size: 1000      │                      │
│  Overlap: 200          │                      │
└───────────┬────────────┘                      │
            │                                   │
            ▼                                   │
┌────────────────────────┐                      │
│       ChromaDB         │◄─────────────────────┘
│                        │
│  Embeddings            │
│  Vector Storage        │
│  Metadata              │
│                        │
│  Source + Page Number  │
└───────────┬────────────┘
            │
            │ Retrieved Context
            ▼
┌──────────────────────────────────────────────────────────────┐
│                         GROQ LLM                             │
│                                                              │
│                     GPT-OSS 20B                              │
│                                                              │
│  Question + Retrieved Context + Conversation History         │
│                         ↓                                    │
│                    Grounded Answer                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Streamlit UI   │
                  │                  │
                  │ Answer + Sources │
                  │ + Page Numbers   │
                  └──────────────────┘
RAG Pipeline

The application follows a complete Retrieval-Augmented Generation pipeline:

                 PDF DOCUMENTS
                      │
                      ▼
              ┌───────────────┐
              │     PyPDF     │
              │               │
              │ Text Extract  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    Chunking   │
              │               │
              │ 1000 chars    │
              │ 200 overlap   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   ChromaDB    │
              │               │
              │  Embeddings   │
              │ Vector Store  │
              └───────┬───────┘
                      │
                      │
              USER QUESTION
                      │
                      ▼
              ┌───────────────┐
              │   Retrieval   │
              │               │
              │ Similarity    │
              │ Search        │
              └───────┬───────┘
                      │
                      ▼
              Relevant Chunks
                      │
                      ▼
              ┌───────────────┐
              │ Context       │
              │ Construction   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Groq LLM     │
              │ GPT-OSS 20B   │
              └───────┬───────┘
                      │
                      ▼
                  ANSWER
                      │
                      ▼
              Source + Page
AI / Components Used
| Component       | Technology                  | Purpose                                        |
| --------------- | --------------------------- | ---------------------------------------------- |
| PDF Extraction  | `PyPDF`                     | Extract text from PDF documents                |
| Text Processing | Custom Python               | Clean and split extracted text                 |
| Embeddings      | ChromaDB embedding function | Convert text into vector representations       |
| Vector Database | `ChromaDB`                  | Store and retrieve document chunks             |
| Retrieval       | ChromaDB similarity search  | Find relevant chunks for a question            |
| LLM             | `openai/gpt-oss-20b`        | Generate answers using retrieved context       |
| UI              | `Streamlit`                 | Interactive document upload and chat interface |
| Configuration   | `python-dotenv`             | Secure API key management                      |
Why RAG?

Large Language Models can answer questions from their training knowledge, but they do not automatically know the contents of a private PDF uploaded by a user.

RAG solves this problem by introducing a retrieval step before generation.

Instead of:

User Question
      ↓
     LLM
      ↓
   Answer

the application uses:

User Question
      ↓
Retrieve relevant information
      ↓
Relevant document chunks
      ↓
LLM
      ↓
Grounded Answer

The LLM therefore receives the relevant document information as context before generating the answer.

How the Application Works
Stage 1 — PDF Upload

The user uploads one or more PDF documents through the Streamlit interface.

Example:

research-paper.pdf
agriculture-policy.pdf
government-report.pdf

Multiple documents can be stored in the same ChromaDB collection.

Each document remains identifiable through metadata.

Stage 2 — PDF Text Extraction

PyPDF reads the PDF page by page.

For every page, the application stores:

Page Number
     +
Extracted Text

Example:

Page: 12

Institutional theory explains how organizations
are influenced by rules, norms and institutions...

Preserving the page number allows the application to later show where retrieved information came from.

Stage 3 — Text Chunking

Large documents are divided into smaller pieces called chunks.

The current implementation uses:

Chunk size : 1000 characters
Overlap    : 200 characters

For example:

                 DOCUMENT

┌─────────────────────────────────────┐
│              Chunk 1                │
│                                     │
│         0 → 1000 characters         │
└──────────────────┬──────────────────┘
                   │
              200 character
                 overlap
                   │
                   ▼
          ┌─────────────────────────────┐
          │          Chunk 2            │
          │                             │
          │       800 → 1800            │
          └─────────────────────────────┘

The overlap reduces the possibility of important information being split exactly at a chunk boundary.

Stage 4 — Embedding and Vector Storage

Each chunk is stored in ChromaDB.

Conceptually:

Text
 ↓
Embedding Model
 ↓
Vector
 ↓
ChromaDB

The vector representation allows semantically similar pieces of text to be found even when the wording is different.

Each stored chunk also contains metadata:

{
    source: "research-paper.pdf",
    page: 12
}
Stage 5 — User Question

The user asks a question through the chat interface.

For example:

What is institutional theory?

The question is passed to ChromaDB for retrieval.

Stage 6 — Semantic Retrieval

ChromaDB compares the question against the stored document embeddings.

The application retrieves the most relevant chunks.

For example:

Question
   │
   ▼
ChromaDB
   │
   ├── Chunk 17 → Page 8
   ├── Chunk 23 → Page 12
   ├── Chunk 31 → Page 15
   ├── Chunk 42 → Page 21
   └── Chunk 48 → Page 24

The current application retrieves up to 5 relevant chunks.

Stage 7 — Context Construction

The retrieved chunks are combined into a context that is passed to the LLM.

Conceptually:

DOCUMENT CONTEXT

Source: research-paper.pdf
Page: 12

Institutional theory explains...

Source: research-paper.pdf
Page: 15

Institutions influence organizational behavior...

USER QUESTION

What is institutional theory?
Stage 8 — LLM Generation

The context and user question are sent to:

Groq
  ↓
openai/gpt-oss-20b

The model is instructed to answer using the supplied document context rather than inventing information.

If the retrieved context does not contain the answer, the application instructs the model to say:

I couldn't find that information in the uploaded documents.
Stage 9 — Source Display

The application keeps the source metadata associated with every retrieved chunk.

The UI can therefore display:

📚 Sources

📄 research-paper.pdf — Page 12
📄 research-paper.pdf — Page 15

This provides basic traceability between the answer and the original document.

Project Structure
simple-rag/
│
├── app.py                    # Streamlit user interface
│
├── rag.py                    # Complete RAG pipeline
│
├── requirements.txt          # Python dependencies
│
├── README.md                 # Project documentation
│
├── .env.example              # API key configuration template
│
├── .gitignore                # Files excluded from Git
│
├── test_pdf.py               # PDF extraction test
│
└── test_groq.py              # Groq API connection test
app.py

The Streamlit application layer.

Responsible for:

PDF upload
Multiple document selection
Processing controls
Progress bar
Chat interface
Conversation history
Source display
Database clearing
rag.py

The core RAG implementation.

Responsible for:

PDF extraction
Page tracking
Text chunking
ChromaDB storage
Similarity retrieval
Context construction
Groq LLM calls
Answer generation
Source tracking
Database management
requirements.txt

Contains all Python packages required to run the application.

test_pdf.py

Development test used to verify that PDF text can be extracted correctly.

test_groq.py

Development test used to verify that the Groq API connection and selected model work correctly.

RAG Pipeline — Step by Step
Stage	Component	What Happens
1	Streamlit	User uploads PDF documents
2	PyPDF	Text is extracted page by page
3	Chunking	Text is divided into overlapping chunks
4	ChromaDB	Chunks are embedded and stored
5	User	User submits a question
6	ChromaDB	Relevant chunks are retrieved
7	Context Builder	Retrieved chunks are combined
8	Groq	Context and question are sent to the LLM
9	GPT-OSS 20B	Generates a grounded answer
10	Streamlit	Answer and sources are displayed
How It All Connects
User
 │
 │ Upload PDF
 ▼
app.py
 │
 ▼
process_pdf()
 │
 ├── extract_pages_from_pdf()
 │
 └── chunk_text()
 │
 ▼
store_chunks()
 │
 ▼
ChromaDB
 │
 │
 │ User asks question
 │
 ▼
answer_question()
 │
 ▼
search_chunks()
 │
 ▼
ChromaDB similarity search
 │
 ▼
Relevant document chunks
 │
 ▼
build_context()
 │
 ▼
generate_answer()
 │
 ▼
Groq API
 │
 ▼
GPT-OSS 20B
 │
 ▼
Generated Answer
 │
 ├── Answer
 │
 └── Source + Page Number
 │
 ▼
Streamlit UI
Setup & Running
Prerequisites

Before running the application, install:

Python 3.10+
Git
A Groq API key
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git

Move into the project:

cd YOUR-REPOSITORY-NAME
2. Create a Virtual Environment
Windows
python -m venv venv

Activate:

venv\Scripts\activate
macOS / Linux
python3 -m venv venv

Activate:

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure the Groq API Key

Create a file named:

.env

in the project root.

Use .env.example as a template.

Add:

GROQ_API_KEY=your_groq_api_key_here

Replace the placeholder with your own Groq API key.

⚠️ Security

Never commit .env to GitHub.

The repository's .gitignore prevents .env from being tracked.

5. Start the Application

Run:

streamlit run app.py

Streamlit will start a local server.

Open the URL shown in the terminal, usually:

http://localhost:8501
Using the Application
Upload Documents

Upload one or more PDF files from the sidebar.

📄 research-paper.pdf
📄 government-report.pdf
📄 agriculture-policy.pdf
Process Documents

Click:

⚙️ Process uploaded PDFs

The application will:

Read PDF
    ↓
Extract text
    ↓
Create chunks
    ↓
Generate embeddings
    ↓
Store chunks in ChromaDB
Ask Questions

After processing the documents, ask questions through the chat interface.

Example:

What is institutional theory?

The application retrieves relevant chunks before generating the answer.

Follow-up Questions

Conversation history is maintained during the Streamlit session.

Example:

User:
What is institutional theory?

Assistant:
...

User:
What are its main characteristics?

Assistant:
...
View Sources

Every retrieved answer can expose its associated document and page information.

📚 Sources

research-paper.pdf — Page 12
research-paper.pdf — Page 15
ChromaDB

ChromaDB is used as the local vector database.

The application creates:

chroma_db/

This directory contains the local vector store.

It is intentionally excluded from GitHub.

When another user clones the repository, they can build their own vector database by uploading their own documents.

Configuration

The current RAG configuration includes:

Parameter	Current Value
Chunk Size	1000 characters
Chunk Overlap	200 characters
Retrieved Chunks	5
LLM	openai/gpt-oss-20b
Temperature	0
Vector Database	ChromaDB
PDF Parser	PyPDF
UI	Streamlit
Known Limitations
1. Scanned PDFs

The current implementation primarily works with PDFs containing selectable text.

Image-only/scanned PDFs require OCR.

OCR support is planned for a future version.

2. Character-Based Chunking

The current chunking strategy uses character boundaries.

It does not yet understand:

Paragraph boundaries
Section headings
Tables
Semantic boundaries

A more advanced chunking strategy could improve retrieval quality.

3. Retrieval

The current system uses semantic similarity retrieval.

More advanced approaches could include:

Hybrid search
Keyword + vector search
Reranking
Metadata filtering
Query rewriting
Multi-query retrieval
4. Follow-up Question Retrieval

Conversation history is provided to the LLM, but retrieval currently operates primarily on the current question.

For example:

User:
What is institutional theory?

User:
What are its characteristics?

The second question may benefit from query rewriting before retrieval.

A future version can convert:

"What are its characteristics?"

into:

"What are the characteristics of institutional theory?"

before querying ChromaDB.

5. Local Vector Database

The current application uses a local ChromaDB database.

A production deployment could use a hosted or persistent vector database depending on scale and deployment requirements.

Acknowledgements
ChromaDB — Vector database and embedding functionality
PyPDF — PDF text extraction
Streamlit — Application interface
Groq — LLM inference
GPT-OSS 20B — Language model used for answer generation


