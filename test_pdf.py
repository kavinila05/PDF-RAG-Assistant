from rag import (
    extract_text_from_pdf,
    chunk_text,
    answer_question
)


# ==========================================
# 1. Read PDF
# ==========================================

with open("sample.pdf", "rb") as file:

    text = extract_text_from_pdf(file)


print("Number of characters:", len(text))


# ==========================================
# 2. Create chunks
# ==========================================

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))


# ==========================================
# 3. Ask a question
# ==========================================

question = "What is institutional theory?"

print("\nQuestion:")
print(question)


# ==========================================
# 4. Run the complete RAG pipeline
# ==========================================

answer = answer_question(question)


# ==========================================
# 5. Display answer
# ==========================================

print("\n==========================================")
print("ANSWER")
print("==========================================")

print(answer)