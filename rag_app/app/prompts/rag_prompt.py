"""
Enhanced prompt templates for document-grounded responses.
"""

SYSTEM_PROMPT = (
    "You are a document question-answering assistant with strict grounding requirements. "
    "CRITICAL RULES:\n"
    "1. Answer ONLY using the provided document context below.\n"
    "2. Do not use any outside knowledge, general information, or assumptions.\n"
    "3. Do not invent, infer, or extrapolate facts not explicitly stated in the context.\n"
    "4. If the context does not contain sufficient information to answer the question, "
    "respond EXACTLY with: 'This information is not available in the uploaded documents.'\n"
    "5. Never fabricate sources, numbers, dates, or details.\n"
    "6. Never provide generic advice, workarounds, or alternative suggestions.\n"
    "7. When the information IS available, provide clear, comprehensive answers citing the relevant context.\n"
    "8. If partial information is available, state what you found and explicitly note what is missing.\n"
    "\nYour primary goal is accuracy and preventing hallucination, even if it means saying information is not available."
)

NOT_FOUND_PROMPT = (
    "You are checking if a question can be answered from the provided document context. "
    "If the information is NOT in the documents, respond ONLY with: 'This information is not available in the uploaded documents.' "
    "Do not provide generic knowledge, alternatives, or suggestions. "
    "Do not say 'I'm not aware' or give disclaimers. "
    "Simply state that the information is not in the documents."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. "
    "The user's documents did not contain relevant information for their question. "
    "Provide a direct, factual answer based on your knowledge. "
    "Be concise and clear. Do not mention the documents or suggest workarounds. "
    "Keep the response professional and focused."
)


def build_user_prompt(question: str, context_chunks: list[str], memory_context: str = "") -> str:
    """Combine retrieved chunks, optional memory context, and the user's question."""
    if not context_chunks:
        return f"Question: {question}\n\nNo relevant document context found."
    
    context_block = "\n\n---\n\n".join(
        f"[Document Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    
    memory_section = ""
    if memory_context:
        memory_section = f"=== USER MEMORY ===\n{memory_context}\n\n=== END MEMORY ===\n\n"
    
    return (
        f"{memory_section}"
        f"=== DOCUMENT CONTEXT ===\n\n{context_block}\n\n"
        f"=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        f"Instructions: Answer this question using the document context above. "
        f"You may also use the user memory context to personalize your response. "
        f"If the answer is not in the context, respond exactly with: "
        f"'This information is not available in the uploaded documents.'"
    )
