from pdf_processing import extract_text_from_pdf, split_text_into_chunks
from embedding import generate_embeddings
from vector_store import VectorStore
from search import search_query
from llm import generate_response

PDF_PATH = "data/KIT Rules and regulation.pdf"

# Extract text from PDF
text = extract_text_from_pdf(PDF_PATH)

# Split text into chunks
chunks = split_text_into_chunks(text)

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Initialize and store in FAISS
vector_store = VectorStore(dimension=len(embeddings[0]))
vector_store.add_documents(chunks, embeddings)

# Run chatbot only when the user enters a question
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break  # Stop execution

    if query:  # Proceed only if input is not empty
        retrieved_text = search_query(query, vector_store)
        answer = generate_response(query, retrieved_text)
        print("\nChatbot Answer:\n", answer)
    else:
        print("Please enter a valid question.")
