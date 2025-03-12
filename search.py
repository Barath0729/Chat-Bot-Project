from embedding import model
from vector_store import VectorStore
import config

def search_query(query, vector_store):
    """Finds the most relevant text chunks based on a query."""
    query_embedding = model.encode(query)
    results = vector_store.search(query_embedding, top_k=config.TOP_K)
    return " ".join(results)
