import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        """Initializes FAISS vector database."""
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_documents(self, chunks, embeddings):
        """Adds document embeddings to FAISS index."""
        np_embeddings = np.array(embeddings).astype('float32')
        self.index.add(np_embeddings)
        self.chunks = chunks  # Store text chunks for retrieval

    def search(self, query_embedding, top_k=3):
        """Searches the FAISS index for the closest vectors."""
        query_embedding = np.array([query_embedding]).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
