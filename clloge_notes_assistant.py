import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from ollama import chat
import os

class CollegeNotesAssistant:
    def __init__(self, pdf_path, model_name='sentence-transformers/all-MiniLM-L6-v2' , chat_model_name = 'llama3.2', show_debug=False, chunk_size=800, overlap=100, n_results=3, threshold=0.5):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name = "CollegeNotesPDFEmbeddings",
            metadata = {
                "hnsw:space": "cosine",
                "description": "A collection of college notes embeddings for similarity search."
            }
        )
        self.pdf_path = os.path.abspath(pdf_path)
        self.model = SentenceTransformer(model_name)
        self.chat_model_name = chat_model_name
        self.embeddings_generated = False
        self.ingest_pdf()
        self.show_debug = show_debug
        # configure the chunks
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_results = n_results
        self.threshold = threshold

    def customize_chunks(self, chunk_size=None, overlap=None, n_results=None, threshold=None):
        if(self.show_debug):
            print(f"Current chunk size: {self.chunk_size}, Current overlap: {self.overlap}, Current number of results: {self.n_results}, Current threshold: {self.threshold}")
            print("You can customize the chunk size, overlap, number of results, and threshold for similarity search.")
            self.chunk_size = chunk_size if chunk_size is not None else self.chunk_size
            self.overlap = overlap if overlap is not None else self.overlap 
            self.n_results = n_results if n_results is not None else self.n_results
            self.threshold = threshold if threshold is not None else self.threshold
        else:
            print("Debug mode is off. Enable debug mode to customize chunks.")


    def extract_text_from_pdf(self):
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"The file '{self.pdf_path}' does not exist. Please provide a valid path.")
            
            reader = PdfReader(self.pdf_path)
            full_text = ""  
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    page_formatted = f"\n\n--- Page {idx + 1} ---\n\n{page_text}\n\n"
                    full_text += page_formatted
                else:
                    print(f"Warning: No text found on page {idx + 1}.")            
            return full_text
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while extracting text from the PDF: {e}")
            return None

    def text_to_chunks(self, chunk_size=None, overlap=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap

        text = self.extract_text_from_pdf()
        try:
            if text is None:
                raise ValueError("No text extracted from the PDF. Cannot generate embeddings.")
        except ValueError as e:
            print(f"Value error: {e}")
            return None
        chunks = []
        chunk_created = 0
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            chunk_created += 1
            if i + chunk_size >= len(text):
                break
        print(f"Total chunks created: {chunk_created}")
        return chunks

    def text_embedding(self):
        chunk_text = self.text_to_chunks()
        try:
            if chunk_text is None:
                raise ValueError("No text extracted from the PDF. Cannot generate embeddings.")
            embedding = self.model.encode(chunk_text, convert_to_tensor=True)
            self.collection.add(
                ids=[str(i) for i in range(len(chunk_text))],
                documents=chunk_text,
                embeddings=embedding.tolist()
            )
            return embedding
        except ValueError as e:
            print(f"Value error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            return None

    def ingest_pdf(self):
        if self.embeddings_generated:
            print("Embeddings have already been generated for this PDF.")
            return
        else:
            embedding = self.text_embedding()
            if embedding is not None:
                self.embeddings_generated = True
                print("PDF ingestion and embedding generation completed successfully.")
            else:
                print("Failed to generate embeddings for the PDF.")
    def query_pdf(self, query):
        try:
            if not self.embeddings_generated:
                raise ValueError("Embeddings have not been generated yet. Please ingest the PDF first.")
            query_vector = self.model.encode([query], convert_to_tensor=True)
            results = self.collection.query(
                query_embeddings=[query_vector[0].tolist()],
                n_results=self.n_results,
            )
            similarity_scores = 1 - results['distances'][0]
            similarity_chunks = [results['documents'][0][i] for i in range(len(results['documents'][0])) if similarity_scores[i] >= self.threshold]
            if self.show_debug:
                for score, chunk in zip(similarity_scores, results['documents'][0]):
                    if score >= self.threshold:
                        print(f"Similarity Score: {score:.4f}\nChunk: {chunk}\n")
            return similarity_chunks
        except ValueError as e:
            print(f"Value error: {e}")
            return None

    def chat_with_pdf(self, query):
        if query.strip() == "":
            print("Query cannot be empty. Please provide a valid query.")
            return None
        try:
            results = self.query_pdf(query)
            if results is None or len(results) == 0:
                raise ValueError("No relevant documents found for the query.")
            
            context = "\n\n".join(results)
            prompt = f"""Use the following context from my college notes to answer the question.\n
            Context:\n{context}
            \n\nQuestion: {query}
            \n\nConstraint: Provide a concise answer based on the context.Dont make up answers.
            \n\nAnswer:
            """
            message = [{"role": "user", "content": prompt}]
            response = chat(model=self.chat_model_name, messages=message)
            return response["message"]["content"]
        except ValueError as e:
            print(f"Value error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while chatting with the PDF: {e}")
            return None
    def summarize_chat(self,concept=""):
        if concept.strip() == "":
            print("Concept cannot be empty. Please provide a valid concept to summarize.")
            return None
        query = f"Summarize:{concept}"
        return self.chat_with_pdf(query)
    def explain_chat(self,concept=""):
        if concept.strip() == "":
            print("Concept cannot be empty. Please provide a valid concept to explain.")
            return None
        query = f"Explain:{concept}"
        return self.chat_with_pdf(query)

        
def main():
    pdf_path = input("Enter the path to the PDF file: ")
    college_notes_instance = CollegeNotesAssistant(pdf_path)

    while True:
        query = input("Enter your query (or type 'q' to quit): ")
        if query.lower() == 'q':
            print("Exiting the program.")
            break
        response = college_notes_instance.chat_with_pdf(query)
        if response:
            print(f"Response:\n{response}")
        else:
            print("No response generated. Please check your query or the PDF content.")

if __name__ == "__main__":
    main()