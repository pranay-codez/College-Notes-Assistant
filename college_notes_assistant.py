import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ollama import chat
import os

class CollegeNotesAssistant:
    def __init__(self, pdf_path, model_name='sentence-transformers/all-MiniLM-L6-v2' , chat_model_name = 'llama3.2', show_debug=False):
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
        # tuneable parameters
        self.chunk_size = 370
        self.overlap = 37
        self.n_results = 2
        self.threshold = 0.4
        self.show_debug = show_debug
        self.ingest_pdf()
        
       

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

    def text_to_chunks(self):
        
        text = self.extract_text_from_pdf()
        try:
            if text is None:
                raise ValueError("No text extracted from the PDF. Cannot generate embeddings.")
        except ValueError as e:
            print(f"Value error: {e}")
            return None
        chunks = []
        chunk_created = 0
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            chunk_created += 1
            if i + self.chunk_size >= len(text):
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
            similarity_scores = [1 - d for d in results['distances'][0]]
            similarity_chunks = [chunk for score, chunk in zip(similarity_scores, results['documents'][0]) if score >= self.threshold]
            
            if self.show_debug:
                for score, chunk in zip(similarity_scores, results['documents'][0]):
                    print(f"Similarity Score: {score:.4f}")
                    print(f"Chunk: {chunk}\n")

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
            prompt = f"""You are a study assistant for college notes. Answer the question using ONLY the context provided below from the student's notes.\n

            Rules:
            1. Use ONLY information from the provided context. Do not use your general knowledge.\n
            2. If the context does not contain enough information to answer, say: "This is not covered in the notes."\n
            3. Be specific and cite details from the context when possible.\n
            4. If multiple chunks contain relevant information, combine them into a complete answer.\n
            5. Do not make up information or add details not present in the context.\n
            Context:\n{context}
            \n\nQuestion: {query}
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

    
    def define_chat(self,concept=""):
        if concept.strip() == "":
            print("Concept cannot be empty. Please provide a valid concept to define.")
            return None
        query = f"Define:{concept}"
        return self.chat_with_pdf(query)

def main():
    print("="*60)
    print("College Notes AI Assistant")
    print("="*60)

    pdf_path = input("\nEnter the path to the PDF file: ")
    college_notes_instance = CollegeNotesAssistant(pdf_path)

    print("\nYou can ask questions about your notes.")
    print("Type 'explain <concept>' to explain a concept.")
    print("Type 'summarize <topic>' to summarize a topic.")
    print("Type 'define <term>' to find a definition.")
    print("Type 'debug on/off' to toggle debug output.")
    print("Type 'q' to quit.\n")

    while True:
        user_input = input("Enter your query (or type 'q' to quit): ").strip()
        if user_input == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        if user_input.lower().startswith("debug "):
            mode = user_input.split()[1].lower()
            if mode == "on":
                college_notes_instance.show_debug = True
                print("Debug mode is ON.\n")
            elif mode == "off":
                college_notes_instance.show_debug = False
                print("Debug mode is OFF.\n")
                continue
            else:
                print("Invalid debug command. Using default condition (debug off).\n")

        elif user_input.lower().startswith("summarize "):
            topic = user_input[10:].strip()
            try:
                if topic == "":
                    raise ValueError("Topic cannot be empty. Please provide a valid topic to summarize.")
                response = college_notes_instance.summarize_chat(topic)
                if response is None:
                    raise ValueError("No response generated for the topic. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue

        elif user_input.lower().startswith("define "):
            term = user_input[7:].strip()
            try:
                if term == "":
                    raise ValueError("Term cannot be empty. Please provide a valid term to define.")
                response = college_notes_instance.define_chat(term)
                if response is None:
                    raise ValueError("No response generated for the term. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue

        elif user_input.lower().startswith("explain "):
            concept = user_input[8:].strip()
            try:
                if concept == "":
                    raise ValueError("Concept cannot be empty. Please provide a valid concept to explain.")
                response = college_notes_instance.explain_chat(concept)
                
                if response is None:
                    raise ValueError("No response generated for the concept. Please check your input.")
                print(f"Response: {response}\n")
                continue
            except ValueError as e:
                print(f"Value error: {e}")
                continue
            
            
        else:
            print("Processing your query...")
            response = college_notes_instance.chat_with_pdf(user_input)
            print(f"Response: {response}\n")
            continue
if __name__ == "__main__":
    main()