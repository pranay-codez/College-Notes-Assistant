# College Notes AI Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that allows students to ask questions about their college notes stored in PDF files.

The application extracts text from a PDF, splits it into chunks, converts the chunks into embeddings, stores them in ChromaDB, and retrieves relevant information using semantic similarity. Ollama is then used to generate an answer based only on the retrieved context.

## Features

* Accepts a PDF file path from the user
* Extracts text from PDF documents
* Splits extracted text into overlapping chunks
* Generates embeddings using Sentence Transformers
* Stores embeddings in ChromaDB
* Performs semantic similarity search
* Filters retrieved chunks using a similarity threshold
* Uses Ollama to generate grounded answers
* Supports:

  * Normal questions
  * `explain <concept>`
  * `summarize <topic>`
  * `define <term>`
* Includes a debug mode for inspecting similarity scores and retrieved chunks

## RAG Pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Sentence Embeddings
 ↓
ChromaDB
 ↓
Semantic Similarity Search
 ↓
Similarity Threshold Filtering
 ↓
Relevant Context
 ↓
Ollama
 ↓
Grounded Answer
```

## Technologies Used

* Python
* ChromaDB
* PyPDF
* Sentence Transformers
* Ollama
* Llama 3.2

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts text into numerical embeddings that can be compared for semantic similarity.

## Current Retrieval Configuration

The parameters were manually tuned and tested using multiple types of questions.

```python
chunk_size = 370
overlap = 37
n_results = 2
threshold = 0.4
```

### What these parameters control

* **chunk_size** — Maximum number of characters in each text chunk.
* **overlap** — Number of characters shared between consecutive chunks.
* **n_results** — Number of top results retrieved from ChromaDB.
* **threshold** — Minimum similarity score required for a chunk to be included in the final context.

These values are currently used as fixed defaults for this learning project.

## Requirements

* Python 3.9+
* Ollama installed and running
* Llama 3.2 model available in Ollama

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the Ollama model:

```bash
ollama pull llama3.2
```

## How to Run

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd college-notes-ai-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python college_notes_assistant.py
```

The program will ask for the path to your PDF:

```text
Enter the path to the PDF file:
```

Enter the path to your notes PDF and start asking questions.

## Example

```text
============================================================
College Notes AI Assistant
============================================================

You can ask questions about your notes.
Type 'explain <concept>' to explain a concept.
Type 'summarize <topic>' to summarize a topic.
Type 'define <term>' to find a definition.
Type 'debug on/off' to toggle debug output.
Type 'q' to quit.

Enter your query (or type 'q' to quit): what are biomolecules?

Response: Biomolecules are organic compounds produced by living organisms
that are essential for life's processes...
```

## Debug Mode

Debug mode can be enabled using:

```text
debug on
```

It displays the similarity scores and retrieved chunks.

Example:

```text
Similarity Score: 0.6056
Chunk: ...

Similarity Score: 0.5402
Chunk: ...
```

Debug mode can be disabled using:

```text
debug off
```

## Testing and Tuning

The retrieval system was tested using different types of questions, including:

* Direct factual questions
* Semantic questions
* Multi-part questions
* Questions requiring information from multiple sections
* Questions about topics not covered in the PDF

The testing showed that the selected configuration works well for the current learning project, while also revealing limitations with complex queries and retrieval precision.

## Known Limitations

* The current implementation uses fixed retrieval parameters.
* Only the top two retrieved chunks are considered.
* Complex questions requiring information from multiple distant sections may not retrieve all necessary context.
* Some semantically similar but irrelevant chunks can occasionally pass the similarity threshold.
* PDF text extraction quality depends on the structure of the PDF.
* Scanned/image-only PDFs may not work correctly because OCR is not implemented.
* ChromaDB is currently used as an in-memory collection, so embeddings are regenerated when the program starts again.
* The system is designed as a learning project rather than a production-ready RAG application.

## What I Learned

Through this project, I learned how to:

* Extract text from PDFs using PyPDF
* Split documents into chunks
* Generate embeddings using Sentence Transformers
* Perform semantic search using vector embeddings
* Store and retrieve embeddings with ChromaDB
* Use similarity thresholds for retrieval
* Connect a local LLM through Ollama
* Build a basic Retrieval-Augmented Generation pipeline
* Test and tune retrieval parameters
* Reduce hallucination by restricting the LLM to retrieved context

## Future Improvements

Possible improvements for a future version include:

* Better chunking strategies
* Dynamic retrieval parameters
* Persistent ChromaDB storage
* Retrieval reranking
* Better handling of multi-part questions
* Support for scanned PDFs using OCR
* Improved answer citations with page numbers
* A graphical or web interface

## Project Status

Completed as part of my AI Automation / AI Agents learning roadmap.

This project focuses on understanding the fundamentals of Retrieval-Augmented Generation rather than building a production-ready system.
