# LLM_QNA
 Project Name: DocuQuery - RAG-Powered Document Q&A with Llama 2

Description:
DocuQuery is a Retrieval-Augmented Generation (RAG) system that enables users to ask questions and get accurate answers from their uploaded documents (PDF, DOCX, TXT). Powered by Llama 2 and vector embeddings, it provides context-aware responses by searching through document contents rather than relying solely on the LLM's knowledge.

Key Features:

Document processing for PDF, DOCX, and TXT files

Vector embeddings using SentenceTransformers

FAISS/ChromaDB for efficient similarity search

Llama 2 for context-aware answer generation

Streamlit-based interactive web interface

Tech Stack:

LLM: Llama 2 (via Hugging Face)

Embeddings: SentenceTransformers/all-MiniLM-L6-v2

Vector DB: FAISS or ChromaDB

Document Processing: PyMuPDF, pdfplumber, docx2txt

Framework: LangChain

UI: Streamlit

README.md
DocuQuery - RAG Document Q&A System
RAG Architecture Diagram

DocuQuery is a document question-answering system that uses Retrieval-Augmented Generation (RAG) with Llama 2 to provide accurate, context-aware answers from your uploaded documents.

Features
📄 Supports PDF, DOCX, and TXT documents

🔍 Semantic search using vector embeddings

🤖 Powered by Meta's Llama 2 LLM

💬 Natural language question answering

🚀 Fast retrieval with FAISS/ChromaDB

🖥️ Easy-to-use Streamlit interface

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/docuquery.git
cd docuquery
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
Usage
Place your documents in the documents/ folder

Run the Streamlit app:

bash
streamlit run app.py
Open the app in your browser and:

Upload documents

Wait for processing to complete

Ask questions about the document content

Configuration
Create a .env file with your configuration:

ini
# For Hugging Face Llama 2
HF_API_KEY=your_huggingface_token

# Vector database settings
VECTOR_DB=faiss  # or chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM settings
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
Architecture
graph TD
    A[User Uploads Documents] --> B[Document Processing]
    B --> C[Text Extraction]
    C --> D[Embedding Generation]
    D --> E[Vector Database Storage]
    F[User Query] --> G[Query Embedding]
    G --> H[Similarity Search]
    H --> I[Context Retrieval]
    I --> J[LLM Answer Generation]
    J --> K[Response to User]

Evaluation
The system is evaluated using:

BLEU score for answer quality

ROUGE metrics for content coverage

Precision/Recall for retrieval accuracy

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License.
