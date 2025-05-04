Objective:
To develop a smart Question-Answering system that allows users to ask questions about resumes (CVs), and automatically get accurate responses by retrieving relevant information from the documents using a Retrieval-Augmented Generation (RAG) approach powered by the LLaMA 2 language model.

Problem:
Recruiters and HR professionals often need to extract specific information from multiple resumes, such as education, experience, skills, or project details. Manually scanning through these resumes is time-consuming and error-prone. There is a need for an intelligent system that can answer specific queries (e.g., "What is the candidate's work experience?", "Has the applicant used Python?") by analyzing unstructured resume data.

Solution:
This project leverages Retrieval-Augmented Generation to:

Ingest resumes in formats like PDF, DOCX, and TXT.

Extract and store the text data efficiently.

Generate embeddings using SentenceTransformers or similar models.

Index the embeddings using a vector store like FAISS or ChromaDB.

Answer queries using the LLaMA 2 model with retrieved context to generate accurate, human-like responses.

Tech Stack:
Language Model: LLaMA 2

Embedding Models: SentenceTransformers / Hugging Face

Vector Store: FAISS / ChromaDB

Frontend: Streamlit

Backend: Python

Document Parsing: PyMuPDF, docx2txt, PDFPlumber

Expected Outcome:
A web-based application that can:

Take in resumes from users.

Let users ask free-form questions about the resume.

Retrieve and present relevant answers based on resume content using RAG.
