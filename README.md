# ContractAdvisor-RAG

Project Overview
This project aims to develop a sophisticated Retrieval-Augmented Generation (RAG) system for contract-related Question and Answer (Q&A) tasks. Our ultimate goal is to build the first fully autonomous artificial contract lawyer capable of drafting, reviewing, and negotiating contracts without human assistance.

Key Objectives:
Build a Contract-Optimized Q&A System
Evaluate and Improve the RAG System
Tasks:
Build a Simple Q&A Pipeline with RAG Using Langchain
Build a RAG Evaluation Pipeline with RAGAS
Integrate Microsoft AutoGen
Optimize the Contract Q&A System
Project Structure:

contract-qa-rag/
├── data/
│   ├── raw/               # Raw contract documents
│   ├── processed/         # Preprocessed contract data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_build_rag_pipeline.ipynb
│   ├── 03_evaluate_rag_pipeline.ipynb
│   ├── 04_optimize_rag_pipeline.ipynb
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   ├── models/
│   │   ├── rag_pipeline.py
│   │   ├── evaluate.py
│   │   ├── optimize.py
│   ├── integration/
│   │   ├── autogen.py
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
├── Dockerfile             # Docker setup
└── main.py                # Entry point for the project
