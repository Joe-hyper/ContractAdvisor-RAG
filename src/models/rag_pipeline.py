# rag_pipeline.py
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceLLM
from langchain.vectorstores import SimpleVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_contract_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def build_rag_pipeline():
    # Load the contract text
    contract_text = load_contract_text('data/processed/contract.txt')
    
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    contract_chunks = text_splitter.split_text(contract_text)
    
    # Load pre-trained embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a vector store
    vector_store = SimpleVectorStore(embeddings)
    vector_store.add_texts(contract_chunks)
    
    # Load the QA chain
    llm = HuggingFaceLLM.from_pretrained("deepset/roberta-base-squad2")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    return qa_chain, vector_store

def ask_question(qa_chain, vector_store, question):
    docs = vector_store.similarity_search(question, k=5)
    answer = qa_chain.run(input_documents=docs, question=question)
    return answer

if __name__ == "__main__":
    qa_chain, vector_store = build_rag_pipeline()
    
    # Example question
    question = "Who are the parties to the Agreement and what are their defined names?"
    answer = ask_question(qa_chain, vector_store, question)
    print(answer)
