# main.py
from src.data.preprocess import load_contract
from src.models.rag_pipeline import build_rag_pipeline, ask_question
from src.models.evaluate import evaluate_rag_pipeline
from src.integration.autogen import integrate_autogen

def main():
    # Step 1: Preprocess the contract document
    contract_text = load_contract('/mnt/data/advisory sample.docx')
    with open('data/processed/contract.txt', 'w') as f:
        f.write(contract_text)
    
    # Step 2: Build the RAG pipeline
    qa_chain, vector_store = build_rag_pipeline()
    
    # Step 3: Integrate Microsoft AutoGen
    autogen_qa_chain = integrate_autogen(qa_chain)
    
    # Step 4: Evaluate the RAG pipeline
    questions_answers = [
        {"question": "Who are the parties to the Agreement and what are their defined names?", "answer": "Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)"},
        {"question": "In which street does the Advisor live?", "answer": "1 Rabin st Tel Aviv Israel"}
    ]
    evaluation_metrics = evaluate_rag_pipeline(autogen_qa_chain, vector_store, questions_answers)
    print(evaluation_metrics)
    
    # Example question with AutoGen
    question = "Who are the parties to the Agreement and what are their defined names?"
    answer = ask_question(autogen_qa_chain, vector_store, question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()
