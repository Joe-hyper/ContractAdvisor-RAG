# evaluate.py
from ragas import Evaluator
from ragas.metrics import accuracy, precision, recall, f1_score

def evaluate_rag_pipeline(qa_chain, vector_store, questions_answers):
    evaluator = Evaluator()
    
    results = []
    for qa in questions_answers:
        question = qa['question']
        expected_answer = qa['answer']
        
        docs = vector_store.similarity_search(question, k=5)
        generated_answer = qa_chain.run(input_documents=docs, question=question)
        
        result = {
            'question': question,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer
        }
        results.append(result)
    
    # Calculate evaluation metrics
    accuracy_score = accuracy(results)
    precision_score = precision(results)
    recall_score = recall(results)
    f1_score_value = f1_score(results)
    
    return {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score_value
    }

if __name__ == "__main__":
    qa_chain, vector_store = build_rag_pipeline()
    
    questions_answers = [
        {"question": "Who are the parties to the Agreement and what are their defined names?", "answer": "Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)"},
        {"question": "In which street does the Advisor live?", "answer": "1 Rabin st Tel Aviv Israel"}
    ]
    
    evaluation_metrics = evaluate_rag_pipeline(qa_chain, vector_store, questions_answers)
    print(evaluation_metrics)
