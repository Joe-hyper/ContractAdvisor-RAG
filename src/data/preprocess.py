# preprocess.py
from docx import Document

def load_contract(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

if __name__ == "__main__":
    contract_text = load_contract('.../data/advisory sample.docx')
    with open('data/processed/contract.txt', 'w') as f:
        f.write(contract_text)
