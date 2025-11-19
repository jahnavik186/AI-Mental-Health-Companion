from rag_utils import build_faiss_index, retrieve_similar, generate_response
import pandas as pd

# Load knowledge base
kb = pd.read_csv('data/knowledge_base/articles.csv')
texts = kb['content'].tolist()

# Build vector index
index, embeddings = build_faiss_index(texts)

def chat():
    print("-----AI Mental Health Companion-----")
    name = input("Your Name: ")
    print(f"Hello {name}, I am here to listen and support you.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Take care! Remember, seek professional help when needed.")
            break
        
        # RAG retrieval
        retrieved_docs = retrieve_similar(user_input, texts, index, embeddings, top_k=3)
        context = "\n".join([doc for doc, _ in retrieved_docs])
        
        # LLM response
        response = generate_response(user_input, context=context)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    chat()
