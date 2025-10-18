# ingest.py
import json
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    load_dotenv()

    print("Loading local drug data...")
    try:
        with open("local_drugs.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: local_drugs.json not found.")
        return

    # Create LangChain Documents
    documents = []
    for drug in data:
        doc = Document(
            page_content=drug["search_terms"], # This is what the vector search will "see"
            metadata=drug # Store the full object here
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents.")

    print("Initializing Azure OpenAI Embeddings...")
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
    except Exception as e:
        print(f"Error initializing embeddings. Check your .env variables: {e}")
        return

    print("Creating FAISS vector store...")
    # This creates the vector store in memory and saves it to a local folder
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")

    print("\n-----------------------------------------------------")
    print("Ingestion complete! Vector store saved to 'faiss_index'.")
    print("You can now run the Streamlit app: streamlit run app.py")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    main()