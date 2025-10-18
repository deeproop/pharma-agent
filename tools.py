# tools.py
import os
import re # <-- Import regex
import requests
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

# (Your existing get_retriever() function and retriever initialization stay the same)
# ...
load_dotenv()

def get_retriever():
    """Initializes and returns the FAISS retriever."""
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("FAISS index not found. Please run ingest.py first.")
        
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return vector_store.as_retriever(search_kwargs={"k": 3}) # <-- Changed k to 3

retriever = get_retriever()
# ...

# (Your existing check_local_database() and query_fda_api() tools stay the same)
# ...
@tool
def check_local_database(drug_name: str) -> dict:
    """
    Searches the local vector database for a drug and returns its ingredients
    if a high-confidence match is found.
    """
    print(f"--- TOOL: Checking local DB for '{drug_name}' ---")
    results = retriever.vectorstore.similarity_search_with_score(drug_name, k=1)
    
    if not results:
        print("--- TOOL: No local match found. ---")
        return {"error": "Drug not found in local database."}

    doc, score = results[0]
    CONFIDENCE_THRESHOLD = 0.3 
    
    if score < CONFIDENCE_THRESHOLD:
        print(f"--- TOOL: Found local match with score {score} ---")
        return doc.metadata 
    else:
        print(f"--- TOOL: Low confidence match (score: {score}). Ignoring. ---")
        return {"error": "Drug not found in local database."}

@tool
def query_fda_api(drug_name: str) -> dict:
    """
    Queries the openFDA API for a drug and returns its active and inactive ingredients.
    """
    print(f"--- TOOL: Querying FDA API for '{drug_name}' ---")
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            active = result.get("active_ingredient", ["Not specified"])[0]
            inactive = result.get("inactive_ingredient", ["Not specified"])[0]
            
            formatted_data = {
                "brand_name": result.get("openfda", {}).get("brand_name", [drug_name])[0],
                "active_ingredients": [active.strip()],
                "inactive_ingredients": [i.strip() for i in inactive.split(',')]
            }
            print("--- TOOL: Found match in FDA API. ---")
            return formatted_data
        else:
            print("--- TOOL: No match found in FDA API. ---")
            return {"error": "Drug not found in FDA database."}
            
    except requests.exceptions.RequestException as e:
        print(f"--- TOOL: FDA API error: {e} ---")
        return {"error": f"API request failed: {e}"}

# --- NEW TOOL FOR STRETCH GOAL ---

@tool
def find_alternative_drugs(active_ingredient: str, original_brand_name: str) -> list[str]:
    """
    Searches the local vector database for drugs with the same active ingredient.
    Filters out the original drug from the results.
    """
    print(f"--- TOOL: Searching alternatives for '{active_ingredient}' ---")
    
    # Clean up the ingredient (e.g., "Acetaminophen 500mg" -> "Acetaminophen")
    # This regex splits at the first digit or " HCl"
    cleaned_ingredient = re.split(r'(\d| HCl)', active_ingredient, 1)[0].strip()
    
    if not cleaned_ingredient:
        print("--- TOOL: Could not parse active ingredient. ---")
        return []

    # Search RAG store for the active ingredient
    results = retriever.vectorstore.similarity_search(cleaned_ingredient, k=3)
    
    alternatives = []
    for doc in results:
        brand_name = doc.metadata.get("brand_name")
        # Check if it's a new drug and not the one we just looked up
        if brand_name and brand_name.lower() != original_brand_name.lower():
            alternatives.append(brand_name)
    
    # Return unique names
    unique_alternatives = list(set(alternatives))
    print(f"--- TOOL: Found alternatives: {unique_alternatives} ---")
    return unique_alternatives