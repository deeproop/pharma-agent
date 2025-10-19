# tools.py
import os
import re
import requests
import json # <-- Make sure json is imported
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

# (get_retriever, retriever, check_local_database, query_fda_api remain the same)
# ...
def get_retriever():
    """Initializes and returns the FAISS retriever."""
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("FAISS index not found. Please run ingest.py first.")
        
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = get_retriever()
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
            # Check if *both* active and inactive are missing or just ["Not specified"]
            active = result.get("active_ingredient", ["Not specified"])
            inactive = result.get("inactive_ingredient", ["Not specified"])
            
            # If only default values are found, consider it not found for our purpose
            if active == ["Not specified"] and inactive == ["Not specified"]:
                 print("--- TOOL: Found in FDA API, but no specific ingredients listed. ---")
                 return {"error": "Ingredients not specified in FDA data."}

            formatted_data = {
                "brand_name": result.get("openfda", {}).get("brand_name", [drug_name])[0],
                "active_ingredients": [a.strip() for a in active if a != "Not specified"],
                "inactive_ingredients": [i.strip() for i in inactive if i != "Not specified"]
            }
            print("--- TOOL: Found match in FDA API. ---")
            return formatted_data
        else:
            print("--- TOOL: No match found in FDA API. ---")
            return {"error": "Drug not found in FDA database."}
            
    except requests.exceptions.RequestException as e:
        print(f"--- TOOL: FDA API error: {e} ---")
        return {"error": f"API request failed: {e}"}
        
# --- NEW TOOL: Query NIH DailyMed ---
@tool
def query_dailymed_api(drug_name: str) -> dict:
    """
    Queries the NIH DailyMed API for a drug using its SPL ID and extracts 
    active and inactive ingredients from the SPL data.
    """
    print(f"--- TOOL: Querying DailyMed API for '{drug_name}' ---")
    
    # 1. Search for the drug to get its SPL ID
    search_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={drug_name}&page_size=1&page=1"
    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Check if the "data" key exists and contains a non-empty list
        spl_list = search_data.get("data") 
        if not spl_list: # Checks for None or empty list []
            print("--- TOOL: Drug not found in DailyMed search (no data returned). ---")
            return {"error": "Drug not found in DailyMed database."}

        # Get the first result from the list and extract its spl_id
        first_spl_result = spl_list[0] 
        spl_id = first_spl_result.get("spl_id")

        if not spl_id:
            print("--- TOOL: Could not find SPL ID in DailyMed search result. ---")
            return {"error": "SPL ID not found for drug in DailyMed."}

        print(f"--- TOOL: Found SPL ID: {spl_id} ---")

            
        # 2. Fetch the SPL details using the SPL ID
        spl_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{spl_id}.json"
        spl_response = requests.get(spl_url)
        spl_response.raise_for_status()
        spl_data = spl_response.json()

        # 3. Parse ingredients from SPL sections (This can be complex)
        active_ingredients = []
        inactive_ingredients = []
        brand_name = spl_data.get("brand_name", drug_name) # Use provided name as fallback

        if "spl_sections" in spl_data:
            for section in spl_data["spl_sections"]:
                title = section.get("title", "").upper()
                text_content = section.get("text", "")
                
                # Simple parsing - look for keywords in titles
                if "ACTIVE INGREDIENT" in title:
                    # Often ingredients are listed directly in text or simple lists
                    # This is a basic extraction, might need refinement
                    lines = text_content.split('\n')
                    for line in lines:
                        if line.strip() and not line.strip().startswith(("Purpose", "Uses", "Warnings")):
                             active_ingredients.append(line.strip())
                             
                elif "INACTIVE INGREDIENT" in title:
                    # Often a comma-separated list or paragraphs
                    lines = text_content.split('\n')
                    for line in lines:
                        parts = [part.strip() for part in line.split(',') if part.strip()]
                        inactive_ingredients.extend(parts)

        # Remove duplicates and clean up
        active_ingredients = list(set(filter(None, active_ingredients)))
        inactive_ingredients = list(set(filter(None, inactive_ingredients)))

        if not active_ingredients and not inactive_ingredients:
            print("--- TOOL: Found SPL, but failed to parse ingredients. ---")
            return {"error": "Could not parse ingredients from DailyMed SPL data."}

        print("--- TOOL: Successfully parsed ingredients from DailyMed. ---")
        return {
            "brand_name": brand_name,
            "active_ingredients": active_ingredients if active_ingredients else ["Not specified"],
            "inactive_ingredients": inactive_ingredients if inactive_ingredients else ["Not specified"]
        }

    except requests.exceptions.RequestException as e:
        print(f"--- TOOL: DailyMed API error: {e} ---")
        return {"error": f"DailyMed API request failed: {e}"}
    except json.JSONDecodeError:
        print("--- TOOL: DailyMed API returned invalid JSON. ---")
        return {"error": "Failed to decode DailyMed API response."}
        

# (web_search tool remains the same)
# ...
@tool
def web_search(query: str) -> str:
    """
    Performs a web search using the Tavily Search API and returns results as a JSON string.
    """
    import json # Make sure json is imported here or at the top
    print(f"--- TOOL: Performing web search for '{query}' ---")
    search = TavilySearchAPIWrapper() 
    results_list = search.results(query=query, max_results=3)
    return json.dumps(results_list, indent=2)