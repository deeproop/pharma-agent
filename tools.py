# tools.py
import os
import re
import requests
import json
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import logging

logger = logging.getLogger('pharma_agent')
load_dotenv()
logger.info("Tools dependencies loaded.")

# --- Retriever Setup ---
# ... (get_retriever, retriever setup remains the same) ...
def get_retriever():
    logger.debug("Initializing FAISS retriever.")
    if not os.path.exists("faiss_index"):
        logger.error("FAISS index not found. Please run ingest.py first.")
        raise FileNotFoundError("FAISS index not found. Please run ingest.py first.")
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logger.debug("FAISS index loaded successfully.")
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.exception("Failed to initialize retriever:")
        raise

retriever = get_retriever()
logger.info("Retriever initialized.")

# --- Tools ---
@tool
def check_local_database(drug_name: str) -> dict:
    """ Docstring """
    # ... (check_local_database code remains the same) ...
    logger.info(f"--- TOOL: Checking local DB for '{drug_name}' ---")
    try:
        results = retriever.vectorstore.similarity_search_with_score(drug_name, k=1)
        if not results:
            logger.info("No local match found.")
            return {"error": "Drug not found in local database."}

        doc, score = results[0]
        CONFIDENCE_THRESHOLD = 0.55
        logger.debug(f"Local DB match found. Score: {score}, Threshold: {CONFIDENCE_THRESHOLD}")

        if score < CONFIDENCE_THRESHOLD:
            logger.info(f"Found local match with score {score}")
            return doc.metadata
        else:
            logger.warning(f"Low confidence match (score: {score}). Ignoring.")
            return {"error": "Drug not found in local database."}
    except Exception as e:
        logger.exception(f"Error during local DB search for '{drug_name}':")
        return {"error": f"Error during local DB search: {e}"}


@tool
def query_fda_api(drug_name: str) -> dict:
    """
    Queries the openFDA API for a drug and returns its active and inactive ingredients.
    """
    logger.info(f"--- TOOL: Querying FDA API for '{drug_name}' ---")
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            
            # --- REVISED INGREDIENT PARSING ---
            active_list_from_api = result.get("active_ingredient", ["Not specified"])
            inactive_list_from_api = result.get("inactive_ingredient", ["Not specified"])

            # Process active ingredients (usually simple list)
            processed_active = [
                a.strip() for a in active_list_from_api 
                if a and a.strip() and a != "Not specified"
            ]

            # Process inactive ingredients (often a list with one long string)
            processed_inactive = []
            for item_string in inactive_list_from_api:
                if item_string and item_string.strip() and item_string != "Not specified":
                    # Remove potential leading label like "Inactive ingredients "
                    cleaned_item_string = re.sub(r"^\s*inactive\s+ingredients?\s*[:\-]?\s*", "", item_string, flags=re.IGNORECASE).strip()
                    # Split by comma (and maybe semicolon as fallback), strip whitespace, filter empty
                    parts = [part.strip() for part in re.split(r'[;,]', cleaned_item_string) if part and part.strip()]
                    processed_inactive.extend(parts)

            # Check if we actually found anything meaningful
            has_active = processed_active and processed_active != ["Not specified"]
            has_inactive = processed_inactive and processed_inactive != ["Not specified"]

            if not has_active and not has_inactive:
                 logger.warning(f"Found '{drug_name}' in FDA API, but no specific ingredients could be parsed.")
                 return {"error": "Ingredients not specified or parsable in FDA data."}
            # --- END REVISED PARSING ---

            formatted_data = {
                "brand_name": result.get("openfda", {}).get("brand_name", [drug_name])[0],
                "active_ingredients": processed_active if has_active else ["Not specified"],
                "inactive_ingredients": processed_inactive if has_inactive else ["Not specified"]
            }
            logger.info(f"Found match for '{drug_name}' in FDA API.")
            return formatted_data
        else:
            logger.info(f"No match found for '{drug_name}' in FDA API.")
            return {"error": "Drug not found in FDA database."}

    # ... (Exception handling remains the same) ...
    except requests.exceptions.Timeout:
        logger.error(f"FDA API request timed out for '{drug_name}'")
        return {"error": "FDA API request timed out."}
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"FDA API HTTP error occurred for '{drug_name}': {http_err} - Status Code: {http_err.response.status_code}")
        return {"error": f"API request failed: {http_err.response.status_code} {http_err.response.reason}"}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"FDA API request failed: {req_err}")
        return {"error": f"API request failed: {req_err}"}
    except json.JSONDecodeError:
        logger.error(f"FDA API returned invalid JSON for '{drug_name}'.")
        return {"error": "Failed to decode FDA API response."}


@tool
def web_search(query: str) -> str:
    """ Docstring """
    # ... (web_search code remains the same) ...
    logger.info(f"--- TOOL: Performing web search for '{query}' ---")
    try:
        search = TavilySearchAPIWrapper()
        results_list = search.results(query=query, max_results=3) 
        logger.debug(f"Tavily search returned {len(results_list)} results.")
        return json.dumps(results_list, indent=2)
    except Exception as e:
        logger.exception(f"Tavily web search failed for query '{query}':") 
        return json.dumps([{"error": f"Web search failed: {e}"}])