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
import logging # <-- Import logging

# Get the specific logger instance
logger = logging.getLogger('pharma_agent')

load_dotenv()
logger.info("Tools dependencies loaded.")

# --- Retriever Setup ---
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
        logger.exception("Failed to initialize retriever:") # Logs traceback
        raise

retriever = get_retriever()
logger.info("Retriever initialized.")

# --- Tools ---
@tool
def check_local_database(drug_name: str) -> dict:
    """
    Searches the local vector database for a drug and returns its ingredients
    if a high-confidence match is found.
    """
    logger.info(f"--- TOOL: Checking local DB for '{drug_name}' ---")
    try:
        results = retriever.vectorstore.similarity_search_with_score(drug_name, k=1)
        if not results:
            logger.info("No local match found.")
            return {"error": "Drug not found in local database."}

        doc, score = results[0]
        # Adjust threshold as needed based on testing
        CONFIDENCE_THRESHOLD = 0.55 # Using the value confirmed to work
        logger.debug(f"Local DB match found. Score: {score}, Threshold: {CONFIDENCE_THRESHOLD}")

        if score < CONFIDENCE_THRESHOLD:
            logger.info(f"Found local match with score {score}")
            return doc.metadata
        else:
            logger.warning(f"Low confidence match (score: {score}). Ignoring.")
            return {"error": "Drug not found in local database."}
    except Exception as e:
        logger.exception(f"Error during local DB search for '{drug_name}':") # Logs traceback
        return {"error": f"Error during local DB search: {e}"}

@tool
def query_fda_api(drug_name: str) -> dict:
    """
    Queries the openFDA API for a drug and returns its active and inactive ingredients.
    """
    logger.info(f"--- TOOL: Querying FDA API for '{drug_name}' ---")
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            active = result.get("active_ingredient", ["Not specified"])
            inactive = result.get("inactive_ingredient", ["Not specified"])

            # Check if *meaningful* data was returned
            if active == ["Not specified"] and inactive == ["Not specified"]:
                 logger.warning(f"Found '{drug_name}' in FDA API, but no specific ingredients listed.")
                 return {"error": "Ingredients not specified in FDA data."}

            formatted_data = {
                "brand_name": result.get("openfda", {}).get("brand_name", [drug_name])[0],
                # Filter out "Not specified" and potential empty strings after stripping
                "active_ingredients": [a.strip() for a in active if a.strip() and a != "Not specified"],
                "inactive_ingredients": [i.strip() for i in inactive if i.strip() and i != "Not specified"]
            }
            # Ensure lists aren't empty after filtering, if so use "Not specified"
            if not formatted_data["active_ingredients"]: formatted_data["active_ingredients"] = ["Not specified"]
            if not formatted_data["inactive_ingredients"]: formatted_data["inactive_ingredients"] = ["Not specified"]

            logger.info(f"Found match for '{drug_name}' in FDA API.")
            return formatted_data
        else:
            logger.info(f"No match found for '{drug_name}' in FDA API.")
            return {"error": "Drug not found in FDA database."}

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
    """
    Performs a web search using the Tavily Search API and returns results as a JSON string.
    """
    logger.info(f"--- TOOL: Performing web search for '{query}' ---")
    try:
        search = TavilySearchAPIWrapper()
        # Use .results() method which returns a list of dicts
        results_list = search.results(query=query, max_results=3) # Specify max results here
        logger.debug(f"Tavily search returned {len(results_list)} results.")
        # Return the list of results formatted as a JSON string
        return json.dumps(results_list, indent=2)
    except Exception as e:
        logger.exception(f"Tavily web search failed for query '{query}':") # Logs traceback
        # Return error as JSON string so downstream LLM knows it failed
        return json.dumps([{"error": f"Web search failed: {e}"}])