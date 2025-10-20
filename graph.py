# graph.py
import os
import re
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# --- Import only the tools we are using ---
from tools import check_local_database, query_fda_api, web_search

load_dotenv()

# --- Synonym Map ---
NAME_MAP = {
    "paracetamol": "Acetaminophen", # UK/Int name -> US name
    "salbutamol": "Albuterol",      # UK/Int name -> US name
    # Add more synonyms as needed
    "tylenol": "Acetaminophen",     # Common Brand -> Generic
    "advil": "Ibuprofen",           # Common Brand -> Generic
    "motrin": "Ibuprofen",          # Common Brand -> Generic
    "benadryl": "Diphenhydramine",  # Common Brand -> Generic
    "lipitor": "Atorvastatin",      # Common Brand -> Generic
    "glucotrol xl": "Glipizide",    # Common Brand -> Generic
}
# -------------------------

# --- State Definition ---
class GraphState(TypedDict):
    drug_name: str
    original_drug_name: Optional[str] # Keep track of original input
    data: Optional[dict]
    classification: Optional[dict]
    alternatives: Optional[List[str]]
    error: Optional[str]

# --- Pydantic Models for Output Parsing ---
class IngredientAnalysis(BaseModel):
    therapeutic_category: str = Field(description="The primary therapeutic category of the active ingredient.")
    common_allergens: List[str] = Field(description="A list of common allergens found in the inactive ingredients (e.g., 'corn starch', 'lactose', 'soy'). List 'None' if no common ones are found.")

class AlternativeDrugs(BaseModel):
    alternatives: List[str] = Field(description="A list of 3-5 alternative brand names.")

# --- LLM Chain Setups ---
def get_classification_chain():
    """Sets up the LLM chain for ingredient classification."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        # Removed temperature=0 as it caused errors
    )
    parser = PydanticOutputParser(pydantic_object=IngredientAnalysis)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful pharmacist assistant. Analyze the provided drug ingredients using the web search context. Respond ONLY with the requested JSON format.\n{format_instructions}"),
            ("user", "Please analyze the following drug:\n\nWEB SEARCH CONTEXT:\n{context}\n\nDRUG INFO:\nActive Ingredients: {active}\nInactive Ingredients: {inactive}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_alternatives_chain():
    """Sets up the LLM chain for extracting alternative drug names."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        # Removed temperature=0 as it caused errors
    )
    parser = PydanticOutputParser(pydantic_object=AlternativeDrugs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful research assistant. Based on the provided web search results, extract a list of alternative drug brand names. Respond ONLY with the requested JSON format.\n{format_instructions}"),
            ("user", "Original Drug: {original_brand_name}\nActive Ingredient: {active_ingredient}\n\nWEB SEARCH RESULTS:\n{context}\n\nPlease list 3-5 alternative brand names. Do not include the original drug in the list.")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

# Initialize chains globally
classification_chain = get_classification_chain()
alternatives_chain = get_alternatives_chain()

# --- NODES ---

def normalize_name(state: GraphState) -> GraphState:
    """Checks the input drug name against a synonym map and normalizes it."""
    print("--- NODE: Normalizing drug name ---")
    original_name = state['drug_name']
    # Use lower case for map lookup, default to original name if not found
    normalized_name = NAME_MAP.get(original_name.lower(), original_name)
    if normalized_name != original_name:
        print(f"--- Normalizing '{original_name}' to '{normalized_name}' ---")
    # Always set both drug_name (potentially normalized) and original_drug_name
    return {"drug_name": normalized_name, "original_drug_name": original_name}

def fetch_from_local(state: GraphState) -> GraphState:
    """Node that calls the local RAG tool."""
    print("--- NODE: Checking local RAG database ---")
    drug_name = state['drug_name'] # Use normalized name
    result = check_local_database.invoke(drug_name)
    if "error" in result:
        print("--- NODE: Not found locally. ---")
        return {"data": None, "error": None} # Clear potential old data/error
    else:
        print("--- NODE: Found locally. ---")
        return {"data": result, "error": None}

def fetch_from_api(state: GraphState) -> GraphState:
    """Node that calls the external FDA API tool."""
    print("--- NODE: Checking external FDA API ---")
    drug_name = state['drug_name'] # Use normalized name
    result = query_fda_api.invoke(drug_name)
    if "error" in result:
        print(f"--- NODE: Not found or error in FDA API: {result['error']} ---")
        # Set final error state if FDA fails
        return {"data": None, "error": result["error"]}
    else:
        print("--- NODE: Found in FDA API. ---")
        return {"data": result, "error": None}

def handle_not_found(state: GraphState) -> GraphState:
    """Node that formats the final "not found" message."""
    print("--- NODE: Handling 'Not Found' after checking local DB and FDA API ---")
    original_name = state.get("original_drug_name", state['drug_name']) # Use original name in message
    last_error = state.get("error", "details unavailable")
    # Updated message to reflect sources checked
    return {
        "data": None,
        "error": f"Sorry, the drug '{original_name}' could not be found with ingredients in local DB or FDA database. Last error: {last_error}"
    }

def classify_ingredients(state: GraphState) -> GraphState:
    """Node that calls web search and an LLM to classify ingredients."""
    print("--- NODE: Classifying ingredients (with web search) ---")
    if not state.get("data"): return {}
    try:
        active_list = state["data"].get("active_ingredients", [])
        inactive_list = state["data"].get("inactive_ingredients", [])
        # Check if the active list is valid for analysis
        if not active_list or active_list == ["Not specified"]:
             return {"classification": {"error": "No valid active ingredient found to analyze."}}

        active_ingredient = active_list[0]
        # Clean up ingredient name slightly for search
        clean_active_ingredient = re.split(r'(\d| HCl)', active_ingredient, 1)[0].strip()

        search_query = f"therapeutic category and common allergens for {clean_active_ingredient}"
        search_context = web_search.invoke(search_query)

        response = classification_chain.invoke({
            "context": search_context,
            "active": ", ".join(active_list),
            "inactive": ", ".join(inactive_list)
        })
        return {"classification": response.dict()}
    except Exception as e:
        print(f"--- NODE: Classification failed: {e} ---")
        # Provide more specific error messages if possible
        error_msg = f"Failed to analyze ingredients. Error: {e}"
        if "AuthenticationError" in str(e):
            error_msg = "Failed to analyze ingredients due to Azure authentication error. Check API key/endpoint."
        elif "RateLimitError" in str(e):
             error_msg = "Failed to analyze ingredients due to Azure rate limits."
        elif "InvalidRequestError" in str(e): # Include details for invalid requests
             error_msg = f"Failed to analyze ingredients. Azure request error: {e}"

        return {"classification": {"error": error_msg}}

def find_alternatives(state: GraphState) -> GraphState:
    """Node that calls web search and an LLM to find alternative drugs."""
    print("--- NODE: Finding alternatives (with web search) ---")
    if not state.get("data"): return {}
    try:
        original_brand_name = state["data"].get("brand_name", state.get("original_drug_name", "Unknown"))
        active_ingredient_list = state["data"].get("active_ingredients", [])
        # Check if the active list is valid for finding alternatives
        if not active_ingredient_list or active_ingredient_list == ["Not specified"]:
            return {"alternatives": []} # Return empty list if no valid active ingredient

        active_ingredient = active_ingredient_list[0]
        # Clean up ingredient name slightly for search
        clean_active_ingredient = re.split(r'(\d| HCl)', active_ingredient, 1)[0].strip()

        search_query = f"alternative brand names for {clean_active_ingredient}"
        search_context = web_search.invoke(search_query)

        response = alternatives_chain.invoke({
            "original_brand_name": original_brand_name,
            "active_ingredient": clean_active_ingredient,
            "context": search_context
        })

        # Filter out the original name if the LLM includes it
        filtered_alts = [alt for alt in response.alternatives if alt.lower() != original_brand_name.lower()]
        return {"alternatives": filtered_alts}

    except Exception as e:
        print(f"--- NODE: Finding alternatives failed: {e} ---")
        return {"alternatives": []} # Return empty list on failure


# --- CONDITIONAL EDGES ---

def should_query_api(state: GraphState) -> str:
    """Conditional edge: If data found locally, process. Else, query FDA API."""
    print("--- CONDITIONAL EDGE: should_query_api ---")
    if state["data"]:
        print("--- DECISION: Data found locally, processing results. ---")
        return "process_results"
    else:
        print("--- DECISION: No data found, querying FDA API. ---")
        return "query_fda_api"

def check_api_results(state: GraphState) -> str:
    """Conditional edge: If FDA API succeeded, process. If failed, handle error."""
    print("--- CONDITIONAL EDGE: check_fda_api_results ---")
    data = state.get("data")
    # Check if data exists AND has *some* meaningful ingredient information
    has_active = data and data.get("active_ingredients") and data.get("active_ingredients") != ["Not specified"]
    has_inactive = data and data.get("inactive_ingredients") and data.get("inactive_ingredients") != ["Not specified"]

    if data and (has_active or has_inactive):
        print("--- DECISION: FDA API returned data, processing results. ---")
        return "process_results"
    else:
        print("--- DECISION: FDA API failed or returned no ingredients, handling final error. ---")
        return "handle_error" # Go directly to handle_not_found


# --- ASSEMBLE THE GRAPH ---
def create_graph():
    """Creates and compiles the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    # Add all the nodes
    workflow.add_node("normalize_name", normalize_name)
    workflow.add_node("check_local_rag", fetch_from_local)
    workflow.add_node("query_fda_api", fetch_from_api)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("classify_ingredients", classify_ingredients)
    workflow.add_node("find_alternatives", find_alternatives)

    # Set the entry point
    workflow.set_entry_point("normalize_name")

    # Define the edges
    workflow.add_edge("normalize_name", "check_local_rag")

    workflow.add_conditional_edges(
        "check_local_rag",
        should_query_api,
        {
            "process_results": "classify_ingredients",
            "query_fda_api": "query_fda_api"
        }
    )

    workflow.add_conditional_edges(
        "query_fda_api",
        check_api_results,
        {
            "process_results": "classify_ingredients",
            "handle_error": "handle_not_found"
        }
    )

    # Sequential processing edges
    workflow.add_edge("classify_ingredients", "find_alternatives")

    # Edges to the end
    workflow.add_edge("find_alternatives", END)
    workflow.add_edge("handle_not_found", END)

    # Compile the graph
    app = workflow.compile()
    print("--- Graph compiled successfully (with Name Normalization) ---")
    return app

# Expose the compiled app for the Streamlit UI
app = create_graph()