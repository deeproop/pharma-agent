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
import logging # Import logging

# Get the specific logger instance
logger = logging.getLogger('pharma_agent')

# --- Import only the tools we are using ---
from tools import check_local_database, query_fda_api, web_search

load_dotenv()
logger.info("Graph dependencies loaded.")

# --- Synonym Map ---
NAME_MAP = {
    "paracetamol": "Acetaminophen",
    "salbutamol": "Albuterol",
    "tylenol": "Acetaminophen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "benadryl": "Diphenhydramine",
    "lipitor": "Atorvastatin",
    "glucotrol xl": "Glipizide",
}

# --- State Definition ---
class GraphState(TypedDict):
    drug_name: str
    original_drug_name: Optional[str]
    data: Optional[dict]
    classification: Optional[dict]
    alternatives: Optional[List[str]]
    error: Optional[str]

# --- Pydantic Models for Output Parsing ---
class IngredientAnalysis(BaseModel):
    therapeutic_category: str = Field(description="The primary therapeutic category of the active ingredient.")
    common_allergens: List[str] = Field(description="A list of common allergens found in the inactive ingredients. List 'None' if no common ones are found.")

class AlternativeDrugs(BaseModel):
    alternatives: List[str] = Field(description="A list of 3-5 alternative brand names.")

# --- LLM Chain Setups ---
def get_classification_chain():
    logger.debug("Setting up classification chain.")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
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
    logger.debug("Setting up alternatives chain.")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    parser = PydanticOutputParser(pydantic_object=AlternativeDrugs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful research assistant. Based on the provided web search results, extract a list of alternative drug brand names. Respond ONLY with the requested JSON format.\n{format_instructions}"),
            ("user", "Original Drug: {original_brand_name}\nActive Ingredient: {active_ingredient}\n\nWEB SEARCH RESULTS:\n{context}\n\nPlease list 3-5 alternative brand names. Do not include the original drug in the list.")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

classification_chain = get_classification_chain()
alternatives_chain = get_alternatives_chain()
logger.info("LLM Chains initialized.")

# --- NODES ---

def normalize_name(state: GraphState) -> GraphState:
    logger.info("--- NODE: Normalizing drug name ---")
    original_name = state['drug_name']
    normalized_name = NAME_MAP.get(original_name.lower(), original_name)
    if normalized_name != original_name:
        logger.info(f"Normalized '{original_name}' to '{normalized_name}'")
    return {"drug_name": normalized_name, "original_drug_name": original_name}

def fetch_from_local(state: GraphState) -> GraphState:
    logger.info("--- NODE: Checking local RAG database ---")
    drug_name = state['drug_name']
    result = check_local_database.invoke(drug_name)
    if "error" in result:
        logger.info("Drug not found locally.")
        return {"data": None, "error": None}
    else:
        logger.info("Drug found locally.")
        return {"data": result, "error": None}

def fetch_from_api(state: GraphState) -> GraphState:
    logger.info("--- NODE: Checking external FDA API ---")
    drug_name = state['drug_name']
    result = query_fda_api.invoke(drug_name)
    if "error" in result:
        logger.warning(f"FDA API check failed or returned no ingredients: {result['error']}")
        return {"data": None, "error": result["error"]}
    else:
        logger.info("Drug found in FDA API.")
        return {"data": result, "error": None}

def handle_not_found(state: GraphState) -> GraphState:
    logger.warning("--- NODE: Handling 'Not Found' after checking sources ---")
    original_name = state.get("original_drug_name", state['drug_name'])
    last_error = state.get("error", "details unavailable")
    error_message = f"Sorry, the drug '{original_name}' could not be found with ingredients in local DB or FDA database. Last error: {last_error}"
    logger.error(error_message) # Log the final error
    return {"data": None, "error": error_message}

def classify_ingredients(state: GraphState) -> GraphState:
    logger.info("--- NODE: Classifying ingredients (with web search) ---")
    if not state.get("data"):
        logger.warning("No data found in state for classification.")
        return {}
    try:
        active_list = state["data"].get("active_ingredients", [])
        inactive_list = state["data"].get("inactive_ingredients", [])
        if not active_list or active_list == ["Not specified"]:
            logger.warning("No valid active ingredient found to analyze.")
            return {"classification": {"error": "No valid active ingredient found to analyze."}}

        active_ingredient = active_list[0]
        clean_active_ingredient = re.split(r'(\d| HCl)', active_ingredient, 1)[0].strip()
        search_query = f"therapeutic category and common allergens for {clean_active_ingredient}"
        logger.debug(f"Classification web search query: {search_query}")
        search_context = web_search.invoke(search_query)

        response = classification_chain.invoke({
            "context": search_context,
            "active": ", ".join(active_list),
            "inactive": ", ".join(inactive_list)
        })
        logger.info("Successfully classified ingredients.")
        return {"classification": response.dict()}
    except Exception as e:
        logger.exception("Classification failed:") # Use logger.exception to include traceback
        error_msg = f"Failed to analyze ingredients. Error: {e}"
        if "AuthenticationError" in str(e): error_msg = "Failed to analyze ingredients due to Azure authentication error."
        elif "RateLimitError" in str(e): error_msg = "Failed to analyze ingredients due to Azure rate limits."
        elif "InvalidRequestError" in str(e): error_msg = f"Failed to analyze ingredients. Azure request error: {e}"
        return {"classification": {"error": error_msg}}

def find_alternatives(state: GraphState) -> GraphState:
    logger.info("--- NODE: Finding alternatives (with web search) ---")
    if not state.get("data"):
        logger.warning("No data found in state for finding alternatives.")
        return {}
    try:
        original_brand_name = state["data"].get("brand_name", state.get("original_drug_name", "Unknown"))
        active_ingredient_list = state["data"].get("active_ingredients", [])
        if not active_ingredient_list or active_ingredient_list == ["Not specified"]:
            logger.info("No valid active ingredient found to search alternatives.")
            return {"alternatives": []}

        active_ingredient = active_ingredient_list[0]
        clean_active_ingredient = re.split(r'(\d| HCl)', active_ingredient, 1)[0].strip()
        search_query = f"alternative brand names for {clean_active_ingredient}"
        logger.debug(f"Alternatives web search query: {search_query}")
        search_context = web_search.invoke(search_query)

        response = alternatives_chain.invoke({
            "original_brand_name": original_brand_name,
            "active_ingredient": clean_active_ingredient,
            "context": search_context
        })

        filtered_alts = [alt for alt in response.alternatives if alt.lower() != original_brand_name.lower()]
        logger.info(f"Found alternatives: {filtered_alts}")
        return {"alternatives": filtered_alts}

    except Exception as e:
        logger.exception("Finding alternatives failed:") # Use logger.exception
        return {"alternatives": []}


# --- CONDITIONAL EDGES ---

def should_query_api(state: GraphState) -> str:
    logger.debug("--- CONDITIONAL EDGE: should_query_api ---")
    if state["data"]:
        logger.debug("Decision: Data found locally, processing results.")
        return "process_results"
    else:
        logger.debug("Decision: No data found, querying FDA API.")
        return "query_fda_api"

def check_api_results(state: GraphState) -> str:
    logger.debug("--- CONDITIONAL EDGE: check_fda_api_results ---")
    data = state.get("data")
    has_active = data and data.get("active_ingredients") and data.get("active_ingredients") != ["Not specified"]
    has_inactive = data and data.get("inactive_ingredients") and data.get("inactive_ingredients") != ["Not specified"]

    if data and (has_active or has_inactive):
        logger.debug("Decision: FDA API returned data, processing results.")
        return "process_results"
    else:
        logger.warning("Decision: FDA API failed or returned no ingredients, handling final error.")
        return "handle_error"


# --- ASSEMBLE THE GRAPH ---
def create_graph():
    logger.info("Creating LangGraph workflow...")
    workflow = StateGraph(GraphState)

    workflow.add_node("normalize_name", normalize_name)
    workflow.add_node("check_local_rag", fetch_from_local)
    workflow.add_node("query_fda_api", fetch_from_api)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("classify_ingredients", classify_ingredients)
    workflow.add_node("find_alternatives", find_alternatives)

    workflow.set_entry_point("normalize_name")

    workflow.add_edge("normalize_name", "check_local_rag")

    workflow.add_conditional_edges(
        "check_local_rag",
        should_query_api,
        {"process_results": "classify_ingredients", "query_fda_api": "query_fda_api"}
    )

    workflow.add_conditional_edges(
        "query_fda_api",
        check_api_results,
        {"process_results": "classify_ingredients", "handle_error": "handle_not_found"}
    )

    workflow.add_edge("classify_ingredients", "find_alternatives")
    workflow.add_edge("find_alternatives", END)
    workflow.add_edge("handle_not_found", END)

    app = workflow.compile()
    logger.info("--- Graph compiled successfully (with Name Normalization) ---")
    return app

app = create_graph()