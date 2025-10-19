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

# --- Import all three data source tools ---
from tools import check_local_database, query_fda_api, query_dailymed_api, web_search

load_dotenv()

# (GraphState, IngredientAnalysis, AlternativeDrugs, get_classification_chain, get_alternatives_chain remain the same)
# ...
class GraphState(TypedDict):
    drug_name: str
    data: Optional[dict]
    classification: Optional[dict]
    alternatives: Optional[List[str]]
    error: Optional[str]

class IngredientAnalysis(BaseModel):
    therapeutic_category: str = Field(description="The primary therapeutic category of the active ingredient.")
    common_allergens: List[str] = Field(description="A list of common allergens found in the inactive ingredients (e.g., 'corn starch', 'lactose', 'soy'). List 'None' if no common ones are found.")

class AlternativeDrugs(BaseModel):
    alternatives: List[str] = Field(description="A list of 3-5 alternative brand names.")

def get_classification_chain():
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
# ...

# 2. Define the nodes

# (fetch_from_local, fetch_from_api are UNCHANGED)
def fetch_from_local(state: GraphState) -> GraphState:
    print("--- NODE: Checking local RAG database ---")
    drug_name = state['drug_name']
    result = check_local_database.invoke(drug_name)
    if "error" in result:
        print("--- NODE: Not found locally. ---")
        # Clear potential old data/error before next step
        return {"data": None, "error": None} 
    else:
        print("--- NODE: Found locally. ---")
        return {"data": result, "error": None}

def fetch_from_api(state: GraphState) -> GraphState:
    print("--- NODE: Checking external FDA API ---")
    drug_name = state['drug_name']
    result = query_fda_api.invoke(drug_name)
    if "error" in result:
        print(f"--- NODE: Not found or error in FDA API: {result['error']} ---")
         # Clear potential old data before next step, keep error to signal failure
        return {"data": None, "error": result["error"]}
    else:
        print("--- NODE: Found in FDA API. ---")
        return {"data": result, "error": None}

# --- NEW NODE for DailyMed ---
def fetch_from_dailymed(state: GraphState) -> GraphState:
    """
    Node that calls the DailyMed API tool as a fallback.
    """
    print("--- NODE: Checking DailyMed API ---")
    drug_name = state['drug_name']
    result = query_dailymed_api.invoke(drug_name)
    
    if "error" in result:
        print(f"--- NODE: Not found or error in DailyMed API: {result['error']} ---")
        # Ensure data is cleared, set final error state
        return {"data": None, "error": result["error"]} 
    else:
        print("--- NODE: Found in DailyMed API. ---")
         # Clear any previous error state from FDA failure
        return {"data": result, "error": None}

# --- UPDATED handle_not_found node ---
def handle_not_found(state: GraphState) -> GraphState:
    """
    Node that formats the final "not found" message after checking all sources.
    """
    print("--- NODE: Handling 'Not Found' after checking all sources ---")
    # Include the last error if available
    last_error = state.get("error", "details unavailable")
    return {
        "data": None,
        "error": f"Sorry, the drug '{state['drug_name']}' could not be found with ingredients in local DB, FDA, or DailyMed databases. Last error: {last_error}"
    }

# (classify_ingredients, find_alternatives remain the same)
def classify_ingredients(state: GraphState) -> GraphState:
    print("--- NODE: Classifying ingredients (with web search) ---")
    if not state.get("data"): return {}
    try:
        active_list = state["data"].get("active_ingredients", [])
        inactive_list = state["data"].get("inactive_ingredients", [])
        if not active_list or active_list == ["Not specified"]: 
             return {"classification": {"error": "No valid active ingredient found to analyze."}}
        active_ingredient = active_list[0]
        search_query = f"therapeutic category and common allergens for {active_ingredient}"
        search_context = web_search.invoke(search_query)
        response = classification_chain.invoke({
            "context": search_context,
            "active": ", ".join(active_list),
            "inactive": ", ".join(inactive_list)
        })
        return {"classification": response.dict()}
    except Exception as e:
        print(f"--- NODE: Classification failed: {e} ---")
        return {"classification": {"error": "Failed to analyze ingredients. Check Azure Chat Deployment."}}

def find_alternatives(state: GraphState) -> GraphState:
    print("--- NODE: Finding alternatives (with web search) ---")
    if not state.get("data"): return {}
    try:
        original_brand_name = state["data"].get("brand_name")
        active_ingredient_list = state["data"].get("active_ingredients", [])
        if not active_ingredient_list or active_ingredient_list == ["Not specified"]:
            return {"alternatives": []}
        active_ingredient = re.split(r'(\d| HCl)', active_ingredient_list[0], 1)[0].strip()
        search_query = f"alternative brand names for {active_ingredient}"
        search_context = web_search.invoke(search_query)
        response = alternatives_chain.invoke({
            "original_brand_name": original_brand_name,
            "active_ingredient": active_ingredient,
            "context": search_context
        })
        # Filter out the original name if the LLM includes it
        filtered_alts = [alt for alt in response.alternatives if alt.lower() != original_brand_name.lower()]
        return {"alternatives": filtered_alts}
    except Exception as e:
        print(f"--- NODE: Finding alternatives failed: {e} ---")
        return {"alternatives": []}

# 3. Define the conditional edges

# (should_query_api remains the same)
def should_query_api(state: GraphState) -> str:
    print("--- CONDITIONAL EDGE: should_query_api ---")
    if state["data"]:
        print("--- DECISION: Data found locally, processing results. ---")
        return "process_results" 
    else:
        print("--- DECISION: No data found, querying FDA API. ---")
        return "query_fda_api" # <-- Renamed target slightly for clarity

# --- UPDATED check_api_results edge ---
def check_api_results(state: GraphState) -> str:
    """
    Conditional edge: If FDA API succeeded, process. If failed, try DailyMed.
    """
    print("--- CONDITIONAL EDGE: check_fda_api_results ---")
    # Success is defined as having data AND that data not being just placeholders
    data = state.get("data")
    has_active = data and data.get("active_ingredients") and data.get("active_ingredients") != ["Not specified"]
    has_inactive = data and data.get("inactive_ingredients") and data.get("inactive_ingredients") != ["Not specified"]

    if data and (has_active or has_inactive):
        print("--- DECISION: FDA API returned data, processing results. ---")
        return "process_results"
    else:
        print("--- DECISION: FDA API failed or returned no ingredients, trying DailyMed. ---")
        # Clear error state from FDA before trying DailyMed
        state["error"] = None 
        return "query_dailymed_api" # <-- Go to the new node

# --- NEW check_dailymed_results edge ---
def check_dailymed_results(state: GraphState) -> str:
    """
    Conditional edge: If DailyMed succeeded, process. If failed, handle error.
    """
    print("--- CONDITIONAL EDGE: check_dailymed_results ---")
    data = state.get("data")
    has_active = data and data.get("active_ingredients") and data.get("active_ingredients") != ["Not specified"]
    has_inactive = data and data.get("inactive_ingredients") and data.get("inactive_ingredients") != ["Not specified"]

    if data and (has_active or has_inactive):
        print("--- DECISION: DailyMed API returned data, processing results. ---")
        return "process_results"
    else:
        print("--- DECISION: DailyMed API failed or returned no ingredients, handling final error. ---")
        return "handle_error"


# 4. Assemble the graph
def create_graph():
    workflow = StateGraph(GraphState)

    # Add the nodes (original + new DailyMed node)
    workflow.add_node("check_local_rag", fetch_from_local)
    workflow.add_node("query_fda_api", fetch_from_api)
    workflow.add_node("query_dailymed_api", fetch_from_dailymed) # <-- NEW
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("classify_ingredients", classify_ingredients) 
    workflow.add_node("find_alternatives", find_alternatives)       

    workflow.set_entry_point("check_local_rag")

    # Edge from local check
    workflow.add_conditional_edges(
        "check_local_rag",
        should_query_api,
        {
            "process_results": "classify_ingredients", 
            "query_fda_api": "query_fda_api" # <-- Use specific node name
        }
    )
    
    # --- UPDATED Edge from FDA check ---
    workflow.add_conditional_edges(
        "query_fda_api", # Source node is FDA API call
        check_api_results, # Decision function
        {
            "process_results": "classify_ingredients", # Success -> process
            "query_dailymed_api": "query_dailymed_api"  # Failure -> try DailyMed
        }
    )

    # --- NEW Edge from DailyMed check ---
    workflow.add_conditional_edges(
        "query_dailymed_api", # Source node is DailyMed API call
        check_dailymed_results, # Decision function
        {
            "process_results": "classify_ingredients", # Success -> process
            "handle_error": "handle_not_found"         # Failure -> handle final error
        }
    )

    # Sequential edges for processing and final error handling
    workflow.add_edge("classify_ingredients", "find_alternatives")
    workflow.add_edge("find_alternatives", END)
    workflow.add_edge("handle_not_found", END)

    app = workflow.compile()
    print("--- Graph compiled successfully with DailyMed fallback! ---")
    return app

app = create_graph()