# graph.py
import os
import re
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field       # <-- This line will now work
from langchain_core.output_parsers import PydanticOutputParser

# Import all our tools
from tools import check_local_database, query_fda_api, find_alternative_drugs

load_dotenv()

# 1. Define the state (with new fields)
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        drug_name: The name of the drug to search for.
        data: The structured data of the drug (active/inactive ingredients).
        classification: Analysis of the ingredients (stretch goal).
        alternatives: List of alternative drugs (stretch goal).
        error: An error message if the drug is not found.
    """
    drug_name: str
    data: Optional[dict]
    classification: Optional[dict]
    alternatives: Optional[List[str]]
    error: Optional[str]


# --- NEW: Pydantic model for classification (Stretch Goal 1) ---
class IngredientAnalysis(BaseModel):
    therapeutic_category: str = Field(description="The primary therapeutic category of the active ingredient.")
    common_allergens: List[str] = Field(description="A list of common allergens found in the inactive ingredients (e.g., 'corn starch', 'lactose', 'soy'). List 'None' if no common ones are found.")

# --- NEW: Setup for classification node ---
def get_classification_chain():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    
    parser = PydanticOutputParser(pydantic_object=IngredientAnalysis)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful pharmacist assistant. Analyze the provided drug ingredients. Respond ONLY with the requested JSON format.\n{format_instructions}"),
            ("user", "Please analyze the following drug:\nActive Ingredients: {active}\nInactive Ingredients: {inactive}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    return prompt | llm | parser

classification_chain = get_classification_chain()


# 2. Define the nodes
def fetch_from_local(state: GraphState) -> GraphState:
    # (This node's code is unchanged)
    print("--- NODE: Checking local RAG database ---")
    drug_name = state['drug_name']
    result = check_local_database.invoke(drug_name)
    
    if "error" in result:
        print("--- NODE: Not found locally. ---")
        return {"data": None, "error": None} 
    else:
        print("--- NODE: Found locally. ---")
        return {"data": result, "error": None}

def fetch_from_api(state: GraphState) -> GraphState:
    # (This node's code is unchanged)
    print("--- NODE: Checking external FDA API ---")
    drug_name = state['drug_name']
    result = query_fda_api.invoke(drug_name)
    
    if "error" in result:
        print("--- NODE: Not found in API. ---")
        return {"data": None, "error": result["error"]}
    else:
        print("--- NODE: Found in API. ---")
        return {"data": result, "error": None}

def handle_not_found(state: GraphState) -> GraphState:
    # (This node's code is unchanged)
    print("--- NODE: Handling 'Not Found' ---")
    return {
        "data": None,
        "error": f"Sorry, the drug '{state['drug_name']}' could not be found in local or external databases."
    }

# --- NEW: Node for Stretch Goal 1 ---
def classify_ingredients(state: GraphState) -> GraphState:
    """
    Node that calls an LLM to classify ingredients.
    """
    print("--- NODE: Classifying ingredients ---")
    if not state.get("data"):
        return {} # Should not happen if graph is correct

    try:
        active = state["data"].get("active_ingredients", [])
        inactive = state["data"].get("inactive_ingredients", [])
        
        response = classification_chain.invoke({
            "active": ", ".join(active),
            "inactive": ", ".join(inactive)
        })
        
        return {"classification": response.dict()}
    
    except Exception as e:
        print(f"--- NODE: Classification failed: {e} ---")
        return {"classification": {"error": "Failed to analyze ingredients."}}


# --- NEW: Node for Stretch Goal 2 ---
def find_alternatives(state: GraphState) -> GraphState:
    """
    Node that calls the tool to find alternative drugs.
    """
    print("--- NODE: Finding alternatives ---")
    if not state.get("data"):
        return {} # Should not happen

    try:
        original_brand_name = state["data"].get("brand_name")
        active_ingredient_list = state["data"].get("active_ingredients", [])
        
        if not active_ingredient_list:
            return {"alternatives": []}
            
        # Use the first active ingredient
        active_ingredient = active_ingredient_list[0]
        
        alts = find_alternative_drugs.invoke({
            "active_ingredient": active_ingredient,
            "original_brand_name": original_brand_name
        })
        
        return {"alternatives": alts}

    except Exception as e:
        print(f"--- NODE: Finding alternatives failed: {e} ---")
        return {"alternatives": []}


# 3. Define the conditional edges
def should_query_api(state: GraphState) -> str:
    # (This edge's logic is unchanged)
    print("--- CONDITIONAL EDGE: should_query_api ---")
    if state["data"]:
        print("--- DECISION: Data found locally, processing results. ---")
        return "process_results" # <-- CHANGED
    else:
        print("--- DECISION: No data found, querying API. ---")
        return "query_api"

def check_api_results(state: GraphState) -> str:
    # (This edge's logic is unchanged)
    print("--- CONDITIONAL EDGE: check_api_results ---")
    if state["error"]:
        print("--- DECISION: API returned an error, handling. ---")
        return "handle_error"
    else:
        print("--- DECISION: API returned data, processing results. ---")
        return "process_results" # <-- CHANGED


# 4. Assemble the graph
def create_graph():
    workflow = StateGraph(GraphState)

    # Add the nodes (original + new)
    workflow.add_node("check_local_rag", fetch_from_local)
    workflow.add_node("query_external_api", fetch_from_api)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("classify_ingredients", classify_ingredients) # <-- NEW
    workflow.add_node("find_alternatives", find_alternatives)       # <-- NEW

    # Set the entry point
    workflow.set_entry_point("check_local_rag")

    # Add the conditional edges
    workflow.add_conditional_edges(
        "check_local_rag",
        should_query_api,
        {
            "process_results": "classify_ingredients", # <-- CHANGED
            "query_api": "query_external_api"
        }
    )
    
    workflow.add_conditional_edges(
        "query_external_api",
        check_api_results,
        {
            "process_results": "classify_ingredients", # <-- CHANGED
            "handle_error": "handle_not_found"
        }
    )

    # Add the new sequential edges for stretch goals
    workflow.add_edge("classify_ingredients", "find_alternatives")
    workflow.add_edge("find_alternatives", END)

    # Add a final edge from the error handler to the end
    workflow.add_edge("handle_not_found", END)

    # Compile the graph
    app = workflow.compile()
    print("--- Graph compiled successfully with stretch goals! ---")
    return app

# Expose the compiled app for the Streamlit UI
app = create_graph()