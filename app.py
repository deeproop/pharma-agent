# app.py
import streamlit as st
from graph import app # Import the compiled LangGraph app
import logging
import os
import sys

# --- Configure Logging (Keep the existing logger setup) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "agent_log.log")
logger = logging.getLogger('pharma_agent')
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("--- Logger Initialized ---")
# -------------------------

# --- Streamlit UI Code ---
st.set_page_config(page_title="Pharma Ingredient Checker", layout="wide")
st.title("🧪 Pharma: Drug Ingredient Checker Agent")

st.markdown("""
Type a medication name into the search bar below and press **Enter** to search.
Your last few searches are available in the history dropdown to populate the search bar.
Check `logs/agent_log.log` for detailed logs.
""")

st.divider()

# --- Initialize Session State ---
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'history_selectbox_value' not in st.session_state:
     st.session_state.history_selectbox_value = "Select from history..."

# --- Search History Dropdown ---
history_options = ["Select from history..."] + st.session_state.search_history

def handle_history_selection():
    selected_value = st.session_state.history_selectbox_widget
    if selected_value and selected_value != history_options[0]:
        if st.session_state.search_term != selected_value:
             logger.info(f"History selected: '{selected_value}'. Updating search bar.")
             st.session_state.search_term = selected_value
             st.session_state.history_selectbox_value = history_options[0]

st.selectbox(
    "Search History:",
    options=history_options,
    key='history_selectbox_widget',
    on_change=handle_history_selection,
)

# --- Main Search Bar ---
user_input = st.text_input(
    "**Search for a drug name and press Enter:**",
    key="search_term",
    placeholder="e.g., Aspirin"
)

# --- Agent Invocation & Display ---
if user_input:
    logger.info(f"Processing search for: '{user_input}'")

    # --- Update Search History (Keep this logic) ---
    current_history = st.session_state.search_history
    if not current_history or user_input != current_history[0]:
        if user_input in current_history:
            current_history.remove(user_input)
        current_history.insert(0, user_input)
        st.session_state.search_history = current_history[:3]
        logger.info(f"Updated search history: {st.session_state.search_history}")
        st.session_state.history_selectbox_value = history_options[0]
    # --- End History Update ---

    with st.spinner(f"Searching for '{user_input}'... (Check terminal/log file for details)"):
        inputs = {"drug_name": user_input}
        try:
            final_state = app.invoke(inputs)
            logger.info(f"Graph execution finished for '{user_input}'. Final State keys: {final_state.keys()}")

            st.divider()

            # --- Display the results ---
            if final_state.get("error"):
                st.error(final_state["error"])
                logger.error(f"Final error state for '{user_input}': {final_state['error']}")
            elif final_state.get("data"):
                data = final_state["data"]
                display_name = final_state.get('original_drug_name', user_input)
                st.subheader(f"Results for: {display_name.capitalize()}")

                col_info, col_analysis = st.columns([2, 3])

                with col_info:
                    # Active Ingredients Display
                    st.markdown("#### ✅ Active Ingredients")
                    active_ing = data.get("active_ingredients", ["Not specified"])
                    if active_ing and active_ing != ["Not specified"]:
                        for ingredient in active_ing:
                            st.markdown(f"- {ingredient}")
                    else:
                        st.info("No active ingredients specified.")
                    st.write("")

                    # Inactive Ingredients Display (as bullet points)
                    st.markdown("#### ⚪ Inactive Ingredients")
                    inactive_ing = data.get("inactive_ingredients", ["Not specified"])
                    if inactive_ing and inactive_ing != ["Not specified"]:
                         for ingredient in inactive_ing:
                             st.markdown(f"- {ingredient}")
                    else:
                        st.info("No inactive ingredients specified.")

                with col_analysis:
                    # Ingredient Analysis Display
                    st.markdown("#### 🔬 Ingredient Analysis")
                    classification = final_state.get("classification")
                    if classification and "error" not in classification:
                        category = classification.get('therapeutic_category', 'N/A')
                        allergens = classification.get('common_allergens', [])
                        st.markdown(f"**Therapeutic Category:** {category}")
                        st.write("")
                        st.markdown("**Common Allergens Found:**")
                        if allergens and allergens != ["None"]:
                            for allergen in allergens:
                                st.markdown(f"- {allergen}")
                        else:
                            st.info("None specified or detected.")
                    elif classification and "error" in classification:
                        st.warning(f"Analysis Issue: {classification['error']}")
                        logger.warning(f"Ingredient analysis issue for '{user_input}': {classification['error']}")
                    else:
                        st.info("Analysis was not performed or failed.")
                        logger.warning(f"Ingredient analysis missing for '{user_input}'.")
                    st.write("")

                    # --- Suggested Alternatives Display (REVERTED TO st.success) ---
                    st.markdown("#### 🔄 Suggested Alternatives (from Web Search)")
                    alternatives = final_state.get("alternatives")
                    if alternatives:
                        for alt in alternatives:
                            st.success(f"• {alt}") # Use st.success again
                    else:
                        st.info("No alternatives found via web search.")
                        logger.info(f"No alternatives found for '{user_input}'.")
            else:
                st.error("An unknown error occurred during processing.")
                logger.error(f"Unknown error state reached for '{user_input}'. Final State: {final_state}")
            # --- End Display Logic ---

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception(f"Unhandled exception during graph invocation for '{user_input}':")