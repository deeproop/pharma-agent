# app.py
import streamlit as st
from graph import app # Import the compiled LangGraph app
import logging     # Import logging
import os          # Import os
import sys         # Import sys for StreamHandler

# --- Configure Logging (Revised Method) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR) # Create logs directory if it doesn't exist

LOG_FILE = os.path.join(LOG_DIR, "agent_log.log")

# Get a specific logger for your application
logger = logging.getLogger('pharma_agent')
logger.setLevel(logging.INFO) # Set the minimum level for this logger

# Prevent adding handlers multiple times if Streamlit re-runs the script
if not logger.handlers:
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Create File Handler
    fh = logging.FileHandler(LOG_FILE, mode='a') # Append mode
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create Stream Handler (to print to terminal)
    sh = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("--- Logger Initialized ---")
# -------------------------

# --- Streamlit UI Code ---
st.set_page_config(page_title="Pharma Ingredient Checker", layout="wide")
st.title("🧪 Pharma: Drug Ingredient Checker Agent")

st.markdown("""
Enter the name of a medication. The agent will check a local database (RAG), then query the openFDA API.
If found, it will classify the ingredients and suggest alternatives. Check `logs/agent_log.log` for detailed logs.
""")

# --- Test Buttons ---
st.subheader("Quick Tests (from local DB)")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Tylenol"):
        st.session_state.drug_name = "Tylenol"
with col2:
    if st.button("Advil"):
        st.session_state.drug_name = "Advil"
with col3:
    if st.button("Lipitor"): # Now found locally after threshold change
        st.session_state.drug_name = "Lipitor"

st.subheader("Other Tests")
col4, col5, col6 = st.columns(3)
with col4:
    if st.button("Paracetamol"): # Test normalization
        st.session_state.drug_name = "Paracetamol"
with col5:
    if st.button("Gleevec"): # Test FDA
        st.session_state.drug_name = "Gleevec"
with col6:
    if st.button("FakeDrug123"): # Test not found
        st.session_state.drug_name = "FakeDrug123"

# --- User Input ---
user_input = st.text_input("Or, enter any drug name:", key="drug_name")

# --- Agent Invocation & Display ---
if user_input:
    logger.info(f"User searched for: '{user_input}'")
    with st.spinner(f"Searching for '{user_input}'... (Check terminal/log file for details)"):

        # Input still only requires the initial drug name
        inputs = {"drug_name": user_input}

        # Run the graph
        try:
            final_state = app.invoke(inputs)
            logger.info(f"Graph execution finished for '{user_input}'. Final State: {final_state}")

            st.divider()

            # Display the results
            if final_state.get("error"):
                st.error(final_state["error"])
                logger.error(f"Final error state for '{user_input}': {final_state['error']}")
            elif final_state.get("data"):
                data = final_state["data"]
                st.subheader(f"Results for: {data.get('brand_name', user_input)}")

                col_info, col_analysis = st.columns(2)

                with col_info:
                    st.markdown("#### ✅ Active Ingredients")
                    st.json(data.get("active_ingredients", "Not specified"))

                    st.markdown("#### ⚪ Inactive Ingredients")
                    st.json(data.get("inactive_ingredients", "Not specified"))

                with col_analysis:
                    st.markdown("#### 🔬 Ingredient Analysis")
                    classification = final_state.get("classification")
                    if classification and "error" not in classification:
                        st.json(classification)
                    elif classification:
                        st.warning(f"Analysis Issue: {classification['error']}") # Use warning for non-critical analysis errors
                        logger.warning(f"Ingredient analysis issue for '{user_input}': {classification['error']}")
                    else:
                        st.info("Analysis was not performed or failed silently.")
                        logger.warning(f"Ingredient analysis missing for '{user_input}'.")


                    st.markdown("#### 🔄 Suggested Alternatives (from Web Search)")
                    alternatives = final_state.get("alternatives")
                    if alternatives:
                        for alt in alternatives:
                            st.success(f"• {alt}")
                    else:
                        st.info("No alternatives found via web search.")
                        logger.info(f"No alternatives found for '{user_input}'.")
            else:
                st.error("An unknown error occurred during processing.")
                logger.error(f"Unknown error state reached for '{user_input}'. Final State: {final_state}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception(f"Unhandled exception during graph invocation for '{user_input}':") # Logs the full traceback