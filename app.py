import streamlit as st
from graph import app # Import the compiled LangGraph app

st.set_page_config(page_title="Pharma Ingredient Checker", layout="wide")
st.title("🧪 Pharma: Drug Ingredient Checker Agent")

st.markdown("""
Enter the name of a medication. The agent will check a local database (RAG), then query the openFDA API.
If found, it will **classify the ingredients** and **suggest alternatives**.
""")

# (test buttons section)
st.subheader("Quick Tests (from local DB)")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Tylenol"):
        st.session_state.drug_name = "Tylenol"
with col2:
    if st.button("Advil"):
        st.session_state.drug_name = "Advil"
with col3:
    if st.button("Benadryl"):
        st.session_state.drug_name = "Benadryl"

st.subheader("External API Tests")
col4, col5, col6 = st.columns(3)
with col4:
    if st.button("Lipitor"): 
        st.session_state.drug_name = "Lipitor"
with col5:
    if st.button("Gleevec"): 
        st.session_state.drug_name = "Gleevec"
with col6:
    if st.button("FakeDrug123"):
        st.session_state.drug_name = "FakeDrug123"


user_input = st.text_input("Or, enter any drug name:", key="drug_name")

if user_input:
    with st.spinner(f"Searching for '{user_input}'... (Check terminal for agent logs)"):
        
        inputs = {"drug_name": user_input}
        
        # Run the graph
        final_state = app.invoke(inputs)
        
        st.divider()
        
        # Display the results
        if final_state.get("error"):
            st.error(final_state["error"])
        elif final_state.get("data"):
            data = final_state["data"]
            st.subheader(f"Results for: {data.get('brand_name', user_input)}")
            
            # --- UPDATED TO USE COLUMNS ---
            col_info, col_analysis = st.columns(2)
            
            with col_info:
                st.markdown("#### ✅ Active Ingredients")
                st.json(data.get("active_ingredients", "Not specified"))
                
                st.markdown("#### ⚪ Inactive Ingredients")
                st.json(data.get("inactive_ingredients", "Not specified"))
            
            with col_analysis:
                # --- Display Classification ---
                st.markdown("#### 🔬 Ingredient Analysis")
                classification = final_state.get("classification")
                if classification and "error" not in classification:
                    st.json(classification)
                elif classification:
                    st.error(classification["error"])
                else:
                    st.info("Analysis was not performed.")
                
                # --- Display Alternatives ---
                st.markdown("#### 🔄 Suggested Alternatives")
                alternatives = final_state.get("alternatives")
                if alternatives:
                    for alt in alternatives:
                        st.success(f"• {alt}")
                else:
                    st.info("No alternatives found in the local database.")
        else:
            st.error("An unknown error occurred.")