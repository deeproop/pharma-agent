# Pharma: Drug Ingredient Checker Agent 🧪

## Overview

This project implements an agentic AI application designed to retrieve active and inactive ingredients for medications. It addresses the need for healthcare professionals and patients to quickly and accurately verify drug components, mitigating risks associated with allergies, compliance issues, or therapeutic equivalence checks. Manual lookups are often slow and error-prone; this agent automates the process.

The agent queries a local database first for speed and falls back to external APIs (like the FDA) if the drug isn't found locally. It also includes intelligent features like ingredient classification and suggesting alternatives using web search and LLM reasoning.

---

## Features ✨

* **Ingredient Retrieval:** Accepts a drug name (brand or generic) as input.
* **Multi-Source Lookup:**
    * Checks a local FAISS vector store (RAG) for known drugs first.
    * If not found locally, queries the openFDA API.
    * (DailyMed API fallback was implemented but removed due to endpoint instability).
* **Name Normalization:** Handles common synonyms (e.g., "paracetamol" -> "Acetaminophen") before searching.
* **Ingredient Classification (Stretch Goal):** Uses web search (Tavily) and an Azure OpenAI LLM to determine the therapeutic category and identify potential common allergens.
* **Alternative Suggestions (Stretch Goal):** Uses web search (Tavily) and an Azure OpenAI LLM to suggest alternative drugs based on the primary active ingredient.
* **User Interface:** Simple and intuitive web interface built with Streamlit.
* **Error Handling:** Provides clear messages if a drug or its ingredients cannot be found in the available sources.

---

## Architecture & Technologies 🏗️

This project utilizes a **workflow-driven agent** architecture orchestrated by **LangGraph**.

* **Orchestration:** **LangGraph** manages the state and flow control, directing the agent through a defined sequence of steps with conditional logic.
* **Core Framework:** **LangChain** provides fundamental components like:
    * LLM Wrappers (`AzureChatOpenAI`)
    * Embedding Models (`AzureOpenAIEmbeddings`)
    * Prompt Templates (`ChatPromptTemplate`)
    * Output Parsers (`PydanticOutputParser`)
    * Tool Abstraction (`@tool`)
    * LCEL (LangChain Expression Language) for defining LLM chains within nodes.
* **RAG (Retrieval-Augmented Generation):**
    * **Vector Store:** **FAISS** (local) stores embeddings of drug data.
    * **Embeddings:** Generated using **Azure OpenAI Embedding models**.
    * **Retrieval:** The `check_local_database` tool performs similarity searches.
* **External APIs:**
    * **openFDA API:** Queried for drug label information.
    * **Tavily Search API:** Used by `web_search` tool to provide real-time web context for classification and alternative suggestion nodes.
* **LLM:** **Azure OpenAI Chat Models** (e.g., GPT-3.5-Turbo, GPT-4) are used within specific graph nodes (`classify_ingredients`, `find_alternatives`) for analysis and data extraction.
* **UI:** **Streamlit** provides the web-based user interface.
* **State Management:** Handled internally by LangGraph using a `TypedDict` (`GraphState`).

---

## Setup Instructions ⚙️

**Prerequisites:**
* Python 3.9+
* An Azure account with access to Azure OpenAI (for both Embedding and Chat model deployments).
* A Tavily Search API key ([Get one here](https://tavily.com/)).

**Steps:**

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Navigate to Project Directory:** Open your terminal/command prompt in the project folder (e.g., `cd pharma_agent`).
3.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```
4.  **Activate Environment:**
    * Windows: `.\venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt tavily-python --no-cache-dir
    ```
    *(Ensure `requirements.txt` contains: `langchain langgraph langchain-openai langchain-community streamlit faiss-cpu azure-identity python-dotenv requests`)*
6.  **Configure Environment Variables:**
    * Create a file named `.env` in the project root.
    * Add your API keys and endpoints:
        ```ini
        # .env
        AZURE_OPENAI_API_KEY="YOUR_AZURE_API_KEY"
        AZURE_OPENAI_ENDPOINT="httpsS://YOUR_[ENDPOINT.openai.azure.com/](https://ENDPOINT.openai.azure.com/)"
        AZURE_OPENAI_API_VERSION="2024-02-01" # Or your API version
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT="YOUR_EMBEDDINGS_DEPLOYMENT_NAME"
        AZURE_OPENAI_CHAT_DEPLOYMENT="YOUR_CHAT_DEPLOYMENT_NAME"
        TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
        ```
7.  **Build Local Vector Store:**
    * (Optional) Add more drug data to `local_drugs.json` following the existing format.
    * Run the ingestion script (make sure your `venv` is active):
        ```bash
        .\venv\Scripts\python.exe ingest.py
        # Or on macOS/Linux: python ingest.py
        ```
    * This creates the `faiss_index` folder.

---

## Running the Application ▶️

1.  **Activate Environment:** If not already active:
    * Windows: `.\venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
2.  **Run Streamlit:** Use the explicit path to the venv's Python to avoid path issues:
    * Windows: `.\venv\Scripts\python.exe -m streamlit run app.py`
    * macOS/Linux: `venv/bin/python -m streamlit run app.py`
    *(Alternatively, if your environment is set up correctly, `streamlit run app.py` might work directly).*
3.  Open the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

---

## Agent Workflow 🔁

1.  **Input:** Drug name received.
2.  **Normalize:** Check synonym map, update name if needed.
3.  **Check Local:** Query FAISS index. If confident match -> Go to Step 5.
4.  **Check FDA:** Query openFDA API. If ingredients found -> Go to Step 5. If not found/error -> Go to Step 6.
5.  **Process:** (Run only if ingredients were found)
    * **Classify:** Use Web Search + LLM to analyze ingredients.
    * **Alternatives:** Use Web Search + LLM to find alternatives.
    * Go to Step 7.
6.  **Handle Not Found:** Prepare final error message. Go to Step 7.
7.  **Output:** Display results or error message in Streamlit.

---

## Future Enhancements (Optional) 🚀

* Expand `NAME_MAP` and implement fuzzy matching.
* Add a web search fallback node for *ingredient extraction*.
* Integrate a drug interaction checker.
* Refine ingredient parsing logic.
* Improve UI feedback (loading indicators, data source display).