# Product Dataset Analyzer

**Product Dataset Analyzer** is a data analysis project designed to process and explore product datasets, clean and transform the data, and generate meaningful insights for business decision-making.

The project leverages advanced data processing techniques, including data cleaning (handling missing values, duplicates, and formatting), statistical analysis, and visualizations. It also implements **pattern detection and clustering*** using PCA and KMeans to identify trends and group similar products.

Additionally, the project uses a **LLM API (Gemini)** to generate automatic reports summarizing the dataset, key statistics, detected patterns, and actionable insights, making it easier for users to understand their product data.

Users can upload their product datasets, visualize distributions and correlations, detect clusters, and quickly generate downloadable reports, providing a comprehensive tool for data-driven decision-making in product management and business strategy.

## Installation

### 1 - Clone the repository
```bash
git clone https://github.com/Ayoub-Lamkaddem/langgraph-product-analyst-crew.git
cd langgraph-product-analyst-crew
```

### 2 - Install **uv**
Before anything, install uv depending on your OS:

- **For Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- **For macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 3 - Create a virtual environment (recommended) and Install dependencies:

```bash
uv init --python 3.12
uv venv
uv sync
```
### 4 - Configure the .env file
Create a **.env** file in the root of your project with the following structure:

```.env
    ### GEMINI API KEY

    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### 5- Activate the virtual environment and run the project:
- **langgraph-product-analyst-crew**
```bash
# Windows
    ./venv/scripts/activate

# Linux or Mac
    source .venv/bin/activate

cd frontend

# Run the frontend
    streamlit run app.py
```

# Demo

The project is deployed on **Streamlit Cloud**. 

Try the app here: [USA Housing Price Prediction App](https://usahousingpriceprediction-xbezkduhet8gpjzoynktkb.streamlit.app/)