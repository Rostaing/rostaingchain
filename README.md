<p align="center">
  <a href="https://pypi.org/project/rostaingchain/"><img src="https://img.shields.io/pypi/v/rostaingchain?color=blue&label=PyPI%20version" alt="PyPI version"></a>
  <a href="https://pypi.org/project/rostaingchain/"><img src="https://img.shields.io/pypi/pyversions/rostaingchain.svg" alt="Python versions"></a>
  <a href="https://github.com/Rostaing/rostaingchain/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/rostaingchain.svg" alt="License"></a>
  <a href="https://pepy.tech/project/rostaingchain"><img src="https://static.pepy.tech/badge/rostaingchain" alt="Downloads"></a>
</p>

# 🧠 RostaingChain

**The Ultimate Hybrid RAG Framework.**  
Local & Remote LLMs | Real-time Watcher | Deep Data Profiling | DLP Security | Multi-Modal

**RostaingChain** is a production-ready framework designed to build autonomous RAG (Retrieval-Augmented Generation) systems. It bridges the gap between local privacy (Ollama, Local Docs) and cloud power (OpenAI, Groq, Datastores), featuring a unique **Live Watcher** that updates your AI's knowledge in real-time.

---

## 🚀 Key Features

*   **Hybrid Intelligence:** Switch instantly between Local LLMs (Ollama, Llama.cpp) and Remote giants (OpenAI, Groq, Claude, Gemini, DeepSeek, Grok).
*   **Live Watcher (Auto-Sync):** Drop a file in a folder, modify a SQL row, or update a website -> The AI learns it instantly.
*   **Deep Profiling (Anti-Hallucination):** Automatically calculates statistics (Max, Min, Mean) for CSV/SQL data so the LLM never hallucinates numbers.
*   **DLP Security:** Built-in Redaction system to **mask sensitive data:** (
- EMAIL: Email masked,
- PHONE: Phone number masked,
- ID_NUM: Personal ID masked,
- PASSPORT: Passport number masked,
- SSN: Social Security Number masked,
- POSTAL: City/Postal Code masked,
- BIC: BIC code confidential,
- IBAN: IBAN bank details protected,
- VAT_ID: VAT number masked,
- CREDIT_CARD: Credit card number masked,
- MONEY: Financial amount masked,
- CRYPTO: Crypto wallet masked,
- IP_ADDR: IP address masked,
- MAC_ADDR: MAC address masked,
- API_KEY: API Key redacted,
- DATE: Date masked ) before display.
*   **Multi-Modal Native:** Understands Text, PDFs (OCR included), Images, Audio (Whisper), and YouTube videos.
*   **Universal Sources:** Connects to Local Files, PostgreSQL, MySQL, Oracle, SQLite, MongoDB, Neo4j, and the Web.

---

### 📦 Standard installation (Quick):

```bash
pip install rostaingchain
# Optional: Install OCR capabilities
pip install rostaing-ocr
```

### 📦 “Power User” installation (All-inclusive):

```bash
pip install rostaingchain[all]
```

### 📦 Specific installation (e.g., only for SQL and using remote LLMs):

```bash
pip install rostaingchain[database,llms]
```

### 📦 For office documents and advanced OCR:

```bash
pip install rostaingchain[docs,llms]
```

### 📦 For multimedia (YouTube, audio, video, web):

```bash
pip install rostaingchain[media,llms]
```

## 🔑 Managing API Keys (Remote LLMs)

To use remote LLMs (like GPT-4, Groq, Claude) without hardcoding your credentials in the code, RostaingChain supports environment variables.

1.  **Create a file named `.env`** in your project root.
2.  **Add your API keys** following this format:

```env
# Standard Providers
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...

# Fast Inference Providers
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...

# OpenAI-Compatible Providers
DEEPSEEK_API_KEY=sk-...
XAI_API_KEY=...
```

3.  **Load the keys** at the start of your script using `python-dotenv`:

```bash
pip install python-dotenv
```

---

## ⚡ Quick Start

### 1. The "Chat with Anything" Mode

Simply point `data_source` to a file, a folder, a database, or a URL.

```python
from rostaingchain import RostaingBrain

# Initialize the Brain
agent = RostaingBrain(
    llm_model="llama3.2",          # Use local Ollama
    data_source="./my_documents", # Watches this folder
    auto_update=True             # Real-time ingestion
)

# Chat
response = agent.chat("What are the main topics in these documents?")
print(response)
```

---

## 🛠️ Advanced Usage

### 1. YouTube Video Analysis

Extract transcripts and metadata automatically.

```python
from rostaingchain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

agent = RostaingBrain(
    llm_model="openai/gpt-oss-120b",
    llm_provider="groq",
    data_source="https://www.youtube.com/watch?v=3mTK0vYYXA4",
    vector_db="faiss"
)

# Streaming response for better UX
generator = agent.chat("Summarize this video in 3 bullet points.", stream=True)

for token in generator:
    print(token, end="", flush=True)
```

### 2. Data Security (DLP)

Protect sensitive information from being displayed.

```python
from rostaingchain import RostaingBrain

agent = RostaingBrain(
    llm_model="llama3.2",
    data_source="bank_statements.pdf",
    # Enable Security
    security_filters=["IBAN", "BIC", "PHONE", "EMAIL", "MONEY", "CREDIT_CARD"] # Options: True or False
)

response = agent.chat("Give me the IBAN of the supplier.")
# Output: "The IBAN is [Protected IBAN bank details]."
```

### 3. Working with DataFrames (Pandas)

```python
import pandas as pd
from rostaingchain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

df = pd.read_csv("titanic.csv") # supports: Polars

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="gpt-4o",
    data_source=df 
)

print(agent.chat("What is the average age of passengers?"))
```

Here is the updated section to add to your **`README.md`**. It includes the specific example you requested, along with the detailed English explanations for `stream`, `output_format`, and `vector_db`.

---

### 4. Audio Analysis with Streaming & Markdown Output

RostaingChain natively handles audio files (like `.m4a`, `.mp3`) using OpenAI Whisper locally. This example demonstrates how to process an audio file, enforce security filters, and stream the result in a specific JSON format.

```python
from rostaingchain import RostaingBrain
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Assuming your API key is set
llm_api_key = os.getenv("GROQ_API_KEY")

agent = RostaingBrain(
    llm_model="openai/gpt-oss-120b",
    llm_provider="groq",
    llm_api_key=llm_api_key,
    data_source="C:/Users/Rostaing/Desktop/data/audio.m4a", # Supports: .m4a, .mp3, .wav, .ogg, .flac, .webm
    poll_interval=3600, # Check for file updates every hour
    vector_db="faiss",  # Options: 'faiss' or 'chroma'
    reset_db=True,      # Re-index the file on startup
    memory=True,        # Enable conversation history
    security_filters=["PHONE", "BIC", "IBAN", "DATE"] # Active Data Loss Prevention
)

# Request a summary in JSON format with streaming enabled
response = agent.chat("Give me a summary.", stream=True, output_format="markdown") # output_format supports: "json", "text (default)", "markdown", "toon"

# Real-time display loop
for token in reponse:
    # Prints every token as soon as it arrives (ChatGPT-like effect)
    print(token, end="", flush=True)
```

### 5. Chat with a Website (Web RAG)

```python
from RostaingChain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="gpt-4o",
    llm_provider="openai",
    data_source="https://en.wikipedia.org/wiki/Artificial_intelligence",
    vector_db="chroma",  # Options: 'faiss' or 'chroma'
)

response = gent.chat("Give me a summary.")
print(response)
```

### 6. Chat with an image (RAG)

```python
from RostaingChain import RostaingBrain

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="llama3.2", # Ensure you ran 'ollama pull llama3.2' in your terminal
    llm_provider="ollama", # Runs 100% locally on your machine for privacy
    embedding_model="nomic-embed-text", # Ensure you ran 'ollama pull nomic-embed-text' in your terminal
    data_source="invoice.jpg", # Supports: .png, .jpeg, .bmp, .tiff, .webp
    memory=True, # Enable conversation history
    vector_db="chroma",  # Options: 'faiss' or 'chroma'
)

response = gent.chat("Give me a summary.")
print(response)
```

### 7. Video Analysis with Streaming & Markdown Output

```python
from RostaingChain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="gpt-4o",
    llm_provider="openai",
    data_source="my_video.mp4", # Supports: .avi, .mov, .mkv
    vector_db="chroma",  # Options: 'faiss' or 'chroma'
)

response = gent.chat("Give me a summary.", stream=True, output_format="markdown") # output_format supports: "json", "text (default)", "markdown", "toon"

# Real-time display loop
for token in reponse:
    # Prints every token as soon as it arrives (ChatGPT-like effect)
    print(token, end="", flush=True)
```

### 8. Chat with a file (Streaming RAG)

```python
from RostaingChain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="gpt-4o",
    llm_provider="openai",
    data_source="my_file.txt", # Supports: .pdf, .docx, .doc, .xlsx, .xls, .pptx, .ppt, .html, .htm, .xml, .epub, .md, .json, .log, .py, .js, .sql, .yaml, .ini
    vector_db="chroma",  # Options: 'faiss' or 'chroma'
)

response = gent.chat("Give me a summary.", stream=True, output_format="markdown")

# Real-time display loop
for token in reponse:
    # Prints every token as soon as it arrives (ChatGPT-like effect)
    print(token, end="", flush=True)
```

### 9. Connecting to Databases (SQL / NoSQL)

RostaingChain uses a **Polling Watcher** to monitor database changes.

```python
from rostaingchain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# PostgreSQL Configuration
db_config = {
    "type": "sql",
    "connection_string": "postgresql+psycopg2://user:pass@localhost:5432/finance_db",
    "query": "SELECT * FROM sales_2024"
}

agent = RostaingBrain(
    llm_model="gpt-4o",
    llm_provider="openai",
    data_source=db_config,
    poll_interval=30, # Check for DB changes every 30 seconds
    reset_db=True     # Start with a fresh index
)

print(agent.chat("What is the total revenue for Q1?"))
# Thanks to Deep Profiling, the AI will know the exact sum/mean/max.
```

### 10 🗄️ Database Configuration Examples

To connect **RostaingChain** to a database, create a dictionary `db_config` and pass it to the `data_source` parameter.

### 1. SQL Databases (via SQLAlchemy)

**PostgreSQL**
```python
pg_config = {
"type": "sql",
"connection_string": "postgresql+psycopg2://user:pass@localhost:5432/finance_db",
"query": "SELECT * FROM sales_2026"
}
```

**MySQL**
```python
mysql_config = {
    "type": "sql",
    "connection_string": "mysql+pymysql://username:password@localhost:3306/my_database",
    "query": "SELECT * FROM orders WHERE status = 'shipped'"
}
```

**Oracle**
```python
# Requires Oracle Instant Client installed
oracle_config = {
    "type": "sql",
    "connection_string": "oracle+cx_oracle://username:password@localhost:1521/?service_name=ORCL",
    "query": "SELECT * FROM employees"
}
```

**SQLite**
```python
sqlite_config = {
    "type": "sql",
    "connection_string": "sqlite:///C:/path/to/my_data.db",
    "query": "SELECT * FROM invoices"
}
```

**Microsoft SQL Server**
```python
mssql_config = {
    "type": "sql",
    "connection_string": "mssql+pymssql://username:password@localhost:1433/my_database",
    "query": "SELECT top 100 * FROM customers"
}
```

### 2. NoSQL Databases

**MongoDB**
```python
mongo_config = {
    "type": "mongodb",
    "uri": "mongodb://localhost:27017/",
    "db": "ecommerce_db",
    "collection": "products",
    "limit": 50 # Optional: Limit the number of documents to ingest
}
```

**Neo4j (Graph)**
```python
neo4j_config = {
    "type": "neo4j",
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password",
    "query": "MATCH (p:Person)-[:WROTE]->(a:Article) RETURN p.name, a.title LIMIT 20"
}
```

### Usage Example

```python
agent = RostaingBrain(
    llm_model="gpt-4o",
    data_source=mysql_config, # Pass the dictionary here.
    poll_interval=60,         # Watch for changes every minute
    reset_db=True,
)
```

### 11. Use a custom LLM (e.g., vLLM on another server)

```python
from RostaingChain import RostaingBrain
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Direct Memory Ingestion
agent = RostaingBrain(
    llm_model="my-finetuned-model",
    llm_provider="custom",
    llm_base_url="http://192.168.1.50:8000/v1", # Your vLLM server
    llm_api_key="token-if-needed",
    memory=True,
    vector_db="chroma",  # Options: 'faiss' or 'chroma'
    data_source="my_file.pdf", # Supports: .txt, .docx, .doc, .xlsx, .xls, .pptx, .ppt, .html, .htm, .xml, .epub, .md, .json, .log, .py, .js, .sql, .yaml, .ini, .jpg, .png, .jpeg, .bmp, .tiff, .webp, SQL/NoSQL Databases, Audio/Video/Web(link)
    reset_db=True, # Start with a fresh index
    temperature=0,
    top_k=0.1,
    top_p=1,
    max_tokens=1500
)

response = gent.chat("Give me a summary.", stream=True)

# Real-time display loop
for token in reponse:
    # Prints every token as soon as it arrives (ChatGPT-like effect)
    print(token, end="", flush=True)
```

### 12. Universal Intelligence: Switching LLM Providers

**A. Use DeepSeek (the cheaper GPT-4 alternative)**
```python
agent = RostaingBrain(
    llm_model="deepseek-chat", # Auto-detection
    provider="deepseek",
    # If the key is not in the .env:
    llm_api_key="sk-your-deepseek-key" 
)
```

**B. Use Groq (Lightning speed – 500 tokens/s)**
```python
agent = RostaingBrain(
    llm_model="openai/gpt-oss-120b",
    llm_provider="groq" # Force the provider to ensure it
)
```

**C. Use Claude Sonnet (Best for coding)**
```python
agent = RostaingBrain(
    llm_model="claude-4.5-sonnet",
    llm_provider="anthropic" # Force the provider to ensure it
)
```

**D. Use Gemini 3 Pro (Google)**
```python
agent = RostaingBrain(
    llm_model="gemini-3-pro-preview",
    llm_provider="geoogle" # Force the provider to ensure it
)
```

**E. Use Mixtral (via Groq for Speed)**
```python
agent = RostaingBrain(
    llm_model="mistral-large-2512",
    llm_provider="mistral" # Force the provider for ultra-fast inference
)
```

**F. Use Grok (xAI)**
```python
agent = RostaingBrain(
     llm_model="grok-4.1",
    llm_provider="grok" # Automatically configures the xAI API base_url
)
```

**G. Use OpenAI (GPT-4o)**
```python
agent = RostaingBrain(
    llm_model="gpt-4o",
    llm_provider="openai" # Automatically uses OPENAI_API_KEY from your .env file
)
```

**H. Use Local LLMs (Ollama)**
```python
agent = RostaingBrain(
    llm_model="llama3.2",  # Ensure you ran 'ollama pull llama3.2' in your terminal
    llm_provider="ollama", # Runs 100% locally on your machine for privacy
    # llm_base_url="http://localhost:11434" # Optional: Default URL
)
```

#### 📝 Key Parameters Explained

*   **`stream=True`**:
    This is essential for User Experience (UX). Instead of waiting for the entire response to be generated (which can take time for long summaries), the method returns a Python **Generator**. You must iterate over it (using a `for` loop) to display tokens in real-time, exactly like ChatGPT.

*   **`output_format`**:
    This parameter enforces the structure or style of the LLM's response. It accepts three values:
    *   `"text"` (Default): A standard, conversational plain text response.
    *   `"json"`: Forces the LLM to output a valid JSON object. Extremely useful if you are building an API or need to parse the result programmatically.
    *   `"toon"`: Changes the persona to a funny, cartoon-like character.

*   **`vector_db`**:
    Defines the local vector storage engine. RostaingChain currently supports two robust, file-based options:
    *   `"chroma"`: Uses ChromaDB.
    *   `"faiss"`: Uses Facebook AI Similarity Search (highly efficient for CPU).

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `llm_model` | str | `"llama3"` | Name of the model (e.g., "gpt-4o", "llama3"). |
| `llm_provider` | str | `"auto"` | "openai", "groq", "ollama", "anthropic"... |
| `data_source` | str/dict/df | `"./data"` | File path, folder, URL, SQL config, or DataFrame. |
| `vector_db` | str | `"chroma"` | "chroma", "faiss". |
| `auto_update` | bool | `True` | Activates Watcher (File system or Polling). |
| `poll_interval` | int | `60` | Seconds between DB/Web checks. |
| `reset_db` | bool | `False` | Wipes vector DB on startup. |
| `memory` | bool | `False` | Enables conversational history. |
| `cache` | bool | `True` | Enables In-Memory caching for speed. |
| `security_filters` | list/bool | `None` | List of filters (["IBAN", "PHONE"]) or True for all. |
| `temperature` | float | `0.1` | Creativity of the model. |

---

## 💡 Pro Tip: VSCode Autocomplete

Don't memorize the parameters! If you are using **VSCode**, you can view the complete list of available options for `RostaingBrain` instantly.

Just place your cursor inside the parentheses and press:

**`Ctrl` + `Space`**

This will trigger IntelliSense and display all configuration arguments (like `memory`, `security_filters`, `temperature`, `cache`, etc.) with their descriptions.

## 🏗️ Architecture

```mermaid
graph TD
    A[Data Source] -->|Watcher/Polling| B(Universal Loader)
    B -->|Deep Profiling & OCR| C{"Content Type?"}
    C -->|Text/Code| D[Text Splitter]
    C -->|Table/SQL| E[Statistical Summary]
    C -->|Audio/Video| F[Whisper Model]
    
    D & E & F --> G[Embeddings Manager]
    G --> H[("Vector Database")]
    
    User -->|Query| I[Core Engine]
    I -->|"Retrieval (MMR)"| H
    I -->|Context + History| J[LLM Engine]
    J -->|Raw Response| K[Security Layer]
    K -->|Clean Response| User
```

## Useful Links
- [Author's LinkedIn](https://www.linkedin.com/in/davila-rostaing/)
- [Author's YouTube Channel](https://youtube.com/@RostaingAI?sub_confirmation=1)
- [GitHub Repository](https://github.com/Rostaing/rostaingchain)
- [PyPI Project Page](https://pypi.org/project/rostaingchain/)

## 📄 License

MIT License. See `LICENSE` for more information.