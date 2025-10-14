# Corporate LLM System

A private LLM system that answers questions based on your organization's internal documents using Google Gemini API and RAG (Retrieval-Augmented Generation).

## Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, Markdown, CSV, XLSX (Excel)
- 🌍 **Multilingual QA (English & Russian)**: Uses a multilingual embedding model; answers in the language of your question automatically
- 🔧 **Easily Extensible**: Plugin-style architecture to add support for new file types (JSON, XML, YAML, HTML, Excel, etc.)
- 🔍 **Semantic Search**: Vector similarity over chunked documents
- 🤖 **AI-Powered Answers**: Google Gemini with context grounding and source citation
- 💾 **Vector Database**: Persistent storage using ChromaDB
- 🔒 **Private & Secure**: All processing happens on your infrastructure
- 🌐 **REST API**: FastAPI-based web service
- 💻 **CLI Interface**: Command-line tools for easy interaction

## Architecture

```
User Question
     ↓
[Document Loader] → Load & chunk documents → [Vector Store (ChromaDB)]
     ↓                                              ↓
[Semantic Search] ← Retrieve relevant chunks ←─────┘
     ↓
[Google Gemini API] → Generate answer using context
     ↓
Return Answer + Sources
```

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ML_LLM
```

2. **Create a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
copy .env.example .env
```

Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

**Get your Google API key**: https://makersuite.google.com/app/apikey

**Available Gemini Models** (as of Oct 2024):
- `gemini-2.0-flash` - Latest, fastest, most cost-effective (recommended)
- `gemini-2.5-flash` - Newer experimental version
- `gemini-2.5-pro` - Most capable, highest quality responses
- Legacy aliases: `gemini-flash-latest`, `gemini-pro-latest`

## Quick Start

### 1. Add Your Documents

Place your internal documents in the `documents/` folder:
```
documents/
├── company_policies.pdf
├── technical_docs.docx
├── procedures.txt
├── knowledge_base.md
└── employee_data.csv
```

**Supported formats**: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`

**Note**: CSV files are automatically formatted into readable text with headers and row data.

### 2. Index Documents

```bash
python main.py index
```

To clear existing index and re-index:
```bash
python main.py index --clear
```

### 3. Ask Questions

**Using CLI:**
```bash
python main.py ask "What is our company's vacation policy?"
```

**Interactive mode:**
```bash
python main.py interactive
```

**Using REST API:**
```bash
# Start the server
python api_server.py

# In another terminal, make requests
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our company policy on remote work?"}'
```

## Usage Examples

### Command-Line Interface

**Check system statistics:**
```bash
python main.py stats
```

**Ask a question:**
```bash
python main.py ask "How do I submit an expense report?"
```

**Ask without document context (direct Gemini query):**
```bash
python main.py ask "What is Python?" --no-context
```

**Clear the index:**
```bash
python main.py clear
```

### REST API

**Start the server:**
```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

**API Documentation:** Visit `http://localhost:8000/docs` for interactive Swagger UI

**Example requests:**

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are the company benefits?"}
)
print(response.json())

# Index documents
response = requests.post(
    "http://localhost:8000/index",
    json={"clear_existing": False}
)
print(response.json())

# Get statistics
response = requests.get("http://localhost:8000/stats")
print(response.json())
```

## Configuration

Edit `.env` to customize settings:

```bash
# Google Gemini API
GOOGLE_API_KEY=your_api_key_here

# Embedding model for vector search (multilingual recommended for RU + EN)
# Examples:
# paraphrase-multilingual-MiniLM-L12-v2 (default, good balance)
# distiluse-base-multilingual-cased-v2 (slightly larger)
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Document chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Storage paths
VECTOR_DB_PATH=./vector_db
DOCUMENTS_PATH=./documents

# Retrieval settings
TOP_K_RESULTS=3

# Gemini settings
# Available: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
GEMINI_MODEL=gemini-2.0-flash
TEMPERATURE=0.3
MAX_OUTPUT_TOKENS=2048
```

## Project Structure

```
ML_LLM/
├── main.py              # CLI entry point
├── api_server.py        # FastAPI REST server
├── corporate_llm.py     # Main system integration
├── llm_engine.py        # Google Gemini integration
├── vector_store.py      # ChromaDB vector store
├── document_loader.py   # Document loading utilities
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # This file
├── documents/           # Place your documents here
└── vector_db/           # Vector database (auto-created)
```

## How It Works

1. **Document Loading**: Documents are loaded from various formats (PDF, DOCX, TXT, MD)
2. **Text Chunking**: Large documents are split into manageable chunks with overlap
3. **Embedding**: Each chunk is converted to a vector embedding using SentenceTransformers
4. **Storage**: Embeddings are stored in ChromaDB for fast similarity search
5. **Retrieval**: When a question is asked, relevant chunks are retrieved using semantic search
6. **Generation**: Google Gemini generates an answer based on the retrieved context
7. **Response**: The answer is returned along with source citations

## Security Considerations

- **API Key Security**: Never commit your `.env` file. The Google API key should be kept secret
- **Private Deployment**: Run this on your private infrastructure
- **Access Control**: Add authentication to the API server for production use
- **Data Privacy**: Documents never leave your infrastructure except for Gemini API calls
- **Network Security**: Use HTTPS and firewall rules in production

## Troubleshooting

**"No documents found to index"**
- Make sure you have documents in the `documents/` folder
- Check that file formats are supported (.pdf, .docx, .txt, .md)

**"Error: API key not found"**
- Ensure `.env` file exists and contains `GOOGLE_API_KEY`
- Get your API key from https://makersuite.google.com/app/apikey

**"Import errors"**
- Make sure you've installed all requirements: `pip install -r requirements.txt`
- Activate your virtual environment

**Poor answer quality**
- Index more documents for better context
- Adjust `TOP_K_RESULTS` to retrieve more context chunks
- Modify `TEMPERATURE` for more/less creative responses

## Multilingual Usage (Русский / English)

The system automatically detects if your question contains Cyrillic characters and will answer in Russian; otherwise it answers in English.

To ensure high‑quality retrieval for Russian content use a multilingual embedding model (already set by default):

```
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

Ask in Russian, e.g.:

```bash
python main.py ask "Какие ключевые риски указаны в отчёте по проекту AI Platform?"
```

If context is insufficient, the model will explicitly say so (in the same language) instead of hallucinating.

## Advanced Usage

### Programmatic Usage

```python
from corporate_llm import CorporateLLM

# Initialize the system
llm = CorporateLLM()

# Index documents
llm.index_documents()

# Ask questions
result = llm.ask("What is our security policy?")
print(result['answer'])
print(result['sources'])

# Get statistics
stats = llm.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### Custom Configuration

```python
from config import Settings

# Create custom settings
custom_settings = Settings(
    google_api_key="your_key",
    embedding_model="all-mpnet-base-v2",  # More powerful model
    chunk_size=1500,
    temperature=0.1  # More deterministic
)
```

### Adding Support for New File Types

The document loader uses a **plugin-style architecture** that makes it easy to add support for new file formats:

```python
from pathlib import Path
from document_loader import DocumentLoader

# Define a custom loader function
def load_json(file_path: Path) -> str:
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

# Register the loader
loader = DocumentLoader("./documents")
loader.register_loader('.json', load_json)

# Now you can load JSON files!
doc = loader.load_document(Path('./documents/data.json'))
```

**See `custom_loaders_example.py` for ready-to-use loaders for:**
- JSON files
- XML files
- YAML files
- HTML files (requires BeautifulSoup)
- Excel files (requires pandas)
- And more!

Each loader is just a simple function that takes a `Path` and returns a `str`. This makes it incredibly easy to add support for any file format.

## License

MIT License - Feel free to use this for your organization's needs.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Support

For issues or questions, please open an issue on GitHub.
