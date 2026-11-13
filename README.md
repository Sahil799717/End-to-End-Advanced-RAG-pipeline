# End-to-End-Advanced-RAG-pipeline
Built rag pipeline
# Advanced RAG Application

An advanced Retrieval-Augmented Generation (RAG) application built using **LangChain, LangGraph, Streamlit, ChromaDB, and Gemini**. The project focuses on improving retrieval relevance, answer accuracy, response reliability, and production readiness through advanced retrieval techniques, agentic workflows, evaluation, and guardrails.

## Key Features

- **Document Processing:** Supports PDF document ingestion and text extraction.
- **Vector Database:** ChromaDB for storing and retrieving document embeddings.
- **Semantic Chunking:** Splits documents based on semantic similarity using configurable breakpoint thresholds.
- **Multi-Query Retrieval:** Generates multiple search queries to improve document retrieval coverage.
- **FlashRank Reranking:** Reranks retrieved documents using `ms-marco-MiniLM-L-12-v2` to prioritize relevant chunks.
- **Agentic RAG:** Uses LangGraph to orchestrate the retrieval and generation workflow.
- **LLM Integration:** Uses Gemini and Groq integrations.
- **Guardrails:** Implements input and output validation to improve application safety and reliability.
- **Evaluation:** Uses DeepEval to evaluate retrieval and generation quality.
- **Interactive UI:** Streamlit-based interface.
- **Docker Deployment:** Containerized deployment support.

## Tech Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| LLM Framework | LangChain |
| Agentic Workflow | LangGraph |
| Vector Database | ChromaDB |
| Embeddings | Ollama |
| LLM Providers | Google Gemini, Groq |
| Document Processing | PyPDF |
| Chunking | Semantic Chunking |
| Retrieval | Multi-Query Retriever |
| Reranking | FlashRank |
| Evaluation | DeepEval |
| Guardrails | Guardrails AI |
| Deployment | Docker |

## RAG Architecture

### 1. Document Ingestion
- Upload PDF documents.
- Extract document text.
- Process extracted content for indexing.

### 2. Semantic Chunking
- Split documents based on semantic meaning rather than relying exclusively on fixed-size chunks.
- Configure semantic breakpoint parameters:
  - `breakpoint_threshold_type = "percentile"`
  - `breakpoint_threshold_amount = 95`

### 3. Embedding Generation
- Generate vector embeddings using Ollama.
- Store the resulting vectors in ChromaDB.

### 4. Advanced Retrieval
- **Multi-Query Retriever:** Generates alternative queries to improve retrieval coverage and find relevant information from different perspectives.
- **FlashRank Reranker:** Reranks retrieved chunks using `ms-marco-MiniLM-L-12-v2` to improve the relevance of the final context.

### 5. Agentic Workflow
- Uses LangGraph to orchestrate the RAG workflow.
- Manages workflow execution through graph-based state management and conditional processing.

### 6. Answer Generation
- Uses an LLM to generate answers using the retrieved context.
- Aims to provide context-grounded responses.

### 7. Guardrails
- Applies input and output validation.
- Uses Guardrails AI to enforce configured safety and reliability checks.

### 8. Evaluation
- Uses DeepEval to evaluate retrieval and answer generation.
- Evaluates relevant metrics such as:
  - Faithfulness
  - Answer Relevancy
  - Contextual Precision
  - Contextual Recall
  - Noise Sensitivity

## Project Structure

```text
RAG/
│
├── app.py
├── rag_backend.py
├── config.py
│
├── evals/
│   ├── eval_retriever.py
│   └── eval_generator.py
│
├── requirements.txt
├── Dockerfile
├── .env
├── .gitignore
└── README.md
```

*Adjust the structure above to match your actual project files.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG
```

### 2. Create Virtual Environment

```bash
python -m venv myenv312
```

Activate it on Windows:

```bash
myenv312\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Add any additional environment variables required by your configuration.

### 5. Run the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Evaluation

Run retrieval evaluation:

```bash
python -m evals.eval_retriever
```

Run generation evaluation:

```bash
python -m evals.eval_generator
```

DeepEval helps measure the quality and reliability of the RAG pipeline using evaluation datasets and defined metrics.

## Docker Deployment

Build the Docker image:

```bash
docker build -t advanced-rag .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env advanced-rag
```

Open:

```text
http://localhost:8501
```

Ensure that external services such as Ollama are accessible from inside the Docker container. `localhost` inside a container refers to the container itself, not your Windows host.

## Environment Variables and Security

- Keep API keys inside `.env`.
- Never commit `.env` to GitHub.
- Add `.env`, virtual environments, and generated database files to `.gitignore`.
- Configure production secrets through Docker environment variables or a secrets manager.

## Future Improvements

- Improve retrieval performance through hybrid search.
- Add retrieval caching to reduce latency.
- Optimize LLM inference cost.
- Introduce continuous evaluation and monitoring.
- Improve scalability for larger document collections.

## Author

**Sahil Dubey**

AI/ML Developer | Generative AI | Agentic AI | RAG Systems
