# TDS Project 1 – LLM-Powered Data Science Tools

## Project Overview
**TDS Project 1** is an AI-driven tool orchestration system built with FastAPI. It leverages a Large Language Model (LLM) to interpret natural language queries and route them to the appropriate data-processing tools for execution. The system includes various tools, such as web scraping, data analysis, and database queries, each designed for specific functionalities. The LLM acts as a **function classifier**, determining the most suitable tool and extracting the necessary parameters for execution. If no predefined tool is applicable, the system defaults to a general-purpose execution tool (**CodeRunner**) to process the request. This approach enables users to perform complex operations through natural language inputs, with the LLM managing tool selection and execution.

## Installation & Setup
### Prerequisites
- **Python 3.12+** (including Docker support for Python 3.12).
- **Virtual environment** (recommended for dependency isolation).
- **LLM API credentials** (e.g., AI_PROXY API key or Google PaLM API token).

### Installation Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/rohitxiitm/tds-proj-1.git
   cd tds-proj-1
   ```
2. **Install Dependencies**
   ```bash
   pip install -e .
   ```
3. **Configure Environment Variables**
   Create a `.env` file and set the required API credentials:
   ```ini
   AIPROXY_TOKEN=your_proxy_token
   GEMINI_TOKEN=your_gemini_key
   ```
4. **Run the API Server**
   ```bash
   uvicorn app:app --reload
   ```
5. **(Optional) Docker Setup**
   ```bash
   docker build -t tds-proj-1 .
   docker run -p 8000:8000 tds-proj-1
   ```

## Usage
### API Endpoints
- **GET `/health`** – Returns `{ "status": "healthy" }` to indicate service availability.
- **POST `/run`** – Executes a task using an LLM-selected tool.
  ```bash
  curl -X POST "http://localhost:8000/run" -H "Content-Type: application/json" \
       -d '{"task": "Fetch the latest data from the website and compute the total sales."}'
  ```
  The LLM determines the appropriate tool and returns the processed output.
- **GET `/read?path=<file_path>`** – Reads and returns the contents of a specified local file.

## Project Architecture
### Key Components
- **`app.py` (FastAPI App)** – Defines API routes, processes requests, and orchestrates LLM interactions.
- **`llm.py` (LLM Interface)** – Communicates with OpenAI/Gemini APIs, formats queries, and selects tools.
- **`base.py` (Tool Base Class)** – Defines `BaseTool`, which all tools inherit.
- **`tools/` (Tool Implementations)** – Houses tools for data processing, web scraping, SQL queries, etc.
- **`constants.py` & `utils.py`** – Stores reusable constants and helper functions.
- **`Dockerfile`** – Provides a containerized setup for deployment.

### Workflow
1. The **LLM processes** the user’s request.
2. It selects the **most relevant tool** from the available options.
3. The **FastAPI server executes** the tool and returns the results.
4. If no predefined tool matches, the system invokes **CodeRunner** for execution.

## Summary
This project enables **natural language-driven data processing**, where an LLM intelligently selects and executes tools. The modular architecture allows seamless integration of new tools, making the system scalable and adaptable to various data science tasks.

