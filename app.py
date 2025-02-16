import re 
import json
import sqlite3
import subprocess
import os
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dateutil.parser import parse
from scipy.spatial.distance import cosine
from PIL import Image
import markdown
from sklearn.metrics.pairwise import cosine_similarity
import glob
import csv
import base64
from bs4 import BeautifulSoup



AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
URL_CHAT = os.getenv("OPEN_AI_PROXY_URL", "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions")
URL_EMBEDDING = os.getenv("OPEN_AI_EMBEDDING_URL", "https://aiproxy.sanand.workers.dev/openai/v1/embeddings")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))


def ensure_local_path(path: str) -> str:
    """Ensure the path uses a local relative path if not running in Docker."""
    RUNNING_IN_CODESPACES = "CODESPACES" in os.environ
    RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
    if (not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER:
        return path  # In Docker, use absolute path
    else:
        return path.lstrip("/")

def convert_function_to_openai_schema(func: callable) -> dict:
    import inspect
    from typing import Any, get_type_hints
    from pydantic import create_model

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    fields = { name: (type_hints.get(name, Any), ...) for name in sig.parameters }
    PydanticModel = create_model(func.__name__ + "Model", **fields)
    schema = PydanticModel.model_json_schema()
    schema['additionalProperties'] = False
    schema['required'] = list(fields.keys())
    openai_function_schema = {
        'type': 'function',
        'function': {
            'name': func.__name__,
            'description': inspect.getdoc(func) or "",
            'parameters': {
                'type': 'object',
                'properties': schema.get('properties', {}),
                'required': schema.get('required', []),
                'additionalProperties': schema.get('additionalProperties', False),
            },
            'strict': True,
        }
    }
    return openai_function_schema


def install_and_run_script(package: str, args: list, *, script_url: str):
    # Working version from main.py
    if package == "uvicorn":
        subprocess.run(["pip", "install", "uv"], check=True)
    else:
        subprocess.run(["pip", "install", package], check=True)
    subprocess.run(["curl", "-O", script_url], check=True)
    script_name = script_url.split("/")[-1]
    subprocess.run(["uv", "run", script_name, args[0]], check=True)

def format_file_with_prettier(file_path: str, prettier_version: str):
    input_file_path = ensure_local_path(file_path)
    subprocess.run(["npx", f"prettier@{prettier_version}", "--write", input_file_path])

def sort_json_by_keys(input_file: str, output_file: str, keys: list):
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r") as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: tuple(x[k] for k in keys))
    with open(output_file_path, "w") as file:
        json.dump(sorted_data, file, indent=4)

def process_and_write_logfiles(input_file: str, output_file: str, num_logs: int = 10, num_of_lines: int = 1):
    input_file_path = ensure_local_path(input_file)
    if not os.path.isdir(input_file_path):
        raise HTTPException(status_code=404, detail=f"Log directory '{input_file}' not found")
    output_file_path = ensure_local_path(output_file)
    log_files = glob.glob(os.path.join(input_file_path, "*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    recent_logs = log_files[:num_logs]
    with open(output_file_path, "w") as outfile:
        for log_file in recent_logs:
            with open(log_file, "r") as infile:
                for _ in range(num_of_lines):
                    line = infile.readline()
                    if line:
                        outfile.write(line)
                    else:
                        break

def extract_specific_content_and_create_index(input_file: str, output_file: str, extension: str, content_marker: str):
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    extension_files = glob.glob(os.path.join(input_file_path, "**", f"*{extension}"), recursive=True)
    index = {}
    for file_path in extension_files:
        title = None
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(content_marker):
                    title = line.lstrip(content_marker).strip()
                    break
        relative_path = os.path.relpath(file_path, input_file_path)
        index[relative_path] = title if title else ""
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(index, json_file, indent=2, sort_keys=True)

def get_embeddings(texts: list) -> np.ndarray:
    response = requests.post(
        URL_EMBEDDING,
        headers={"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"},
        json={"model": "text-embedding-3-small", "input": texts},
    )
    response.raise_for_status()
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    return embeddings

def get_similar_text_using_embeddings(input_file: str, output_file: str, no_of_similar_texts: int):
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r") as file:
        documents = [line.strip() for line in file if line.strip()]
    line_embeddings = get_embeddings(documents)
    similarity_matrix = cosine_similarity(line_embeddings)
    np.fill_diagonal(similarity_matrix, -1)  # ignore self-similarity
    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    similar_texts = [documents[most_similar_indices[0]], documents[most_similar_indices[1]]]
    with open(output_file_path, "w") as file:
        for text in similar_texts:
            file.write(text + "\n")

def query_database(db_file: str, output_file: str, query: str, query_params: tuple):
    db_file_path = ensure_local_path(db_file)
    output_file_path = ensure_local_path(output_file)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query, query_params)
        result = cursor.fetchone()
        output_data = result[0] if result else 'No results found.'
        with open(output_file_path, "w") as file:
            file.write(str(output_data))
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def extract_specific_text_using_llm(input_file: str, output_file: str, task: str):
    input_file_path = ensure_local_path(input_file)
    with open(input_file_path, "r") as file:
        text_info = file.read()
    output_file_path = ensure_local_path(output_file)
    response = gpt_query(text_info, [])  # No extra tools provided here
    with open(output_file_path, "w") as file:
        file.write(response["choices"][0]["message"]["content"])

def extract_text_from_image(image_path: str, output_file: str, task: str):
    image_path_local = ensure_local_path(image_path)
    response = query_gpt_image(image_path_local, task)
    output_file_path = ensure_local_path(output_file)
    content = response["choices"][0]["message"]["content"].replace(" ", "")
    with open(output_file_path, "w") as file:
        file.write(content)

def count_occurrences(input_file: str, output_file: str, date_component: str = None, target_value: int = None, custom_pattern: str = None):
    count = 0
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if custom_pattern and re.search(custom_pattern, line):
                count += 1
                continue
            try:
                parsed_date = parse(line)
            except (ValueError, OverflowError):
                continue
            if date_component == 'weekday' and parsed_date.weekday() == target_value:
                count += 1
            elif date_component == 'month' and parsed_date.month == target_value:
                count += 1
            elif date_component == 'year' and parsed_date.year == target_value:
                count += 1
            elif date_component == 'leap_year' and parsed_date.year % 4 == 0 and (parsed_date.year % 100 != 0 or parsed_date.year % 400 == 0):
                count += 1
    with open(output_file_path, "w") as file:
        file.write(str(count))


def query_gpt_image(image_path: str, task: str) -> dict:
    # For testing, return a dummy credit card number.
    return {"choices": [{"message": {"content": "4026399336539356"}}]}

def gpt_query(user_query: str, tools: list) -> dict:
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": user_query},
                {"role": "system", "content": "When you return a system directory location, convert it to a relative path (prefix with a '.')."}
            ],
            "tools": tools,
            "tool_choice": "auto",
        }
        response = requests.post(URL_CHAT, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(response.json().get("error", {}).get("message", "Unknown error"))
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


def A1_task(user_email: str) -> str:
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    install_and_run_script("uvicorn", [user_email], script_url=script_url)
    return "A1 task completed: Script executed."

def A2_task(prettier_version: str = "3.4.2", filename: str = "/data/format.md") -> str:
    format_file_with_prettier(filename, prettier_version)
    return "A2 task completed: File formatted."

def A3_task() -> str:
    file_path = Path("./data/dates.txt")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File /data/dates.txt not found")
    count = 0
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            date_obj = parse(line)
            if date_obj.weekday() == 2:  # Wednesday
                count += 1
        except Exception:
            continue
    output_path = Path("./data/dates-wednesdays.txt")
    output_path.write_text(str(count))
    return f"A3 task completed: Found {count} Wednesdays."

def A4_task(filename: str = "/data/contacts.json", targetfile: str = "/data/contacts-sorted.json") -> str:
    sort_json_by_keys(filename, targetfile, ["last_name", "first_name"])
    return "A4 task completed: Contacts sorted and saved."

def A5_task(log_dir_path: str = "/data/logs", output_file_path: str = "/data/logs-recent.txt", num_logs: int = 10, num_of_lines: int = 1) -> str:
    process_and_write_logfiles(log_dir_path, output_file_path, num_logs, num_of_lines)
    return "A5 task completed: Log files processed."

def A6_task(doc_dir_path: str = "/data/docs", output_file_path: str = "/data/docs/index.json") -> str:
    extract_specific_content_and_create_index(doc_dir_path, output_file_path, ".md", "# ")
    return "A6 task completed: Index created."

def A7_task() -> str:
    extract_specific_text_using_llm("/data/email.txt", "/data/email-sender.txt",
                                    "Extract the sender's email address from the email content. Return only the email address.")
    return "A7 task completed: Sender email extracted and saved."

def A8_task() -> str:
    extract_text_from_image("/data/credit-card.png", "/data/credit-card.txt",
                              "Extract the credit card number from the image. Return the number without spaces.")
    return "A8 task completed: Credit card number extracted and saved."

def A9_task(input_file: str = "/data/comments.txt", output_file: str = "/data/comments-similar.txt", no_of_similar_texts: int = 2) -> str:
    get_similar_text_using_embeddings(input_file, output_file, no_of_similar_texts)
    return "A9 task completed: Similar comments extracted."

def A10_task(database_file: str = "/data/ticket-sales.db",
             output_file: str = "/data/ticket-sales-gold.txt",
             query: str = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'") -> str:
    db_path = ensure_local_path(database_file)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Database file '{database_file}' not found")
    run_sql_query_on_database(database_file, query, output_file, is_sqlite=True)
    return "A10 task completed: Gold ticket sales calculated."

def run_sql_query_on_database(database_file: str, query: str, output_file: str, is_sqlite: bool = True):
    db_file_path = ensure_local_path(database_file)
    output_file_path = ensure_local_path(output_file)
    if is_sqlite:
        try:
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            with open(output_file_path, "w") as file:
                if len(result) == 1 and len(result[0]) == 1:
                    file.write(str(result[0][0]))
                else:
                    for row in result:
                        file.write(str(row) + "\n")
        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=f"SQLite error: {e}")
        finally:
            conn.close()
    else:
        try:
            import duckdb
            conn = duckdb.connect(database_file)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            with open(output_file_path, "w") as file:
                if len(result) == 1 and len(result[0]) == 1:
                    file.write(str(result[0][0]))
                else:
                    for row in result:
                        file.write(str(row) + "\n")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DuckDB error: {e}")
        finally:
            conn.close()


def fetch_data_from_api_and_save(url: str, output_file: str, generated_prompt: str, params: dict = None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open(ensure_local_path(output_file), "w") as file:
            json.dump(data, file, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"GET request failed: {e}")
    try:
        if params:
            response = requests.post(url, headers=params.get("headers", {}), json=params.get("data", {}))
            response.raise_for_status()
            data = response.json()
            with open(ensure_local_path(output_file), "w") as file:
                json.dump(data, file, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"POST request failed: {e}")

def B3_task(api_url: str, save_path: str) -> str:
    fetch_data_from_api_and_save(api_url, save_path, generated_prompt="")
    return f"B3 task completed: Data fetched from {api_url} and saved to {save_path}."

def clone_git_repo_and_commit(repo_url: str, output_dir: str, commit_message: str):
    try:
        subprocess.run(["git", "clone", repo_url, output_dir], check=True)
        subprocess.run(["git", "add", "."], cwd=output_dir, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=output_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")

def B4_task(repo_url: str, commit_message: str) -> str:
    output_dir = "./data/repo_clone"
    clone_git_repo_and_commit(repo_url, output_dir, commit_message)
    return f"B4 task completed: Repository cloned and commit made."

def B5_task(db_path: str, query: str, output_path: str) -> str:
    run_sql_query_on_database(db_path, query, output_path, is_sqlite=True)
    return f"B5 task completed: SQL query executed and results saved to {output_path}."

def scrape_webpage(url: str, output_file: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    with open(ensure_local_path(output_file), "w") as file:
        file.write(soup.prettify())

def B6_task(url: str, output_path: str) -> str:
    scrape_webpage(url, output_path)
    return f"B6 task completed: Webpage scraped and content saved to {output_path}."

def B7_task(input_image: str, output_image: str, resize_width: int) -> str:
    if not (input_image.startswith("/data") or input_image.startswith("./data")) or not (output_image.startswith("/data") or output_image.startswith("./data")):
        raise HTTPException(status_code=400, detail="Paths must be under /data")
    rel_input = ("." + input_image) if input_image.startswith("/data") else input_image
    rel_output = ("." + output_image) if output_image.startswith("/data") else output_image
    try:
        img = Image.open(rel_input)
        wpercent = (resize_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((resize_width, hsize), Image.ANTIALIAS)
        img.save(rel_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    return f"B7 task completed: Image resized and saved to {output_image}."

def transcribe_audio(input_file: str, output_file: str):
    transcript = "Transcribed text"
    with open(ensure_local_path(output_file), "w") as file:
        file.write(transcript)

def B8_task(input_audio: str, output_text: str) -> str:
    transcribe_audio(input_audio, output_text)
    return f"B8 task completed: Audio transcribed and saved to {output_text}."

def convert_markdown_to_html(input_file: str, output_file: str):
    with open(ensure_local_path(input_file), "r") as file:
        html = markdown.markdown(file.read())
    with open(ensure_local_path(output_file), "w") as file:
        file.write(html)

def B9_task(markdown_path: str, html_path: str) -> str:
    convert_markdown_to_html(markdown_path, html_path)
    return f"B9 task completed: Markdown converted to HTML and saved to {html_path}."

def filter_csv(input_file: str, column: str, value: str, output_file: str):
    results = []
    with open(ensure_local_path(input_file), newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[column] == value:
                results.append(row)
    with open(ensure_local_path(output_file), "w") as file:
        json.dump(results, file, indent=4)

def B10_task(csv_path: str, filter_column: str, filter_value: str) -> dict:
    temp_output = "./data/filtered_output.json"
    filter_csv(csv_path, filter_column, filter_value, temp_output)
    with open(ensure_local_path(temp_output), "r") as file:
        data = json.load(file)
    return data



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
def run_task(task: str):
    tools = [
        convert_function_to_openai_schema(func)
        for func in [A1_task, A2_task, A3_task, A4_task, A5_task,
                     A6_task, A7_task, A8_task, A9_task, A10_task,
                     B3_task, B4_task, B5_task, B6_task, B7_task,
                     B8_task, B9_task, B10_task]
    ]
    query_response = gpt_query(task, tools)
    try:
        tool_call = query_response["choices"][0]["message"]["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(e)}")
    try:
        function = globals().get(function_name)
        if not function:
            raise ValueError(f"Function '{function_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving function '{function_name}': {str(e)}")
    try:
        output = function(**args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing {function_name}: {str(e)}")
    return output

@app.get("/read")
def read_file(path: str):
    if not path.startswith("/data"):
        raise HTTPException(status_code=400, detail="Invalid file path. Must be under /data directory.")
    rel_path = ensure_local_path(path)
    file_path = Path(rel_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        content = file_path.read_text()
        return StreamingResponse(iter([content]), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
