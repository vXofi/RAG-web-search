from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from playwright.async_api import async_playwright
import re
from llama_cpp import Llama
from datetime import datetime
import json


"""
TODO:
when retrieving numerical data it is not explicitly similar to query
no prints in console after first message
somehow model answers with markdown when markdown is mentioned in query
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global playwright_instance, browser, chat_model, tokenizer

    print("Startup: Initializing Playwright")
    playwright_instance = await async_playwright().start()
    browser = await playwright_instance.firefox.launch(headless=True)

    print("Loading model...")
    my_model_path = "./model/Phi-3.5-mini-instruct-Q4_K_L.gguf"
    CONTEXT_SIZE = 2048
    chat_model = Llama(model_path=my_model_path,
                       n_ctx=CONTEXT_SIZE, n_threads=6)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    print("Models loaded successfully!")

    yield
    print("Shutdown: Cleaning up Playwright")
    await browser.close()
    await playwright_instance.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


app.mount("/static", StaticFiles(directory="static"), name="static")

headers = {"user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}


async def ddg_search(query, tokenizer):
    """Performs a DuckDuckGo search and returns relevant content."""
    query = query.replace('"', '')
    results = DDGS(headers=headers).text(query, max_results=10)
    urls = [result['href'] for result in results]

    docs = await get_page(urls)

    content = []

    for doc in docs:
        page_text = process_page_content(doc)
        text = text_to_chunks(page_text, tokenizer)
        content.extend(text)

    return content


async def get_page(urls):
    global browser
    context = await browser.new_context(user_agent=headers["user_agent"])
    page = await context.new_page()

    html_documents = []
    for url in urls:
        try:
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            html_documents.append(content)
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    await context.close()

    return html_documents

def process_page_content(html):
    soup = BeautifulSoup(html, "html.parser")

    structured_text = []
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        if tag.name == "h1":
            structured_text.append(f"# {tag.text.strip()}\n")
        elif tag.name == "h2":
            structured_text.append(f"## {tag.text.strip()}\n")
        elif tag.name == "h3":
            structured_text.append(f"### {tag.text.strip()}\n")
        elif tag.name == "p":
            paragraph = re.sub(
                r"(http[s]?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b)",
                "",
                tag.text
            ) # removes links
            paragraph = re.sub(r"\n+", " ", paragraph).strip()
            if paragraph:
                structured_text.append(paragraph)
    
    return "\n\n".join(structured_text)


def text_to_chunks(text, tokenizer, chunk_size=512, overlap_size=128):
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    move = chunk_size - overlap_size
    end = len(tokens)

    while start < end:
        chunk_tokens = tokens[start:start + chunk_size]
        chunks.append(chunk_tokens)

        start += move

    return chunks


def rank_snippets(query, snippets, tokenizer, model_name="all-MiniLM-L6-v2", top_k=2):
    embedder = SentenceTransformer(model_name)

    text_chunks = [tokenizer.decode(chunk) for chunk in snippets]

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    snippet_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, snippet_embeddings).squeeze(0)

    top_indices = similarities.argsort(descending=True)
    top_snippets = [text_chunks[idx] for idx in top_indices[:top_k]]
    
    return top_snippets

def extract_json(response_text):
    try:
        json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
        return json.loads(json_str)
    except (AttributeError, json.JSONDecodeError):
        raise ValueError("No valid JSON found in response")

def generate_answer_stream(query, context, model,
                             max_tokens=1024,
                             temperature = 0.7,
                             top_p = 0.1,
                             stop = ["<|end|>"]):
        prompt_messages = [{
        "role": "system",
        "content": f"""You are a helpful assistant. Current date: {datetime.now().strftime('%Y-%m-%d')}.
        1. Use web search results ONLY when:
        - Query requires current information
        - Needs specific data/statistics
        - About recent events/developments
        2. For common knowledge answer directly

        Web results (if available):
        {context}"""
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }]
        
        response_stream = model.create_chat_completion(
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True)

        print("\n--- Assistant Response ---\n", end="")

        for chunk in response_stream:

            token = chunk["choices"][0]["delta"].get("content", "")

            print(token, end="", flush=True)

            yield f"data: {token}\n\n"

class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/rag")
async def rag_endpoint(query: str):

    try:
        '''
        decision_prompt = f"""<|system|>You are a helpful assistant with access to a web search function.
        For the given user query, carefully evaluate if it requires using the 'web_search' function.

        Only use the web search function when the query:
        1. Requires up-to-date information
        2. Needs specific details that aren't common knowledge
        3. Involves current events, statistics, or complex technical information
        4. Requires verification of facts that you're not completely certain about

        For basic questions about common knowledge, general facts, or simple concepts, respond directly without using the web search.

        You have access to the following tools:
        <function_calls>
        {{
            "web_search": {{
                "name": "web_search",
                "description": "Searches the web for relevant information.",
                "parameters": {{
                    "query": {{
                        "type": "string",
                        "description": "The search query to use."
                    }}
                }}
            }}
        }}
        </function_calls>

        Response format:
        - For questions that can be answered directly: Provide a plain text response without any JSON formatting or code blocks
        - For questions requiring web search: Provide the function call in the exact format shown below:
        {{
            "tool_calls": [{{
                "id": "unique_id",
                "type": "function",
                "function": {{
                    "name": "web_search",
                    "arguments": "{{\\"query\\": \\"search query\\"}}"
                }}
            }}]
        }}

        Example responses:
        Query: "What is the capital of France?"
        Paris is the capital of France.

        Query: "What are the latest COVID-19 statistics in Moscow?"
        {{
            "tool_calls": [{{
                "id": "covid_stats_query",
                "type": "function",
                "function": {{
                    "name": "web_search",
                    "arguments": "{{\"query\\": \\"current COVID-19 statistics Moscow\\"}}"
                }}
            }}]
        }}<|end|>
        <|user|>Query: {query}<|end|>
        <|assistant|>"""
        '''
        decision_prompt_messages = [{
        "role": "system",
        "content": """You are a query evaluator. Strictly follow these rules:
        1. Use web_search ONLY for queries requiring up-to-date info, specific details, or non-common knowledge
        2. Answer directly for common knowledge (e.g., capitals, basic facts)
        3. Responses must be either plain text or valid JSON (NO explanations or code formatting)

        Examples:
        User: Current weather in Moscow
        Assistant:
        {
            "tool_calls": [{
                "id": "weather_123",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": {
                        "query": "current weather Moscow"
                    }
                }
            }]
        }

        User: Capital of France
        Assistant: Paris is the capital of France."""
        }, {"role": "user", "content": f"Query: {query}"}]

        print("evaluating query")

        decision_response = chat_model.create_chat_completion(
            messages=decision_prompt_messages,
            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Searches the web for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use."
                            }
                        },
                        "required": ["query"]
                    }
                }
                }],
                tool_choice="auto",
                temperature=0.3,
                stop=["<|end|>", "\n\n"]
        )

        response_data = decision_response["choices"][0]["message"]

        print(response_data)

        try:
            content_text = response_data["content"]

            content_data = extract_json(content_text)
            
            if "tool_calls" in content_data:
                tool_call = content_data["tool_calls"][0]
                if tool_call["function"]["name"] == "web_search":
                    search_query = tool_call["function"]["arguments"]["query"]
                    print(f"Model decided to use web search with query: {search_query}")
                    search_results = await ddg_search(search_query, tokenizer)
                    if search_results:
                        print("Ranking snippets...")
                        top_snippets = rank_snippets(query, search_results, tokenizer)
                        context = "\n\n".join(top_snippets)
                        print("Generating answer stream with web search context")

                        return StreamingResponse(generate_answer_stream(query, context, chat_model), media_type="text/event-stream")
                    else:
                        return StreamingResponse(generate_answer_stream(query, "No relevant web content found.", chat_model), media_type="text/event-stream")
            else:
                raise ValueError

        except (ValueError, json.JSONDecodeError):
            print("Model decided to answer without web search.")
            return StreamingResponse(generate_answer_stream(query, "", chat_model), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Error during RAG processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the query")
