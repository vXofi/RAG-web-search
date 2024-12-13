import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from duckduckgo_search import DDGS
from playwright.async_api import async_playwright
import torch
import re
import os
from llama_cpp import Llama
from datetime import datetime
import asyncio


"""
TODO:
handle accuweather (???)
handle retrieved context length > model context length
maybe change input
length in tokens is different depending on used language
phi 3.5 is probably too bad for multilanguage tasks
preserve structure of retrieved pages
add proper prompt
problem with context for multiple questions
test markdown formatting
"""


headers = {"user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}


async def ddg_search(query, tokenizer):
    results = DDGS(headers=headers).text(query, max_results=5)
    urls = [result['href'] for result in results]

    docs = await get_page(urls)

    content = []

    print("query: ", query)
    print("urls: ", urls)

    for doc in docs:
        page_text = process_page_content(doc)
        text = text_to_chunks(page_text, tokenizer)
        content.extend(text)

    return content


async def get_page(urls):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
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
        
        await browser.close()

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


def text_to_chunks(text, tokenizer, chunk_size=256, overlap_size=64):
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


def rank_snippets(query, snippets, tokenizer, model_name="all-MiniLM-L6-v2", top_k=3):
    embedder = SentenceTransformer(model_name)

    text_chunks = [tokenizer.decode(chunk) for chunk in snippets]

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    snippet_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, snippet_embeddings).squeeze(0)

    top_indices = similarities.squeeze(0).argsort(descending=True)
    top_snippets = [text_chunks[idx] for idx in top_indices[:top_k]]
    
    return top_snippets


def generate_answer(query, context, model,
                             max_tokens=1024,
                             temperature = 0.7,
                             top_p = 0.1,
                             echo = False,
                             stop = ["User:"]):
        context = f"Current date (Year/Month/Day) is:{datetime.now().year}/{datetime.now().month}/{datetime.now().day}" + context
        
        input_text = f"User: {query}\nContext: {context}\nAssistant:"

        response_stream = model(
            input_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
            stream=True)
        
        output=""

        print("Assistant: ", end="")
        for chunk in response_stream:

            token = chunk["choices"][0]["text"]

            print(token, end="", flush=True)
            output += token

        print()
        return output

async def main():
    print("Loading model...")
    # model_name = "SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored_FP8"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    my_model_path = "./model/Phi-3.5-mini-instruct-Q4_K_L.gguf"
    CONTEXT_SIZE = 1024


    chat_model = Llama(model_path=my_model_path,
                       n_ctx=CONTEXT_SIZE, n_threads=6)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct") 

    query = input("Enter your question: ")
    while query != 'break':

        print("Retrieving snippets...")
        search_results = await ddg_search(query, tokenizer)

        print("Ranking snippets...")
        top_snippets = rank_snippets(query, search_results, tokenizer)

        context = "".join(top_snippets)
        print("context: ", context)

        print("Generating answer...")
        answer = generate_answer(query, context, chat_model)
        query = input("Enter your question: ")
    
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())