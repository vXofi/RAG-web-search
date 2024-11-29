import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from duckduckgo_search import DDGS
import torch
import re
import os
from llama_cpp import Llama


"""
TODO:
handle accuweather (???)
handle retrieved context length > model context length
add ranking of smaller fragments over entire pages
maybe fix input

"""


headers = {"User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}
# performs DuckDuckGo search, urls are extracted and status checked
# 
def ddg_search(query):
    results = DDGS().text(query, max_results=8)
    urls = []
    for result in results:
        url = result['href']
        urls.append(url)

    docs = get_page(urls)

    content = []
    for doc in docs:
        page_text = re.sub("\n\n+", "\n", doc.page_content)
        text = truncate(page_text)
        content.append(text)

    return content

# retrieves pages and extracts text by tag
def get_page(urls):
    loader = AsyncChromiumLoader(urls, headless=True, user_agent=headers["User-agent"])
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"], remove_unwanted_tags=["a"])

    return docs_transformed

# helper function to reduce the amount of text
def truncate(text):
    words = text.split()
    truncated = " ".join(words[:100])

    return truncated


# === Step 2: Retriever Model for Ranking ===
def rank_snippets(query, snippets, model_name="all-MiniLM-L6-v2", top_k=3):
    """
    Rank snippets by relevance to the query using embeddings.
    
    Args:
        query (str): User's query.
        snippets (list): List of text snippets.
        model_name (str): Pre-trained model for embeddings.
        top_k (int): Number of top relevant snippets to return.
    
    Returns:
        list: Top-K relevant snippets.
    """
    # Load the embedding model
    embedder = SentenceTransformer(model_name)

    # Compute embeddings for the query and snippets
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    snippet_embeddings = embedder.encode(snippets, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, snippet_embeddings).squeeze(0)

    # Rank snippets by similarity
    ranked_indices = torch.argsort(similarities, descending=True)
    top_snippets = [snippets[idx] for idx in ranked_indices[:top_k]]
    
    return top_snippets

# === Step 3: Generator (Language Model) ===
def generate_answer(query, context, model,
                             max_tokens=1024,
                             temperature = 0.7,
                             top_p = 0.1,
                             echo = False,
                             stop = ["User:"]):
        
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

# === Step 4: Unified Pipeline ===
def main():
    # Load the model (Phi-3.5 in this example)
    print("Loading model...")
    # model_name = "SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored_FP8"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    my_model_path = "./model/Phi-3.5-mini-instruct-Q4_K_L.gguf"
    CONTEXT_SIZE = 2048


    chat_model = Llama(model_path=my_model_path,
                       n_ctx=CONTEXT_SIZE, n_threads=6)

    # Get user query
    query = input("Enter your question: ")

    # Retrieve snippets from Google search
    print("Retrieving snippets...")
    search_results = ddg_search(query)


    # Rank snippets using the retriever model
    print("Ranking snippets...")
    top_snippets = rank_snippets(query, search_results)
    context = "\n".join(top_snippets)

    print(context)

    # Generate an answer using the model
    print("Generating answer...")
    answer = generate_answer(query, context, chat_model)
    
    # Display the answer
    print("\n=== Answer ===")
    print(answer)

# Run the unified pipeline
if __name__ == "__main__":
    main()