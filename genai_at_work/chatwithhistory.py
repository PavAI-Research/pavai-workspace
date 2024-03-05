from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import os
import argparse
from tqdm import tqdm
import chromadb
import argparse
import os
from typing import List, Dict
from openai.types.chat import ChatCompletionMessageParam
import openai
import chromadb
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from chromadb.utils import embedding_functions
import json
from openai import OpenAI

llmclient = OpenAI(base_url="http://192.168.0.18:12345/v1", api_key="sk-1234")
# LLAMACPP 
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://192.168.0.29:8004/v1"
os.environ["MODEL_NAME"] = "openhermes-2.5-mistral-7b"

#llmclient = OpenAI(base_url="http://192.168.0.29:8004/v1", api_key="sk-1234")
#llamacpp_llm=ChatOpenAI(model_name="mistral-7b-instruct-v0.2", temperature=0.7)
#ollama_llm = Ollama(model="openhermes",base_url="http://192.168.0.18:12345")

def ingest_files(
    documents_directory: str = "workspace/documents",
    collection_name: str = "documents_collection",
    persist_directory: str = "resources/historydb",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    files = os.listdir(documents_directory)
    for filename in files:
        with open(f"{documents_directory}/{filename}", "r") as file:
            for line_number, line in enumerate(
                tqdm((file.readlines()), desc=f"Reading {filename}"), 1
            ):
                # Strip whitespace and append the line to the documents list
                line = line.strip()
                # Skip empty lines
                if len(line) == 0:
                    continue
                documents.append(line)
                metadatas.append(
                    {"filename": filename, "line_number": line_number})

    # Instantiate a persistent chroma client in the persist_directory.
    client = chromadb.PersistentClient(path=persist_directory)

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(name=collection_name)

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i: i + 100],
            documents=documents[i: i + 100],
            metadatas=metadatas[i: i + 100],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")

def ingest_workspace_data() -> None:
    # Read the data directory, collection name, and persist directory
    ingest_files(documents_directory="workspace/documents",
                 collection_name="documents_collection",
                 persist_directory="resources/historydb")

    ingest_files(documents_directory="workspace/session_logs",
                 collection_name="session_logs_collection",
                 persist_directory="resources/historydb")

def build_prompt(query: str, context: List[str]) -> List[ChatCompletionMessageParam]:
    """
    Builds a prompt for the LLM. #
    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (List[ChatCompletionMessageParam]).
    """

    system: ChatCompletionMessageParam = {
        "role": "system",
        "content": "I am going to ask you a question, which I would like you to answer"
        "based only on the provided context, and not any other information."
        "If there is not enough information in the context to answer the question,"
        'say "I am not sure", then try to make a guess.'
        "Break your answer up into nicely readable paragraphs.",
    }
    user: ChatCompletionMessageParam = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    } 

    return [system, user]

# mistral-7b-instruct-v0.2.q5_0
def get_llm_response(query: str, context: List[str], model_name: str="mistrallite") -> str:
    """
    Queries the GPT API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """
    response = llmclient.chat.completions.create(
        model=model_name,
        messages=build_prompt(query, context),
    )

    return response.choices[0].message.content  # type: ignore

def list_collections(persist_directory: str = "resources/historydb"):
    client = chromadb.PersistentClient(path=persist_directory)
    names=[]
    for col in client.list_collections():
        names.append(col.name)
    return names

def query_history(query:str, collection_name: str = "documents_collection", persist_directory: str = "resources/historydb") -> None:
    client = chromadb.PersistentClient(path=persist_directory)
    # Get the collection.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5") ## 768
    collection = client.get_collection(name=collection_name,embedding_function=sentence_transformer_ef)

    # Query the collection to get the 5 most relevant results
    results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas"])
    print(results)
    # Get the response from GPT
    response = get_llm_response(query, results["documents"][0])  # type: ignore
    # Output, with sources
    print(response)
    # print("\n")
    # print(f"Source documents:\n{sources}")
    print("\n")

if __name__ == "__main__":
   # ingest_workspace_data()
   cols = list_collections()
   print(cols)

   collection_name="web_search_collection"
   result=query_history(query="What is the best way to learn a new language?",collection_name=collection_name)
   print(result)






