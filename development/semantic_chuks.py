#pip install torch sentence-transformers

import os
import logging
import sys
import numpy as np

# Set OPENAI_API_KEY environment variable
#os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Importing required modules
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM, LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
#from llama_index.llama_pack import download_llama_pack
from llama_index.core.llama_pack import download_llama_pack
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
#from llama_index.node_parser import SentenceSplitter
#from llama_index.indices.postprocessor import SentenceTransformerRerank
# from llama_index.response.notebook_utils import display_source_node

# Download Semantic Chunking Package
download_llama_pack(
    "SemanticChunkingQueryEnginePack",
    "./semantic_chunking_pack",
    skip_load=True,
)

# Load documents from directory
documents = SimpleDirectoryReader(input_files=["essay.txt"]).load_data()

# Initialize LlamaCPP model
llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q5_K_M.gguf',
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Initialize HuggingFaceEmbedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize SentenceSplitter with baseline settings
base_splitter = SentenceSplitter(chunk_size=512)

# Initialize SentenceTransformerRerank for reranking
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=3
)

# Create ServiceContext with default settings
service_context = ServiceContext.from_defaults(
    chunk_size=512,
    llm=llm,
    embed_model=embed_model
)

# Get nodes from documents using baseline splitter
base_nodes = base_splitter.get_nodes_from_documents(documents)

# Initialize VectorStoreIndex and QueryEngine with baseline settings
base_vector_index = VectorStoreIndex(base_nodes, service_context=service_context)
base_query_engine = base_vector_index.as_query_engine(node_postprocessors=[rerank])

# Query using baseline query engine
response = base_query_engine.query(
    "Tell me about the author's programming journey through childhood to college"
)
print(str(response))

