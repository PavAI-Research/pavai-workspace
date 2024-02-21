### keep dataset history ###
# pip install chromadb
# pip install sentence_transformers
import time
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

default_ef = embedding_functions.DefaultEmbeddingFunction() ## 384
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
) ## 768

historydb = "./historydb"
settings = Settings(allow_reset=True)
client = chromadb.PersistentClient(path=historydb,settings=settings)

upload_file_collection = client.get_or_create_collection(
    name="upload_files_collection",
    metadata={"hnsw:space": "cosine"},  # l2 is the default
    embedding_function=sentence_transformer_ef
)

upload_image_collection = client.get_or_create_collection(
    name="upload_image_collection",
    metadata={"hnsw:space": "cosine"},  
    embedding_function=sentence_transformer_ef    
)

upload_audio_collection = client.get_or_create_collection(
    name="upload_audio_collection",
    metadata={"hnsw:space": "cosine"},  
    embedding_function=sentence_transformer_ef    
)

upload_video_collection = client.get_or_create_collection(
    name="upload_video_collection",
    metadata={"hnsw:space": "cosine"}, 
    embedding_function=sentence_transformer_ef    
)

youtube_video_collection = client.get_or_create_collection(
    name="youtube_video_collection",
    metadata={"hnsw:space": "cosine"}, 
    embedding_function=sentence_transformer_ef
)

web_search_collection = client.get_or_create_collection(
    name="web_search_collection", metadata={"hnsw:space": "cosine"},
    embedding_function=sentence_transformer_ef 
)

web_scrapping_collection = client.get_or_create_collection(
    name="web_scrapping_collection",
    metadata={"hnsw:space": "cosine"}, 
    embedding_function=sentence_transformer_ef    
)


def reset():
    global client
    """# Empties and completely resets the database. ⚠️ This is destructive and not reversible."""
    client.reset()
    historydb = "./historydb"
    settings = Settings(allow_reset=True)
    client = chromadb.PersistentClient(path=historydb,settings=settings)    

def list_collections():
    colnames=[]
    cols=client.list_collections()
    for col in cols:
        colnames.append(col.name)
    return colnames

def get_collection(colname: str):
    col = client.get_collection(colname, embedding_function=sentence_transformer_ef) 
    return col   

def add_collection_record(colname: str, weburl: str, filecontent: str):
    col = client.get_collection(colname, embedding_function=sentence_transformer_ef)        
    col.upsert(
        documents=[filecontent],
        metadatas=[{"name": str(weburl)}],
        ids=[str(weburl)],
    )

def query_collection(colname: str,query: str, n_results:int=5,where:dict={}, where_document:dict={}):
    col = client.get_collection(colname, embedding_function=sentence_transformer_ef)    
    records=col.query(
        query_texts=[query],
        n_results=5,
        where=where,
        where_document=where_document
    )
    i=0
    result=[]
    for i in range(len(records["documents"])):
        result.append([records["ids"][i],records["documents"][i]])
    return result

def peek_collection(colname: str, limit:int=10):
    col = client.get_collection(colname,embedding_function=sentence_transformer_ef)
    records=col.peek(limit)
    i=0
    result=[]
    for i in range(len(records["documents"])):
        result.append([records["ids"][i],records["documents"][i]])
    return result

def get_document(colname: str, docid:str, chatbot:list=[],history:list=[]):
    t0=time.perf_counter()    
    col = client.get_collection(colname,embedding_function=sentence_transformer_ef)
    records = col.get(ids=[docid])
    result=[]
    for i in range(len(records["documents"])):
        result.append([records["ids"][i],records["documents"][i]])

    t1=time.perf_counter()            
    took=(t1-t0)    
    chatbot=[] if chatbot is None else chatbot  
    chatbot.append((f"document id:{docid}",result[0][1]))
    ## update history
    history=[] if history is None else history    
    history.append({"role": "user", "content": f"document id:{docid}\n{result[0][1]}"})        
    print(f"get_document took {took}s")
    return chatbot, history 


if __name__=="__main__":
   # client.reset()
   cols=list_collections()
   print(cols)
#    val=sentence_transformer_ef("hello world")
#    print(len(val))
   #print(cols)
#    add_collection_record("web_scrapping_collection","google.com","what the world is godzilla")
#    add_collection_record("web_scrapping_collection","test.com","hello world, what is the world happening")   
#    res=query_collection(colname="web_scrapping_collection",query="happy")
#    print(res[0][0])
#    print(res[0][1])

   res=peek_collection(colname="web_scrapping_collection")
   print(res[1][0])
   print(res[1][1])


