# pip install ollama
# pip install gradio
import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}
import base64
import time
import random
#from openai import OpenAI
import ollama
from ollama import Client
import asyncio
from ollama import AsyncClient

from ollama._types import Message, Options, RequestError, ResponseError
from typing import Any, AnyStr, Union, Optional, Sequence, Mapping, Literal
import tiktoken

#openai.api_key = "YOUR_API_KEY"
#prompt = "Enter Your Query Here"
LLM_PROVIDER=config["LLM_PROVIDER"]

API_HOST=config["OLLAMA_API_HOST"]
client = Client(host=API_HOST)
asclient = AsyncClient(host=API_HOST)

# client = Client(host='http://192.168.0.18:12345')
# asclient = AsyncClient(host='http://192.168.0.18:12345')


# def add_messages(history:list=None,system_prompt:str=None, 
#                  user_prompt:str=None,ai_prompt:str=None,image_list:list=None):
#     messages=[]
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     if user_prompt:
#         messages.append({"role": "user", "content": user_prompt})    
#     if ai_prompt:
#         messages.append({"role": "assistant", "content": ai_prompt})  
#     if image_list:
#         messages.append({
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": image_list[0]
#                     },
#                 },
#                 {"type": "text", "text": user_prompt},
#             ],})            
#     if history:
#         return history+messages
#     else:
#         return messages

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_messages_token(messages:list):
    if messages is None:
        return 0    
    if isinstance(messages, str):
        content=" ".join(messages)
    else:
        for m in messages:
            content=content+str(m) 
    return num_tokens_from_string(content)

def add_messages(history:list=None,system_prompt:str=None, 
                 user_prompt:str=None,ai_prompt:str=None,image_list:list=None):
    messages=[]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})    
    if ai_prompt:
        messages.append({"role": "assistant", "content": ai_prompt})  
    if image_list:
        messages.append({"role": "user","content": user_prompt,'images': image_list})    
    if history:
        return history+messages
    else:
        return messages

def embeddings(model:str,prompt:str):
    return client.embeddings(model=model, prompt=prompt)

def list_models(model):
    return client.list()

def pull_models(model):
    return client.pull(model)

async def async_api_calling(prompt:str, history:list=None,
                model:str="zephyr",
                stream: bool = False,
                format: Literal['', 'json'] = '',
                options: Options | None = None,
                keep_alive: float | str | None = None): 
    
    messages = add_messages(user_prompt=prompt, history=history)

    response = await asclient.chat(
        model=model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
        )
    # response = client.chat(
    #     model=model, 
    #     messages=messages,
    #     stream=stream,format=format,options=options,keep_alive=keep_alive
    # )    
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages

def api_calling(prompt:str, history:list=None,
                model:str="zephyr",
                stream: bool = False,
                format: Literal['', 'json'] = '',
                options: Options | None = None,
                keep_alive: float | str | None = None): 
    
    messages = add_messages(user_prompt=prompt, history=history)
    response = client.chat(
        model=model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
    )    
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages

def message_and_history(input, chatbot, history): 
    history = history or [] 
    print(history) 
    s = list(sum(history, ())) 
    print(s) 
    s.append(input) 
    print('#########################################') 
    print(s) 
    prompt = ' '.join(s) 
    print(prompt) 
    ##output, output_messages = api_calling(prompt,history) 
    output, output_messages = asyncio.run(async_api_calling(prompt,history))      
    history.append((input, output)) 
    print('------------------') 
    print(history)
    print("*********************")     
    print(output_messages) 
    print("*********************") 
    return history, history

def api_calling_v2(
        api_host:str=None,
        api_key:str="EMPTY",
        active_model:str="zephyr:latest",    
        user_prompt:str=None,         
        history:list=None,
        system_prompt:list=None,         
        stream: bool = False,
        raw: bool = False,
        format: Literal['', 'json'] = '',
        options: Options | None = None,
        keep_alive: float | str | None = None): 
    
    if api_host is not None:
        client = Client(host=api_host)
        print(f"Use method API host: {api_host} and model: {active_model}")

    messages = add_messages(user_prompt=user_prompt,system_prompt=system_prompt, history=history)
    t0=time.perf_counter()
    response = client.chat(
        model=active_model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
    )  
    t1=time.perf_counter()
    took_time = t1-t0    
    #reply_status = f"<p align='right'>api done: {response['done']}. It took {took_time:.2f}s</p>"   
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages, response['done']

def message_and_history_v2(
    api_host:str=None,
    api_key:str="EMPTY",
    active_model:str="zephyr:latest",           
    user_prompt: str=None,
    chatbot: list = [],
    history: list = [],
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0,
):
    t0=time.perf_counter()
    chatbot = chatbot or []
    print("#########################################")
    print(user_prompt)
    if isinstance(stop_words, str):
        stop_words=[stop_words]
    else:
        stop_words=["\n", "user:"]        

    options={
        # "num_keep": 5,
        "seed": 228,
        # "num_predict": 1,
        # "top_k": 20,
        "top_p": top_p,
        # "tfs_z": 0.5,
        # "typical_p": 0.7,
        # "repeat_last_n": 33,
        "temperature": temperature,
        # "repeat_penalty": 1.2,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        # "mirostat": 1,
        # "mirostat_tau": 0.8,
        # "mirostat_eta": 0.6,
        # "penalize_newline": True,
        "stop": stop_words,
        "numa": False,
        "num_ctx": max_tokens,
        # "num_batch": 2,
        # "num_gqa": 1,
        # "num_gpu": 1,
        # "main_gpu": 0,
        # "low_vram": False,
        # "f16_kv": True,
        # "vocab_only": False,
        # "use_mmap": True,
        # "use_mlock": False,
        # "embedding_only": False,
        # "rope_frequency_base": 1.1,
        # "rope_frequency_scale": 0.8,
        # "num_thread": 8
    }
    output, output_messages, output_status = api_calling_v2(
        api_host=api_host,
        api_key=api_key,
        active_model=active_model,            
        user_prompt=user_prompt,
        history=history,
        system_prompt=system_prompt,
        options=options
    )
    chatbot.append((user_prompt, output))
    print("------------------")
    print(chatbot)
    print("*********************")
    print(output_messages)
    print("*********************")
    tokens = count_messages_token(history)
    t1=time.perf_counter()
    took=(t1-t0)    
    output_status=f"<i>tokens:{tokens} | api status: {output_status} | took {took:.2f} seconds</i>"
    return chatbot, output_messages, output_status

def query_the_image(client, query: str, image_list: list[str], selected_model:str="llava:latest") -> ollama.chat:
    try:
        res = client.chat(
            model=selected_model,
            messages=[
                {
                'role': 'user',
                'content': query,
                'images': image_list,
                }
            ]
        )
    except Exception as e:
        print(f"Error: {e}")
        return None
    return res['message']['content']

def upload_image(
        api_host:str=None,
        api_key:str="EMPTY",
        active_model:str="llava:7b-v1.6-mistral-q5_0",    
        user_prompt:str=None,         
        image=None,
        chatbot:list=None,                 
        history:list=None,
        system_prompt:list=None): 
    
    if image is None:
        print("image removed!")
        return

    if api_host is not None:
        client = Client(host=api_host)
        print(f"Use method API host: {api_host} and model: {active_model}")

    user_prompt=user_prompt.strip()
    if len(user_prompt)==0 and image is not None:
        user_prompt="what is this image about?"

    messages = add_messages(user_prompt=user_prompt,system_prompt=system_prompt, history=history, image_list=[image])
    t0=time.perf_counter()
    response = client.chat(
        model=active_model, 
        messages=messages
    )  
    t1=time.perf_counter()
    took_time = t1-t0    
    reply_status = f"<p align='right'>api done: {response['done']}. It took {took_time:.2f}s</p>"   
    reply_text = response["message"]["content"] 
    #reply_messages = add_messages(ai_prompt=reply_text, history=messages)  
    chatbot.append((image, reply_text))
    return chatbot, messages

# upload_websearch=None,
# upload_weburl=None,
# upload_youtube_url=None,
# upload_file=None,
# upload_image=None,
# upload_video=None,
# upload_audio=None

def message_and_history_v3(
    api_host:str=None,
    api_key:str="EMPTY",
    active_model:str="zephyr:latest",           
    user_prompt: str=None,
    chatbot: list = [],
    history: list = [],
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0
):
    t0=time.perf_counter()
    chatbot = chatbot or []
    print("#########################################")
    print(user_prompt)
    if isinstance(stop_words, str):
        stop_words=[stop_words]
    else:
        stop_words=["\n", "user:"]        

    options={
        "seed": 228,
        "top_p": top_p,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop_words,
        "numa": False,
        "num_ctx": max_tokens,
    }
    output, output_messages, output_status = api_calling_v2(
        api_host=api_host,
        api_key=api_key,
        active_model=active_model,            
        user_prompt=user_prompt,
        history=history,
        system_prompt=system_prompt,
        options=options
    )
    chatbot.append((user_prompt, output))
    print("------------------")
    print(chatbot)
    print("*********************")
    print(output_messages)
    print("*********************")
    tokens = count_messages_token(history)
    t1=time.perf_counter()
    took=(t1-t0)    
    output_status=f"<i>tokens:{tokens} api status: {output_status} took {took:.2f}s</i>"
    return chatbot, output_messages, output_status    


## Converting HTML to Markdown
# pandoc --to=plain --from=html --output=files/linkedin.txt files/linkedin.html
# Image to Base 64 Converter
def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

# def segment_text(text: str, pattern: str) -> T.Iterator[tuple[str, str]]:
#     import re
#     """Segment the text in title and content pair by pattern."""
#     splits = re.split(pattern, text, flags=re.MULTILINE)
#     pairs = zip(splits[1::2], splits[2::2])
#     return pairs

# segments_h1 = segment_text(text=text, pattern=r"^# (.+)")
# segments_h2 = segment_text(text=h1_text, pattern=r"^## (.+)")

# def import_file(
#     file: T.TextIO,
#     collection: lib.Collection,
#     encoding_function: T.Callable,
#     max_output_tokens: int = lib.ENCODING_OUTPUT_LIMIT,
# ):
#     """Import a markdown file to a database collection."""
#     text = file.read()
#     filename = file.name
#     segments_h1 = segment_text(text=text, pattern=r"^# (.+)")
#     for h1, h1_text in segments_h1:
#         segments_h2 = segment_text(text=h1_text, pattern=r"^## (.+)")
#         for h2, content in segments_h2:
#             id_ = f"{filename} # {h1} ## {h2}"  # unique doc id
#             document = f"# {h1}\n\n## {h2}\n\n{content.strip()}"
#             metadata = {"filename": filename, "h1": h1, "h2": h2}
#             collection.add(ids=id_, documents=document, metadatas=metadata)

# PROMPT_CONTEXT = """
# You are Fmind Chatbot, specialized in providing information regarding Médéric Hurier's (known as Fmind) professional background.
# Médéric is an MLOps engineer based in Luxembourg. He is currently working at Decathlon. His calendar is booked until the conclusion of 2024.
# Your responses should be succinct and maintain a professional tone. If inquiries deviate from Médéric's professional sphere, courteously decline to engage.

# You may find more information about Médéric below (markdown format):
# """

# def answer(message: str, history: list[str]) -> str:
#     """Answer questions about my resume."""
#     # counters
#     n_tokens = 0
#     # messages
#     messages = []
#     # - context
#     n_tokens += len(ENCODING(PROMPT_CONTEXT))
#     messages += [{"role": "system", "content": PROMPT_CONTEXT}]
#     # - history
#     for user_content, assistant_content in history:
#         n_tokens += len(ENCODING(user_content))
#         n_tokens += len(ENCODING(assistant_content))
#         messages += [{"role": "user", "content": user_content}]
#         messages += [{"role": "assistant", "content": assistant_content}]
#     # - message
#     n_tokens += len(ENCODING(message))
#     messages += [{"role": "user", "content": message}]
#     # database
#     results = COLLECTION.query(query_texts=message, n_results=QUERY_N_RESULTS)
#     distances, documents = results["distances"][0], results["documents"][0]
#     for distance, document in zip(distances, documents):
#         # - distance
#         if distance > QUERY_MAX_DISTANCE:
#             break
#         # - document
#         n_document_tokens = len(ENCODING(document))
#         if (n_tokens + n_document_tokens) >= PROMPT_MAX_TOKENS:
#             break
#         n_tokens += n_document_tokens
#         messages[0]["content"] += document
#     # response
#     api_response = MODEL(messages=messages)
#     content = api_response["choices"][0]["message"]["content"]
#     # return
#     return content
