## pip install llama-cpp-python
#CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
#pip install llama-cpp-python  --upgrade --force-reinstall --no-cache-dir
import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
from rich import pretty
pretty.install()

print("MULTIMODAL")
def download_models():
    HF_CACHE_DIR=config["HF_CACHE_DIR"]
    model_file=hf_hub_download(repo_id=config["HF_MM_REPO_ID"], filename=config["HF_MM_REPO_MODEL_FILE"],cache_dir=HF_CACHE_DIR)
    project_file=hf_hub_download(repo_id=config["HF_MM_REPO_ID"], filename=config["HF_MM_REPO_PROJECT_FILE"],cache_dir=HF_CACHE_DIR)
    print(model_file)
    print(project_file)
    return model_file, project_file

def load_llm(model_file,project_file):
    chat_handler = Llava15ChatHandler(clip_model_path=project_file)
    llm = Llama(
        model_path=model_file,
        chat_handler=chat_handler,
        n_ctx=4096, # n_ctx should be increased to accomodate the image embedding
        logits_all=True,# needed to make llava work
        n_gpu_layers=-1,
        n_threads=8
    )

def chat(user_prompt:str, system_prompt:str):
    model_file, project_file = download_models()
    llm = load_llm(model_file, project_file)
    result = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return result

def mmchat(user_prompt:str,image_url:str,system_prompt:str):
    model_file, project_file = download_models()
    llm = load_llm(model_file, project_file)    
    ## "file:///home/pop/software_engineer/chatbotatwork/samples/Bill_Image_Receipt.png"
    result= llm.create_chat_completion(
        messages = [
            {"role": "system", "content": system_prompt},
            {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type" : "text", "text": user_prompt}
                ]
            }
        ])
    return result 

# print(result)

# #mmchat()
# result=llm.create_chat_completion(
# messages = [
#     {"role": "system", "content": "You are an assistant who perfectly describes images and helpful assistant that outputs in JSON."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": "file:///home/pop/software_engineer/chatbotatwork/samples/Bill_Image_Receipt.png"}},
#             {"type" : "text", "text": "Describe this image in detail in json schema format."}
#         ]
#     },

# ],
# response_format={
#         "type": "json_object",
# },
# temperature=0.2
# )

# print(result)
#-----------
# llm.create_chat_completion(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that outputs in JSON.",
#         },
#         {"role": "user", "content": "Who won the world series in 2020"},
#     ],

# messages=[
# {
#             "role": "system",
#             "content": "You are a helpful assistant that outputs in JSON.",
# },
#         {"role": "user", "content": "Who won the world series in 2020"},
# ],
# response_format={
#         "type": "json_object",
# },
# #temperature=0.7
# #)
