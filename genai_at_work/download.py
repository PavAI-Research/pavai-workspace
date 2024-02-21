import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

from huggingface_hub import hf_hub_url
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import time
from rich.progress import Progress

HF_CACHE_DIR="./models"

with Progress(transient=True) as progress:
    task1 = progress.add_task("Downloading model files", total=100)    
    ##  Download repository and save locally
    # print("LANG_TRANSLATOR")
    # lang_file=snapshot_download(repo_id=config["LANG_TRANSLATOR"],  cache_dir=HF_CACHE_DIR)
    # print(lang_file)
    # progress.update(task1, advance=20)

    # print("DISTIL_WHISPER")
    # distil_file=snapshot_download(repo_id=config["DISTIL_WHISPER"],  cache_dir=HF_CACHE_DIR)
    # print(distil_file)
    # progress.update(task1, advance=20)    

    # snapshot_download(repo_id=config["FASTER_WHISPER"],  cache_dir=HF_CACHE_DIR)
    print("FASTER_WHISPER")
    faster_file=snapshot_download(repo_id=config["FASTER_WHISPER"],  cache_dir=HF_CACHE_DIR)
    print(faster_file)
    progress.update(task1, advance=20)

    ## Download a single file
    print("MULTIMODAL")
    model_file=hf_hub_download(repo_id=config["HF_MM_REPO_ID"], filename=config["HF_MM_REPO_MODEL_FILE"],cache_dir=HF_CACHE_DIR)
    project_file=hf_hub_download(repo_id=config["HF_MM_REPO_ID"], filename=config["HF_MM_REPO_PROJECT_FILE"],cache_dir=HF_CACHE_DIR)
    print(model_file)
    print(project_file)
    progress.update(task1, advance=40)

print(f"Download completed. saved files to {HF_CACHE_DIR}")

#./models/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/c30ab8669e9d6b14105d0652a015ad94ad54ef72/llava-v1.6-mistral-7b.Q4_K_M.gguf
#./models/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/c30ab8669e9d6b14105d0652a015ad94ad54ef72/mmproj-model-f16.gguf
##

