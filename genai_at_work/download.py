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

HF_CACHE_DIR=config["HF_CACHE_DIR"] #"resources/models"

def check_and_get_downloads():
    t0 = time.perf_counter()
    with Progress(transient=True) as progress:
        task1 = progress.add_task("Downloading model files", total=100)    

        ## Download a single file
        print("MULTIMODAL")
        model_file=hf_hub_download(repo_id=config["LOCAL_MM_REPO_ID"], filename=config["LOCAL_MM_REPO_MODEL_FILE"],cache_dir=HF_CACHE_DIR)
        project_file=hf_hub_download(repo_id=config["LOCAL_MM_REPO_ID"], filename=config["LOCAL_MM_REPO_PROJECT_FILE"],cache_dir=HF_CACHE_DIR)
        print(model_file)
        print(project_file)
        progress.update(task1, advance=40)

        # snapshot_download(repo_id=config["FASTER_WHISPER"],  cache_dir=HF_CACHE_DIR)
        print("FASTER_WHISPER")
        faster_file=snapshot_download(repo_id=config["FASTER_WHISPER"],  cache_dir=HF_CACHE_DIR)
        print(faster_file)
        progress.update(task1, advance=20)

        print("DISTIL_WHISPER")
        distil_file=snapshot_download(repo_id=config["DISTIL_WHISPER"],  cache_dir=HF_CACHE_DIR)
        print(distil_file)
        progress.update(task1, advance=20)    

        ##  Download repository and save locally
        if not eval(config["SKIP_LANG_TRANSLATOR"]):
            print("Download LANG_TRANSLATOR")
            lang_file=snapshot_download(repo_id=config["LANG_TRANSLATOR"],  cache_dir=HF_CACHE_DIR)
            print(lang_file)
            progress.update(task1, advance=20)
        else:
            print("Skipping LANG_TRANSLATOR")        

    took = (time.perf_counter()-t0)
    print(f"Download completed. files saved to {HF_CACHE_DIR}")
    print(f"it took {took:.2f} seconds")

## run health check
check_and_get_downloads()
#./models/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/c30ab8669e9d6b14105d0652a015ad94ad54ef72/llava-v1.6-mistral-7b.Q4_K_M.gguf
#./models/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/c30ab8669e9d6b14105d0652a015ad94ad54ef72/mmproj-model-f16.gguf
