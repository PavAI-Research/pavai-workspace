# PavAI-At-Work
Private Seamless productivity focus AI workspace crafted for daily use, characterized by simplicity, minimalism, and effortlessly integrated multilingual communication assistance catering to professionals' needs. 


### llamacpp-python-local installation

## POETRY

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 poetry run pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

## downgrade
CMAKE_ARGS="-DLLAMA_CUBLAS=on" poetry run pip install llama-cpp-python==0.2.27 --force-reinstall --no-cache-dir


### ollama installation
poetry export --without-hashes --format=requirements.txt > requirements.txt
