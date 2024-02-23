## Ollama Setup -- latest version with openai API support
## see https://ollama.com/download

## Linux
```
curl -fsSL https://ollama.com/install.sh | sh
```

## Mac
```
https://ollama.com/download/mac
```

## Windows
```
https://ollama.com/download/windows
```

## pull models

```
## zephyr model 7b size=4.1 gb
curl http://localhost:11434/api/pull -d '{
  "name": "zephyr"
}'

## mistral model 7b size=4.1 gb
curl http://localhost:11434/api/pull -d '{
  "name": "mistral"
}'

## smaller model 3b size=1.6 gb
curl http://localhost:11434/api/pull -d '{
  "name": "stablelm-zephyr"
}'

```
## verify models pull to local machine
```
curl http://localhost:11434/api/tags

```
## run a test call
```
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "options": {
    "seed": 101,
    "temperature": 0
  }
}'
```
