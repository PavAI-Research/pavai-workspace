import chatllamacpp
import chatollama
import chatopenai
import gradio as gr
import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}


def api_calling_v2(
    api_host: str = None,
    api_key: str = "EMPTY",
    active_model: str = "zephyr:latest",
    prompt: str = None,
    history: list = None,
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0
):
    provider = config["LLM_PROVIDER"]
    match provider:
        case "all-in-one":
            # llamacpp local
            reply_text, reply_messages, reply_status = chatllamacpp.LlamaCppLocal().api_calling_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
        case "ollama":
            reply_text, reply_messages, reply_status = chatollama.api_calling_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
        case "openai":
            reply_text, reply_messages, reply_status = chatopenai.api_calling_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            ...
        case _:
            raise ValueError(f"Untested LLM provider yet {provider}")

    return reply_text, reply_messages, reply_status


def message_and_history_v2(
    api_host: str = None,
    api_key: str = "EMPTY",
    active_model: str = "zephyr:latest",
    prompt: str = None,
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
    provider = config["LLM_PROVIDER"]
    match provider:
        case "all-in-one":
            # llamacpp local
            chatbot, output_messages, output_status = chatllamacpp.LlamaCppLocal().message_and_history_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
        case "ollama":
            chatbot, output_messages, output_status = chatollama.message_and_history_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
        case "openai":
            chatbot, output_messages, output_status = chatopenai.message_and_history_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            ...
        case _:
            raise ValueError(f"Untested LLM provider yet {provider}")

    return chatbot, output_messages, output_status


def list_models():
    provider = config["LLM_PROVIDER"]
    match provider:
        case "all-in-one":
            # llamacpp local
            model_names = chatllamacpp.LlamaCppLocal().list_models()
        case "ollama":
            model_names = chatollama.list_models()
        case "openai":
            model_names = chatopenai.list_models()
        case _:
            raise ValueError(f"Untested LLM provider yet {provider}")

    return model_names


def upload_image(
        api_host: str = None,
        api_key: str = "EMPTY",
        active_model: str = "llava:7b-v1.6-mistral-q5_0",
        user_prompt: str = None,
        image=None,
        chatbot: list = None,
        history: list = None,
        system_prompt: list = None):
    provider = config["LLM_PROVIDER"]
    match provider:
        case "all-in-one":
            # llamacpp local
            chatbot, messages = chatllamacpp.LlamaCppLocal().upload_image(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                user_prompt=user_prompt,
                image=image,
                chatbot=chatbot,
                history=history,
                system_prompt=system_prompt
            )
        case "ollama":
            chatbot, messages = chatollama.upload_image(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                user_prompt=user_prompt,
                image=image,
                chatbot=chatbot,
                history=history,
                system_prompt=system_prompt
            )
        case "openai":
            chatbot, messages = chatopenai.upload_image(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                user_prompt=user_prompt,
                image=image,
                chatbot=chatbot,
                history=history,
                system_prompt=system_prompt
            )
        case _:
            raise ValueError(f"Untested LLM provider yet {provider}")

    return chatbot, messages
