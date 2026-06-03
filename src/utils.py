from typing import Any, List, Dict
import random
import time
import os
import logging
import copy
import string
import requests
logging.basicConfig(level=logging.INFO)

OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3:8b')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


class Utils:
    punctuations = set(string.punctuation)

    @classmethod
    def is_chat(cls, model: str):
        # All Ollama models use chat interface
        return True

    @classmethod
    def is_code(cls, model: str):
        return 'code' in model

    @classmethod
    def no_stop(cls, model: str):
        # Ollama models use chat interface
        return True


class NoKeyAvailable(Exception):
    pass


def retry_with_exponential_backoff(
    func,
    max_reqs_per_min: int = 0,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors_to_catch: tuple = (requests.exceptions.ConnectionError, requests.exceptions.Timeout, NoKeyAvailable),
    errors_to_raise: tuple = (),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        num_retries = 0

        while True:
            try:
                start_t = time.time()
                logging.info('Ollama API call start')
                results = func(*args, **kwargs)
                logging.info('Ollama API call end')
                return results

            except errors_to_catch as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f'maximum number of retries ({max_retries}) exceeded.')
                delay *= exponential_base * (1 + jitter * random.random())
                logging.info(f'retry on {e}, sleep for {delay}')
                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper


def _ollama_chat(messages, model=None, temperature=0.0, max_tokens=2048, stop=None, **kwargs):
    """Call Ollama chat API."""
    model = model or OLLAMA_MODEL
    url = f'{OLLAMA_BASE_URL}/api/chat'

    options = {
        'temperature': temperature,
        'num_predict': max_tokens,
    }
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        options['stop'] = stop

    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'options': options,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    # Return in a format compatible with the rest of the codebase
    return {
        'choices': [{
            'message': {'content': result['message']['content']},
            'finish_reason': 'stop' if result.get('done', True) else 'length',
        }],
        'model': model,
    }


def _ollama_generate(prompt, model=None, temperature=0.0, max_tokens=2048, stop=None, **kwargs):
    """Call Ollama generate API for completion-style requests."""
    model = model or OLLAMA_MODEL
    url = f'{OLLAMA_BASE_URL}/api/generate'

    options = {
        'temperature': temperature,
        'num_predict': max_tokens,
    }
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        options['stop'] = stop

    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': options,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    text = result.get('response', '')
    return {
        'choices': [{
            'text': text,
            'logprobs': {'tokens': [], 'token_logprobs': [], 'text_offset': []},
            'finish_reason': 'stop' if result.get('done', True) else 'length',
        }],
        'model': model,
    }


@retry_with_exponential_backoff
def openai_api_call(*args, **kwargs):
    """Unified API call that routes to Ollama."""
    # Remove keys not needed for Ollama
    kwargs.pop('api_key', None)
    kwargs.pop('logit_bias', None)
    kwargs.pop('frequency_penalty', None)
    kwargs.pop('top_p', None)
    kwargs.pop('logprobs', None)
    kwargs.pop('echo', None)

    model = kwargs.pop('model', OLLAMA_MODEL)
    temperature = kwargs.pop('temperature', 0.0)
    max_tokens = kwargs.pop('max_tokens', 2048)
    stop = kwargs.pop('stop', None)

    if 'messages' in kwargs:
        messages = kwargs.pop('messages')
        if len(messages) <= 0:
            return []
        if isinstance(messages[0], list):  # batch request
            results = []
            for msg_list in messages:
                result = _ollama_chat(
                    msg_list, model=model, temperature=temperature,
                    max_tokens=max_tokens, stop=stop)
                results.append(result)
            return results
        else:
            return _ollama_chat(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens, stop=stop)
    elif 'prompt' in kwargs:
        prompts = kwargs.pop('prompt')
        if isinstance(prompts, str):
            prompts = [prompts]
        all_choices = []
        for prompt in prompts:
            result = _ollama_generate(
                prompt, model=model, temperature=temperature,
                max_tokens=max_tokens, stop=stop)
            all_choices.extend(result['choices'])
        return {'choices': all_choices, 'model': model}
    else:
        raise ValueError('Either "messages" or "prompt" must be provided')
