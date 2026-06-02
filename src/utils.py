from typing import Any, List, Dict
import random
import time
import os
import logging
import copy
import string
import asyncio
import concurrent.futures
import json
import urllib.request
import urllib.error
logging.basicConfig(level=logging.INFO)

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', OLLAMA_MODEL)
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '120'))
OPENAI_MODEL_ALIASES = {
    'code-davinci-002',
    'text-davinci-002',
    'text-davinci-003',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo',
}


class Utils:
    punctuations = set(string.punctuation)

    @classmethod
    def is_chat(cls, model: str):
        return 'turbo' in model

    @classmethod
    def is_code(cls, model: str):
        return 'code' in model

    @classmethod
    def no_stop(cls, model: str):
        return 'turbo' in model


class NoKeyAvailable(Exception):
    pass


def retry_with_exponential_backoff(
    func,
    max_reqs_per_min: int = 0,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors_to_catch: tuple = (urllib.error.URLError, TimeoutError, ConnectionError, NoKeyAvailable),
    errors_to_raise: tuple = (),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # initialize variables
        is_code_model = Utils.is_chat(kwargs['model'])
        mrpm = max_reqs_per_min
        mrpm = mrpm or (15 if is_code_model else 1000)
        const_delay = 60 / mrpm
        delay = initial_delay
        num_retries = 0

        # loop until a successful response or max_retries is hit or an exception is raised
        while True:
            # initialize key-related variables
            api_key = get_key_func = return_key_func = None
            forbid_key = False

            try:
                # get key
                _kwargs = copy.deepcopy(kwargs)
                if 'api_key' in kwargs:
                    ori_api_key = kwargs['api_key']
                    if type(ori_api_key) is tuple:  # get a key through a call
                        get_key_func, return_key_func = ori_api_key
                        api_key = get_key_func()
                    else:  # a specified key
                        api_key = ori_api_key
                    _kwargs['api_key'] = api_key

                # query API
                start_t = time.time()
                logging.info('API call start')
                results = func(*args, **_kwargs)
                logging.info('API call end')
                return results

            # retry on specific errors
            except errors_to_catch as e:
                # check num of retries
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f'maximum number of retries ({max_retries}) exceeded.')

                # incremental delay
                delay *= exponential_base * (1 + jitter * random.random())
                logging.info(f'retry on {e}, sleep for {const_delay + delay}')
                time.sleep(const_delay + delay)

            # raise on specific errors
            except errors_to_raise as e:
                raise e

            # raise exceptions for any errors not specified
            except Exception as e:
                raise e

            finally:  # return key if necessary
                if api_key is not None and return_key_func is not None:
                    end_t = time.time()
                    return_key_func(api_key, time_spent=end_t - start_t, forbid=forbid_key)

    return wrapper


def _is_openai_model(model: str) -> bool:
    return model in OPENAI_MODEL_ALIASES


def _resolve_ollama_model(model: str, is_chat_model: bool) -> str:
    if _is_openai_model(model):
        return OLLAMA_CHAT_MODEL if is_chat_model else OLLAMA_MODEL
    return model


def _ollama_options_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if 'temperature' in kwargs and kwargs['temperature'] is not None:
        options['temperature'] = kwargs['temperature']
    if 'top_p' in kwargs and kwargs['top_p'] is not None:
        options['top_p'] = kwargs['top_p']
    if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
        options['num_predict'] = kwargs['max_tokens']
    stop = kwargs.get('stop')
    if stop:
        if isinstance(stop, str):
            options['stop'] = [stop]
        else:
            options['stop'] = stop
    return options


def _ollama_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(f'Ollama request failed for {endpoint}: {exc}') from exc


def _ollama_chat_completion(messages: List[Dict[str, Any]], model: str, **kwargs) -> Dict[str, Any]:
    options = _ollama_options_from_kwargs(kwargs)
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
    }
    if options:
        payload['options'] = options
    response = _ollama_request(f'{OLLAMA_BASE_URL}/api/chat', payload)
    return {
        'model': response.get('model', model),
        'choices': [{
            'message': {'content': response.get('message', {}).get('content', '')},
            'finish_reason': response.get('done_reason', 'stop'),
        }],
    }


def _ollama_text_completion(prompt: str, model: str, **kwargs) -> Dict[str, Any]:
    options = _ollama_options_from_kwargs(kwargs)
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
    }
    if options:
        payload['options'] = options
    response = _ollama_request(f'{OLLAMA_BASE_URL}/api/generate', payload)
    text = response.get('response', '')
    if kwargs.get('echo'):
        text = f'{prompt}{text}'
    return {
        'model': response.get('model', model),
        'choices': [{
            'text': text,
            'finish_reason': response.get('done_reason', 'stop'),
            'logprobs': None,
        }],
    }


async def async_chatgpt(
    *args,
    messages: List[List[Dict[str, Any]]],
    model: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    tasks = [
        asyncio.to_thread(_ollama_chat_completion, message_set, model=model, **kwargs)
        for message_set in messages
    ]
    return await asyncio.gather(*tasks)


@retry_with_exponential_backoff
def openai_api_call(*args, **kwargs):
    model = kwargs['model']
    is_chat_model = Utils.is_chat(model)
    resolved_model = _resolve_ollama_model(model, is_chat_model)
    request_kwargs = dict(kwargs)
    request_kwargs.pop('model', None)
    if is_chat_model:
        if len(request_kwargs['messages']) <= 0:
            return []
        messages = request_kwargs.pop('messages')
        if isinstance(messages[0], list):  # batch request
            return asyncio.run(async_chatgpt(messages=messages, model=resolved_model, **request_kwargs))
        else:
            return _ollama_chat_completion(messages, resolved_model, **request_kwargs)
    else:
        prompt = request_kwargs['prompt']
        request_kwargs.pop('prompt', None)
        if isinstance(prompt, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_ollama_text_completion, item, resolved_model, **request_kwargs)
                    for item in prompt
                ]
                choices = [future.result()['choices'][0] for future in futures]
            return {'model': resolved_model, 'choices': choices}
        return _ollama_text_completion(prompt, resolved_model, **request_kwargs)
