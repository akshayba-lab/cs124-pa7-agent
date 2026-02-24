"""Utility methods to load movie data from data files.

Ported to Python 3 by Matt Mistele (@mmistele) and Sam Redmond (@sredmond).

Intended for PA7 in Stanford's CS124.
"""
import csv
import json
from typing import Tuple, List, Dict
from functools import lru_cache

import numpy as np
from openai import OpenAI, APIConnectionError

DEFAULT_STOP = ["\n\n\n\n\n", "<</SYS>>"]

def load_ratings(src_filename, delimiter: str = '%',
                 header: bool = False) -> Tuple[List, np.ndarray]:
    title_list = load_titles('data/movies.txt')
    user_id_set = set()
    with open(src_filename, 'r') as f:
        content = f.readlines()
        for line in content:
            user_id = int(line.split(delimiter)[0])
            if user_id not in user_id_set:
                user_id_set.add(user_id)
    num_users = len(user_id_set)
    num_movies = len(title_list)
    mat = np.zeros((num_movies, num_users))

    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        for line in reader:
            mat[int(line[1])][int(line[0])] = float(line[2])
    return title_list, mat


def load_titles(src_filename: str, delimiter: str = '%',
                header: bool = False) -> List:
    title_list = []
    with open(src_filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)

        for line in reader:
            movieID, title, genres = int(line[0]), line[1], line[2]
            if title[0] == '"' and title[-1] == '"':
                title = title[1:-1]
            title_list.append(title)
    return title_list


def load_sentiment_dictionary(src_filename: str, delimiter: str = ',',
                              header: bool = False) -> Dict:
    with open(src_filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        return dict(reader)

@lru_cache
def load_together_client():
    together_client = None
    try:
        from api_keys import TOGETHER_API_KEY

        together_client = OpenAI(api_key=TOGETHER_API_KEY,
            base_url='https://api.together.xyz',
        )
    except ImportError:
        print("\001\033[93m\002WARNING: Unable to load Together API client (TOGETHER_API_KEY in api_keys.py not found)\001\033[0m\002")
        print("\001\033[93m\002LLM Calls will not work.  Please add a TOGETHER_API_KEY to api_keys.py before starting parts 2 and 3.\001\033[0m\002")

    return together_client

def call_llm(messages, client, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens
    )

    return chat_completion.choices[0].message.content

# model = "meta-llama/Llama-2-70b-chat-hf"
def stream_llm_to_console(messages, client, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256, stop=None):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            stop=stop
        )

        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ""
            print(chunk.choices[0].delta.content or "", end="", flush=True)

        print()
    except APIConnectionError as e:
        print("\001\033[91m\002ERROR connecting to Together API!  Please check your TOGETHER_API_KEY in api_keys.py and try again.\001\033[0m\002")
        response = None

    return response
