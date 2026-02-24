import os
from api_keys import TOGETHER_API_KEY, SERPAPI_API_KEY

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

def test_together_key():
    import requests
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 5,
        }
    )
    assert response.status_code == 200, f"Together API failed: {response.status_code} {response.text}"
    print("Together API key works!")

def test_serpapi_key():
    import requests
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "q": "test",
            "api_key": SERPAPI_API_KEY,
        }
    )
    assert response.status_code == 200, f"SerpAPI failed: {response.status_code} {response.text}"
    print("SerpAPI key works!")

def check_keys_not_empty():
    assert TOGETHER_API_KEY != "", "TOGETHER_API_KEY is empty in api_keys.py"
    assert SERPAPI_API_KEY != "", "SERPAPI_API_KEY is empty in api_keys.py"
    print("Keys are non-empty.")

if __name__ == "__main__":
    check_keys_not_empty()
    test_together_key()
    test_serpapi_key()