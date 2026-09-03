
from typing import List, Optional, Dict, Any
import os
from typing import Optional
from unittest import result
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import requests
import re

# Configuration
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set it in your environment or .env file."
    )

# You can change this to another model available to your account.
MODEL_NAME = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Initialize Tavily client
tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)

# Structured response schema
class RelevanceCheck(BaseModel):
    is_relevant: bool
    reason: str  # short explanation, useful for debugging

def check_relevance_with_llm(topic: str, content: str, raw_content: str) -> RelevanceCheck:
    """Ask a small LLM call whether a page's content matches the topic."""
    # excerpt = page_text[:3000]  # keep the prompt short and cheap

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "You judge whether a webpage's content is relevant to a given topic. Be strict but fair.",
            },
            {
                "role": "user",
                "content": (
                    f"Topic:\n{topic.strip()}\n\n"
                    f"Webpage content (may be truncated):\n"
                    f"Actual Content: {content.strip()}\n\n"
                    f"Raw Content: {raw_content.strip()}\n\n"
                    "Is this webpage relevant to the topic?"
                ),
            },
        ],
        text_format=RelevanceCheck,
    )

    # The SDK parses the structured response into our Pydantic model.
    result: Optional[RelevanceCheck] = response.output_parsed

    if result is None:
        raise RuntimeError(
            "The model did not return a structured RelevanceCheck response."
        )

    return result

# 4. The verification Function
def verify_reference(url: str, topic: str, content: str, raw_content: str) -> dict:
    """
    Check that a reference URL is:
      - reachable (HTTP 200 response), and
      - relevant to the topic, judged by an LLM reading the page content.

    Returns a dict the agent can read to decide whether to keep/replace the URL.
    """
    result = {"url": url, "accessible": False, "relevant": False, "reason": ""}
    try:
        # A short timeout + browser-like header keeps this simple and fast.
        response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        result["accessible"] = response.status_code == 200

        if result["accessible"]:
            relevance = check_relevance_with_llm(topic, content, raw_content)
            result["relevant"] = relevance.is_relevant
            result["reason"] = relevance.reason
        else:
            result["reason"] = f"Bad status code: {response.status_code}"

    except requests.RequestException as e:
        result["reason"] = f"Request failed: {e}"
    print(result)
    return result


def search_and_verify_references(
    questions: List[str],
) -> List[Dict[str, Any]]:
    """
    Find one candidate reference URL for every question.

    Retries search + verification up to 3 attempts if
    the returned reference is not relevant.
    """

    results = []

    for question in questions:
        result_data = {
            "question": question,
            "url": None,
            "title": None,
            "content": None,
            "url_relevant": False,
            "reason": None,
        }

        for attempt in range(1, 4):
            response = tavily_client.search(
                query=question,
                search_depth="advanced",
                max_results=1,
                include_answer=True,
                include_raw_content=True,
            )

            search_results = response.get("results", [])

            if not search_results:
                result_data["reason"] = f"No search results (attempt {attempt})"
                continue

            best = search_results[0]

            verification = verify_reference(
                best.get("url"),
                question,
                best.get("content"),
                best.get("raw_content")
            )

            if verification["relevant"]:
                result_data.update({
                    "url": best.get("url"),
                    "title": best.get("title"),
                    "content": best.get("content"),
                    "url_relevant": True,
                    "reason": verification["reason"],
                })
                break

            else:
                continue # try again with a new search result

        results.append(result_data)

    return results
