"""
Qwen3Guard-Gen-8B via OpenAI Chat Completions API

Model: https://huggingface.co/Qwen/Qwen3Guard-Gen-8B
Uses local vLLM/SGLang server with OpenAI-compatible API for content moderation.

Categories:
- Violent: Content promoting violence or physical harm
- Non-violent Illegal Acts: Content related to illegal activities
- Sexual Content or Sexual Acts: Sexual or adult content
- PII: Personally Identifiable Information
- Suicide & Self-Harm: Content related to self-harm or suicide
- Unethical Acts: Unethical behavior or activities
- Politically Sensitive Topics: Politically sensitive content
- Copyright Violation: Copyright infringement
- Jailbreak: Attempts to bypass safety measures
"""

import asyncio
import os
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI

from moderators.utils import async_moderate

# Load environment variables
load_dotenv()

# OpenAI-compatible server configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = os.getenv("QWEN3GUARD_API_BASE", "http://localhost:8000/v1") # Start the server via: vllm serve Qwen/Qwen3Guard-Gen-8B --port 8000
MODEL_NAME = "Qwen/Qwen3Guard-Gen-8B"

MODERATION_BATCH_SIZE: int = 50
NUM_CONCURRENT_REQUESTS: int = 20
MAX_RETRIES: int = 10

RATE_LIMIT = 1000  # calls per minute (adjust based on your server capacity)
RATE_LIMIT_WINDOW = 60  # seconds

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def extract_label_and_categories(content: str) -> tuple[str, list[str]]:
    """
    Extract safety label and categories from Qwen3Guard output.

    Args:
        content: The raw output text from the model

    Returns:
        Tuple of (safety_label, categories_list)
        - safety_label: "Safe", "Unsafe", or "Controversial"
        - categories_list: List of flagged categories
    """
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"

    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else "Safe"

    categories = re.findall(category_pattern, content)
    # Remove "None" from categories if present
    categories = [cat for cat in categories if cat != "None"]

    return label, categories


async def moderate(texts: List[str]) -> List[Dict[str, str]]:
    """
    Process a batch of texts with retry logic.

    Args:
        texts: List of texts to moderate

    Returns:
        List of dictionaries containing safety labels and category flags
    """
    results = []

    for text in texts:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                # Create messages in OpenAI format
                messages = [{"role": "user", "content": text}]

                # Call the chat completions endpoint
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=128,
                    temperature=0.0  # Use deterministic output for moderation
                )

                # Extract the response content
                content = response.choices[0].message.content

                # Extract label and categories from output
                safety_label, flagged_categories = extract_label_and_categories(content)

                # Initialize all categories with "Safe"
                categories = {
                    "Violent": "Safe",
                    "Non-violent Illegal Acts": "Safe",
                    "Sexual Content or Sexual Acts": "Safe",
                    "PII": "Safe",
                    "Suicide & Self-Harm": "Safe",
                    "Unethical Acts": "Safe",
                    "Politically Sensitive Topics": "Safe",
                    "Copyright Violation": "Safe",
                    "Jailbreak": "Safe"
                }

                # Update flagged categories with the safety label
                for cat in flagged_categories:
                    if cat in categories:
                        categories[cat] = safety_label

                results.append(categories)
                break  # Success, exit retry loop

            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"Failed to moderate text after {MAX_RETRIES} retries: {e}")
                    # Return a default safe result
                    results.append({
                        "Violent": "Safe",
                        "Non-violent Illegal Acts": "Safe",
                        "Sexual Content or Sexual Acts": "Safe",
                        "PII": "Safe",
                        "Suicide & Self-Harm": "Safe",
                        "Unethical Acts": "Safe",
                        "Politically Sensitive Topics": "Safe",
                        "Copyright Violation": "Safe",
                        "Jailbreak": "Safe"
                    })
                else:
                    await asyncio.sleep(5)
                    continue

    return results


def main(df, lang='en'):
    """
    Main function to process the dataframe and save moderation results.

    Args:
        df: DataFrame with 'prompt_id' and 'text' columns
        lang: Language code (default: 'en')
    """
    # Prepare messages in batches
    df = df[["prompt_id", "text"]].copy()
    messages = df["text"].tolist()
    batches = [
        messages[i : i + MODERATION_BATCH_SIZE]
        for i in range(0, len(messages), MODERATION_BATCH_SIZE)
    ]

    # Run the asynchronous moderation check
    batch_results = asyncio.run(
        async_moderate(
            batches,
            moderate,
            num_concurrent_requests=NUM_CONCURRENT_REQUESTS,
            rate_limit=RATE_LIMIT,
            rate_limit_window=RATE_LIMIT_WINDOW,
        )
    )

    # Flatten the results
    flattened_results = [result for batch in batch_results for result in batch]

    # Create columns for each category
    categories = flattened_results[0].keys()
    for category in categories:
        df[category] = [result[category] for result in flattened_results]

    # Save to CSV
    df.to_csv(f"data/{lang}/rabakbench_{lang}_qwen3guard.csv", index=False)
    print(f"Saved results to data/{lang}/rabakbench_{lang}_qwen3guard.csv")
