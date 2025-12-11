"""
GPT OSS Safeguard - 20B via HuggingFace Inference Endpoints

This script uses HuggingFace Inference Endpoints (via Groq provider)
to call OpenAI's GPT OSS Safeguard 20B model for content moderation.

Model: https://huggingface.co/openai/gpt-oss-safeguard-20b
Guide: https://cookbook.openai.com/articles/gpt-oss-safeguard-guide

Categories:
1. Hateful (level_1_discriminatory, level_2_hate_speech)
2. Insults
3. Sexual (level_1_not_appropriate_for_minors, level_2_not_appropriate_for_all_ages)
4. Physical Violence
5. Self-Harm (level_1_self_harm_intent, level_2_self_harm_action)
6. All Other Misconduct (level_1_not_socially_accepted, level_2_illegal_activities)
"""

import os
import json
import asyncio
import gc
from typing import List, Dict

from huggingface_hub import AsyncInferenceClient
import pandas as pd
import torch

from moderators.batch_utils import async_moderate

# Configuration
HF_API_KEY = os.getenv("HF_TOKEN")

MODERATION_BATCH_SIZE: int = 10
NUM_CONCURRENT_REQUESTS: int = 5  # Cannot raise, will have `429 Too Many Requests`
RETRY_WAIT_TIME: int = 10
MAX_RETRIES: int = 5

# Groq rate limits for openai/gpt-oss-safeguard-20b
# TPM (Tokens Per Minute): 150,000
# Each request uses ~1800 tokens (1400 token policy + text + response)
# Maximum safe requests: 150000 / 1800 = ~83 per minute
RATE_LIMIT = 60  # calls per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Initialize HuggingFace Inference Client
huggingface_client = AsyncInferenceClient(api_key=HF_API_KEY)

# Model name on HuggingFace
MODEL_NAME = "openai/gpt-oss-safeguard-20b"


def get_system_prompt():
    """Get the policy prompt for GPT OSS Safeguard."""
    # Read the policy from POLICY.md in the moderators directory
    policy_path = os.path.join(os.path.dirname(__file__), "POLICY.md")
    with open(policy_path, "r") as f:
        return f.read()


async def process_text(text: str) -> Dict[str, int]:
    """Process a single text with GPT OSS Safeguard."""
    retry_count = 0
    output_text = None  # Initialize to track for error reporting

    while retry_count <= MAX_RETRIES:
        try:
            # Format messages for chat completion
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": text}
            ]

            # Call HuggingFace Inference API via Groq
            completion = await huggingface_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1500,  # Enough for reasoning + JSON response
                temperature=0.0,  # Deterministic output for moderation
            )

            # Extract the response text
            output_text = completion.choices[0].message.content.strip()

            # Handle empty responses
            if not output_text:
                raise ValueError("Model returned empty response")

            # Parse JSON output
            try:
                # Try to parse as JSON
                result = json.loads(output_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                # Sometimes models wrap JSON in markdown code blocks
                if "```json" in output_text:
                    json_start = output_text.find("```json") + 7
                    json_end = output_text.find("```", json_start)
                    result = json.loads(output_text[json_start:json_end].strip())
                elif "```" in output_text:
                    json_start = output_text.find("```") + 3
                    json_end = output_text.find("```", json_start)
                    result = json.loads(output_text[json_start:json_end].strip())
                else:
                    # Try to find JSON object in the text
                    json_start = output_text.find("{")
                    json_end = output_text.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        result = json.loads(output_text[json_start:json_end])
                    else:
                        raise ValueError(f"Could not parse JSON from output: {output_text}")

            # Convert to sub-level binary labels (10 columns total)
            # Initialize all sub-levels to 0
            labels = {
                "level_1_discriminatory": 0,
                "level_2_hate_speech": 0,
                "insults": 0,
                "level_1_not_appropriate_for_minors": 0,
                "level_2_not_appropriate_for_all_ages": 0,
                "physical_violence": 0,
                "level_1_self_harm_intent": 0,
                "level_2_self_harm_action": 0,
                "level_1_not_socially_accepted": 0,
                "level_2_illegal_activities": 0,
            }

            # Map the model outputs to specific sub-level columns
            hateful = result.get("hateful", "FALSE")
            if hateful == "level_1_discriminatory":
                labels["level_1_discriminatory"] = 1
            elif hateful == "level_2_hate_speech":
                labels["level_2_hate_speech"] = 1

            insults = result.get("insults", "FALSE")
            if insults == "insults":
                labels["insults"] = 1

            sexual = result.get("sexual", "FALSE")
            if sexual == "level_1_not_appropriate_for_minors":
                labels["level_1_not_appropriate_for_minors"] = 1
            elif sexual == "level_2_not_appropriate_for_all_ages":
                labels["level_2_not_appropriate_for_all_ages"] = 1

            physical_violence = result.get("physical_violence", "FALSE")
            if physical_violence == "physical_violence":
                labels["physical_violence"] = 1

            self_harm = result.get("self_harm", "FALSE")
            if self_harm == "level_1_self_harm_intent":
                labels["level_1_self_harm_intent"] = 1
            elif self_harm == "level_2_self_harm_action":
                labels["level_2_self_harm_action"] = 1

            all_other_misconduct = result.get("all_other_misconduct", "FALSE")
            if all_other_misconduct == "level_1_not_socially_accepted":
                labels["level_1_not_socially_accepted"] = 1
            elif all_other_misconduct == "level_2_illegal_activities":
                labels["level_2_illegal_activities"] = 1

            return labels

        except Exception as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                print(f"\n{'!'*60}")
                print(f"ERROR: Failed to process text after {MAX_RETRIES} retries")
                print(f"{'!'*60}")
                print(f"Error: {e}")
                print(f"Text: {text[:200]}...")
                if output_text is not None:
                    print(f"Raw model output: '{output_text}'")
                else:
                    print("No output received from model")
                print(f"{'!'*60}\n")
                # Return all zeros on failure (treating as safe/unclassifiable)
                return {
                    "level_1_discriminatory": 0,
                    "level_2_hate_speech": 0,
                    "insults": 0,
                    "level_1_not_appropriate_for_minors": 0,
                    "level_2_not_appropriate_for_all_ages": 0,
                    "physical_violence": 0,
                    "level_1_self_harm_intent": 0,
                    "level_2_self_harm_action": 0,
                    "level_1_not_socially_accepted": 0,
                    "level_2_illegal_activities": 0,
                }
            # Don't spam retry messages
            await asyncio.sleep(RETRY_WAIT_TIME)


async def moderate(texts: List[str]) -> List[Dict[str, int]]:
    """Process a batch of texts with retry logic."""
    # Create tasks for all texts in the batch
    tasks = []
    for text in texts:
        tasks.append(process_text(text))

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Return all results
    return responses


def main(df, lang: str = 'en', batch_size: int = MODERATION_BATCH_SIZE):
    """Main function to process a dataframe with GPT OSS Safeguard."""
    # Prepare messages in batches
    df = df[["prompt_id", "text"]].copy()
    messages = df["text"].tolist()

    print(f"Processing {len(df)} texts with GPT OSS Safeguard ({lang})")
    print(f"Model: {MODEL_NAME}")
    print(f"Provider: groq (billed to govtech)")

    batches = [
        messages[i : i + batch_size]
        for i in range(0, len(messages), batch_size)
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

    # Flatten the results and filter the DataFrame
    flattened_results = [categories for batch in batch_results for categories in batch]

    # Create columns for each category
    categories = flattened_results[0].keys()
    for category in categories:
        df[category] = [result[category] for result in flattened_results]

    # Save to CSV
    df.to_csv(f"data/{lang}/rabakbench_{lang}_gptoss.csv", index=False)
    print(f"Saved results to data/{lang}/rabakbench_{lang}_gptoss.csv")

    # Clear gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # Process all languages
    for lang in ['en', 'ms', 'ta', 'zh']:
        df = pd.read_csv(f"data/{lang}/rabakbench_{lang}.csv")
        main(df, lang=lang)
