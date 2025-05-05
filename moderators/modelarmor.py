"""
Google Model Armor API - Text moderation

Model Armor (Responsible AI) checks for 4 categories:
- Sexually Explicit
- Dangerous
- Harassment
- Hate Speech

"""

import asyncio
import os
from typing import List

import aiohttp
from dotenv import load_dotenv

from moderators.utils import async_moderate

# Load environment variables
load_dotenv()  # Load variables from .env file
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = 'us-central1'
GCP_TEMPLATE_ID = 'rai-medium-template-id-1234'

MODERATION_BATCH_SIZE: int = 50
NUM_CONCURRENT_REQUESTS: int = 20
RETRY_WAIT_TIME: int = 3 # in seconds
MAX_RETRIES: int = 5

RATE_LIMIT = 200  # calls per minute
RATE_LIMIT_WINDOW = 60  # seconds

async def get_gcp_token() -> str:
    """Get GCP access token using gcloud CLI"""
    process = await asyncio.create_subprocess_exec(
        'gcloud', 'auth', 'print-access-token',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    return stdout.decode().strip()

async def moderate(texts: List[str]) -> List[dict]:
    """
    Check if the texts are flagged by the Google Model Armor API with retry logic.
    """
    results = []
    token = await get_gcp_token()

    # Set up the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    url = f"https://modelarmor.{GCP_LOCATION}.rep.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/templates/{GCP_TEMPLATE_ID}:sanitizeUserPrompt"

    async with aiohttp.ClientSession() as session:
        # Create tasks for all texts in the batch
        tasks = []
        for text in texts:
            tasks.append(process_single_text(session, url, headers, text))
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                # If there was an error, add empty result
                print(f"Error during moderation: {result}")
                results.append({
                    "dangerous": False,
                    "harassment": False,
                    "sexually_explicit": False,
                    "hate_speech": False
                })
            else:
                results.append(result)
                
    return results

async def process_single_text(session, url, headers, text):
    """Process a single text with retry logic."""
    body = {
        "user_prompt_data": {
            "text": text
        }
    }

    # Retry loop - will attempt the request up to MAX_RETRIES times
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json=body, timeout=10) as response:
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', RETRY_WAIT_TIME))
                    print(f"Rate limit hit, waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                res = await response.json()
                res = res['sanitizationResult']['filterResults']['rai']['raiFilterResult']['raiFilterTypeResults']
                labels = {k: True if v['matchState'] == 'MATCH_FOUND' else False for k, v in res.items()}
                
                return labels

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Request failed, retrying in {RETRY_WAIT_TIME} seconds... Error: {str(e)}")
                await asyncio.sleep(RETRY_WAIT_TIME)
            else:
                print(f"Max retries reached. Error: {str(e)}")
                raise e
    
    # This should never be reached due to the raise in the exception handler
    return {
        "dangerous": False,
        "harassment": False,
        "sexually_explicit": False,
        "hate_speech": False
    }
    
def main(df, lang='en'):
    # Prepare messages in batches 
    df = df[["prompt_id","text"]].copy()
    messages = df["text"].tolist()
    batches = [
        messages[i : i + MODERATION_BATCH_SIZE]
        for i in range(0, len(messages), MODERATION_BATCH_SIZE)
    ]

    # Run the asynchronous moderation check
    batch_results = asyncio.run(async_moderate(
        batches, 
        moderate,
        num_concurrent_requests=NUM_CONCURRENT_REQUESTS,
        rate_limit=RATE_LIMIT,
        rate_limit_window=RATE_LIMIT_WINDOW
    ))

    # Flatten the results and filter the DataFrame
    flattened_results = [categories for batch in batch_results for categories in batch]

    # Create columns for each category
    categories = flattened_results[0].keys()
    for category in categories:
        df[category] = [result[category] for result in flattened_results]

    df.to_csv(f"data/rabakbench_en_modelarmor.csv", index=False)

