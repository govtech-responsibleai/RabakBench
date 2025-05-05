"""
Azure AI Content Safety API - Text moderation

Azure Text Moderation checks for 4 categories:
- Hate - any content that attacks or uses discriminatory language with reference to a person or identity group based on certain differentiating attributes of these groups. 
- SelfHarm - describes language related to physical actions intended to purposely hurt, injure, damage one’s body or kill oneself
- Sexual - language related to anatomical organs and genitals, romantic relationships and sexual acts, acts portrayed in erotic or affectionate terms, including those portrayed as an assault or a forced sexual violent act against one’s will. 
- Violence - language related to physical actions intended to hurt, injure, damage, or kill someone or something; describes weapons, guns, and related entities. 
"""
import asyncio
import os
from typing import List

from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from moderators.utils import async_moderate

# Load environment variables
load_dotenv()  # Load variables from .env file
AZURE_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")

MODERATION_BATCH_SIZE: int = 50
NUM_CONCURRENT_REQUESTS: int = 20
MAX_RETRIES: int = 10

RATE_LIMIT = 1000  # calls per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Check if the provided texts are harmful using Azure AI Text Moderation.
async def moderate(texts: List[str]) -> List[dict]:
    """Process a batch of texts. """
    results = []
    async with ContentSafetyClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_API_KEY)) as client:
        # Collect all tasks
        tasks = []
        for text in texts:
            request = AnalyzeTextOptions(text=text)
            task = client.analyze_text(request)
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        for response in responses:
            flagged_categories = {
                category["category"]: category["severity"]
                for category in response["categoriesAnalysis"]
            }
            results.append(flagged_categories)

    return results


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
        )
    )

    # Flatten the results and filter the DataFrame
    flattened_results = [categories for batch in batch_results for categories in batch]

    # Create columns for each category
    categories = flattened_results[0].keys()
    for category in categories:
        df[category] = [result[category] for result in flattened_results]

    df.to_csv(f"data/{lang}/rabakbench_{lang}_azure.csv", index=False)
 