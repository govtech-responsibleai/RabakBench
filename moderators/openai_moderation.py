"""
OpenAI Moderation API

13 categories:
Hate: Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harassment.
Hate/threatening: Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.
Harassment: Content that expresses, incites, or promotes harassing language towards any target. 
Harassment/threatening: Harassment content that also includes violence or serious harm towards any target.
Self-harm: Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
Self-harm/intent: Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.
Self-harm/instructions: Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.
Sexual: Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).
Sexual/minors: Sexual content that includes an individual who is under 18 years old.
Violence: Content that depicts death, violence, or physical injury.
Violence/graphic: Content that depicts death, violence, or physical injury in graphic detail.
(Omni-only) Illicit: Content that includes instructions or advice that facilitate the planning or execution of wrongdoing, or that gives advice or instruction on how to commit illicit acts. For example, "how to shoplift" would fit this category.
(Omni-only) Illicit/violent: Content that includes instructions or advice that facilitate the planning or execution of wrongdoing that also includes violence, or that gives advice or instruction on the procurement of any weapon.
"""

import asyncio
from typing import List   

from dotenv import load_dotenv
from openai import AsyncOpenAI

from moderators.utils import async_moderate

# Load environment variables
load_dotenv()  # Load variables from .env file
MODERATION_MODEL: str = "omni-moderation-latest"
MODERATION_BATCH_SIZE: int = 50
NUM_CONCURRENT_REQUESTS: int = 20
MAX_RETRIES: int = 10

RATE_LIMIT = 5000  # calls per minute
RATE_LIMIT_WINDOW = 60  # seconds
TOKEN_REFRESH_RATE = RATE_LIMIT / RATE_LIMIT_WINDOW  # tokens per second

client = AsyncOpenAI()

# Check if the provided texts are harmful using OpenAI's moderation API.
async def moderate(texts: List[str]) -> List[bool]:
    """Process a batch of texts. """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = await client.moderations.create(
                model=MODERATION_MODEL,
                input=texts
            )
            
            return [result.categories for result in response.results]
        except Exception as err:
            print(f"Error checking harmful content: {err}")
            await asyncio.sleep(5)
            retries += 1
            continue
    return [False] * MODERATION_BATCH_SIZE


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

    # Create new columns for each category
    categories = vars(flattened_results[0]).keys()
    for category in categories:
        df[category] = [vars(result)[category] for result in flattened_results]

    df.to_csv(f"data/{lang}/rabakbench_{lang}_openai.csv", index=False)
 
