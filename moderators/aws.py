"""
Amazon Bedrock Guardrails Content Filters 

5 categories:
Hate: Describes input prompts and model responses that discriminate, criticize, insult, denounce, or dehumanize a person or group on the basis of an identity (such as race, ethnicity, gender, religion, sexual orientation, ability, and national origin).
Insults: Describes input prompts and model responses that includes demeaning, humiliating, mocking, insulting, or belittling language. This type of language is also labeled as bullying.
Sexual: Describes input prompts and model responses that indicates sexual interest, activity, or arousal using direct or indirect references to body parts, physical traits, or sex.
Violence: Describes input prompts and model responses that includes glorification of, or threats to inflict physical pain, hurt, or injury toward a person, group, or thing.
Misconduct: Describes input prompts and model responses that seeks or provides information about engaging in criminal activity, or harming, defrauding, or taking advantage of a person, group or institution.

More details at: https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf#guardrails-content-filters
"""
import asyncio

import aioboto3
import boto3
from dotenv import load_dotenv

from moderators.utils import async_moderate

# Load environment variables
load_dotenv()  # Load variables from .env file
GUARDRAIL_ID = None
GUARDRAIL_VERSION = None

MODERATION_BATCH_SIZE: int = 50
NUM_CONCURRENT_REQUESTS: int = 20
RETRY_WAIT_TIME: int = 5
MAX_RETRIES: int = 5

RATE_LIMIT = 1000  # calls per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Initialise Amazon clients
guardrail_client = boto3.client(service_name="bedrock", region_name="us-east-1")
aioboto3_session = aioboto3.Session()

def create_content_filter():
    """
    Basic content filter for Hate, Insults, Sexual, Violence, Misconduct. 
    (All categories besides from Prompt Attack.)
    No additional word configs. 

    Returns the created guardrail id and version
    """
    response = guardrail_client.create_guardrail(
        name="BasicContentFilter",
        description="Basic content filter for Hate, Insults, Sexual, Violence, Misconduct.",
        contentPolicyConfig={
            'filtersConfig': [
                {
                    'type': 'HATE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'INSULTS',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'SEXUAL',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'VIOLENCE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'MISCONDUCT',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                }

            ]
        },
        blockedInputMessaging='Blocked',
        blockedOutputsMessaging='Blocked'
    )

    guardrail_id = response['guardrailId']
    guardrail_version = response['version']
    return guardrail_id, guardrail_version

def get_guardrail(name):
    """
    Get guardrail from name if previously created 

    Returns the guardrail id and version if found, else None
    """
    guardrails = guardrail_client.list_guardrails()['guardrails']
    for guardrail in guardrails:
        if guardrail['name'] == name:
            return guardrail['id'], guardrail['version']
            
    return None, None


async def process_text(client, text):
    """Process a single text. """
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            content = [{"text": {"text": text}}]
            response = await client.apply_guardrail(
                guardrailIdentifier=GUARDRAIL_ID,
                guardrailVersion=GUARDRAIL_VERSION,
                source='INPUT',
                content=content
            )

            # Process response and return labels
            labels = {
                'HATE': False, 
                'INSULTS': False, 
                'SEXUAL': False, 
                'VIOLENCE': False, 
                'MISCONDUCT': False
            }
            if response['action'] == "GUARDRAIL_INTERVENED":
                response_filters = response['assessments'][0]['contentPolicy']['filters']
                for result in response_filters:
                    if result['type'] in labels:
                        labels[result['type']] = True
                        
            return labels
            
        except Exception as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                raise e
            
            await asyncio.sleep(RETRY_WAIT_TIME)

async def moderate(texts):
    """Process a batch of texts. """
    global GUARDRAIL_ID, GUARDRAIL_VERSION
    async with aioboto3_session.client(service_name="bedrock-runtime", region_name="us-east-1") as client:
        # Create tasks for all texts in the batch
        tasks = []
        for text in texts:
            tasks.append(process_text(client, text))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    return results

def main(df, lang='en'):
    # Get guardrail ID and version if exists, else create new guardrail
    global GUARDRAIL_ID, GUARDRAIL_VERSION
    guardrail_name = 'BasicContentFilter'
    GUARDRAIL_ID, GUARDRAIL_VERSION = get_guardrail(guardrail_name)
    if not GUARDRAIL_ID:
        GUARDRAIL_ID, GUARDRAIL_VERSION = create_content_filter()

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

    df.to_csv(f"data/{lang}/rabakbench_{lang}_aws.csv", index=False)
 
 