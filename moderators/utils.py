import asyncio
import time
from typing import List, Callable

from tqdm.asyncio import tqdm_asyncio

# Rate limiter to prevent API rate limit
class RateLimiter:
    def __init__(self, rate_limit: int, window: int):
        self.rate_limit = rate_limit
        self.window = window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.rate_limit,
                self.tokens + time_passed * (self.rate_limit / self.window)
            )
            self.last_update = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / (self.rate_limit / self.window)
                await asyncio.sleep(sleep_time)
                self.tokens = 0
                self.last_update = time.time()
            else:
                self.tokens -= 1

# Asynchronously check harmful content for messages in parallel (batched).
async def async_moderate(
        batches: List[List[str]], 
        moderation_function: Callable[[List[str]], List[bool]],
        num_concurrent_requests: int = 10,
        rate_limit: int = 100,
        rate_limit_window: int = 60
    ) -> List[List[bool]]:
    """
    Asynchronously check harmful content for messages in parallel (batched).
    """
    semaphore = asyncio.Semaphore(num_concurrent_requests)
    rate_limiter = RateLimiter(rate_limit, rate_limit_window)

    async def process_batch(batch: List[str]) -> List[bool]:
        async with semaphore:
            await rate_limiter.acquire()  # Wait for rate limit
            return await moderation_function(batch)

    # Create tasks with their original indices
    tasks = [process_batch(batch) for batch in batches]
    
    # Wait for all tasks to complete in order with progress bar
    all_results = await tqdm_asyncio.gather(
        *tasks,
        desc="Checking harmful content",
        unit="batch"
    )

    return all_results
    