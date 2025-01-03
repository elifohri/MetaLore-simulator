from collections import deque
from typing import Dict, Optional, Union, Iterator

# Type alias for the job data
Job = Dict[str, Optional[Union[float, int]]]

class JobQueue:
    """
    A simple FIFO queue for managing job queues.
    """
    
    def __init__(self) -> None:
        # Initialize the buffer
        self.data_queue: deque[Job] = deque()

    def enqueue_job(self, job: Job) -> None:
        # Add a job to the queue
        self.data_queue.append(job)

    def dequeue_job(self) -> Optional[Job]:
        # Remove and return the job from the front of the queue
        if not self.data_queue:
            raise IndexError("Cannot dequeue from an empty queue.")
        return self.data_queue.popleft()

    def peek_job(self) -> Optional[Job]:
        # Retrieve the job at the front of the queue without removing it
        return self.data_queue[0] if self.data_queue else None
        
    def current_size(self) -> int:
        # Return the current size of the queue
        return len(self.data_queue)
    
    def is_empty(self) -> bool:
        # Check if the queue is empty
        return len(self.data_queue) == 0
    
    def clear(self) -> None:
        # Clear all jobs from the queue.
        self.data_queue.clear()
    
    def __iter__(self) -> Iterator[Job]:
        # Iterate over the jobs in the queue without removing them.
        return iter(self.data_queue)
