"""
Manages persistence operations with batching and optimization
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import json
import logging
from collections import deque
import asyncpg

from .state_repository import StateRepository
from ..core.event import Event

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manages efficient persistence of neurochemical data
    Handles batching, queuing, and async operations
    
    Future extensibility:
    - Can queue knowledge base search requests
    - Can manage distributed storage across servers
    - Can handle high-frequency state updates efficiently
    """
    
    def __init__(self, repository: StateRepository):
        self.repository = repository
        self.state_queue = deque(maxlen=1000)
        self.event_queue = deque(maxlen=1000)
        self.pattern_queue = deque(maxlen=100)
        
        self.batch_interval = 5.0  # seconds
        self.is_running = False
        self.persist_task = None
        
        # Future: Knowledge search queue
        self.knowledge_search_queue = deque(maxlen=100)
        
    async def start(self):
        """Start the persistence manager"""
        if not self.is_running:
            self.is_running = True
            self.persist_task = asyncio.create_task(self._persistence_loop())
            logger.info("Persistence manager started")
    
    async def stop(self):
        """Stop the persistence manager"""
        self.is_running = False
        if self.persist_task:
            # Flush remaining data
            await self._flush_all()
            self.persist_task.cancel()
            try:
                await self.persist_task
            except asyncio.CancelledError:
                pass
            logger.info("Persistence manager stopped")
    
    def queue_state(self, user_id: str, state: Dict[str, Any]):
        """Queue a state for persistence"""
        self.state_queue.append({
            'user_id': user_id,
            'state': state,
            'timestamp': datetime.now()
        })
    
    def queue_event(self, user_id: str, event: Event, responses: Dict[str, float],
                    outcome: Optional[Dict[str, float]] = None):
        """Queue an event for persistence"""
        self.event_queue.append({
            'user_id': user_id,
            'event': event,
            'responses': responses,
            'outcome': outcome
        })
    
    def queue_patterns(self, user_id: str, patterns: Dict[str, Any]):
        """Queue patterns for persistence"""
        self.pattern_queue.append({
            'user_id': user_id,
            'patterns': patterns
        })
    
    def queue_knowledge_search(self, user_id: str, query: str, 
                              search_type: str, neurochemical_trigger: Dict):
        """
        Queue a knowledge search request
        Future: This will trigger actual web/textbook searches
        """
        self.knowledge_search_queue.append({
            'user_id': user_id,
            'query': query,
            'search_type': search_type,  # 'web', 'textbook', 'specialist'
            'trigger': neurochemical_trigger,  # What neurochemical state triggered this
            'timestamp': datetime.now()
        })
        logger.info(f"Queued {search_type} search: {query[:50]}...")
    
    async def _persistence_loop(self):
        """Main persistence loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._flush_all()
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
    
    async def _flush_all(self):
        """Flush all queues"""
        # Process states
        if self.state_queue:
            states_to_save = []
            while self.state_queue:
                states_to_save.append(self.state_queue.popleft())
            
            for item in states_to_save:
                await self.repository.save_state(
                    item['user_id'],
                    item['state']
                )
        
        # Process events
        if self.event_queue:
            events_to_save = []
            while self.event_queue:
                events_to_save.append(self.event_queue.popleft())
            
            for item in events_to_save:
                await self.repository.save_event(
                    item['user_id'],
                    item['event'],
                    item['responses'],
                    item['outcome']
                )
        
        # Process patterns
        if self.pattern_queue:
            patterns_to_save = []
            while self.pattern_queue:
                patterns_to_save.append(self.pattern_queue.popleft())
            
            for item in patterns_to_save:
                await self.repository.save_patterns(
                    item['user_id'],
                    item['patterns']
                )
        
        # Future: Process knowledge searches
        if self.knowledge_search_queue:
            await self._process_knowledge_searches()
    
    async def _process_knowledge_searches(self):
        """
        Process queued knowledge searches
        Future: This will integrate with web search APIs and vector databases
        """
        searches_to_process = []
        while self.knowledge_search_queue:
            searches_to_process.append(self.knowledge_search_queue.popleft())
        
        for search in searches_to_process:
            logger.info(f"Would trigger {search['search_type']} search: {search['query'][:50]}...")
            # Future implementation:
            # - If web search: call web search API
            # - If textbook: query vector database
            # - If specialist: query specialized knowledge base
            # - Store results linked to neurochemical state
    
    async def get_state_at_time(self, user_id: str, timestamp: datetime) -> Optional[Dict]:
        """Get neurochemical state at specific time"""
        states = await self.repository.get_recent_states(user_id, hours=1)
        
        # Find closest state to timestamp
        closest = None
        min_diff = float('inf')
        
        for state in states:
            diff = abs((state['timestamp'] - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest = state
        
        return closest
    
    async def get_learning_data(self, user_id: str) -> Dict[str, Any]:
        """Get data for learning algorithms"""
        patterns = await self.repository.get_user_patterns(user_id)
        recent_states = await self.repository.get_recent_states(user_id, hours=24)
        
        return {
            'patterns': patterns,
            'recent_states': recent_states,
            'state_count': len(recent_states)
        }