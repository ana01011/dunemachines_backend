"""
Repository for persisting and retrieving neurochemical states
Designed for future extensibility with web search and knowledge bases
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import asyncpg
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

from ..core.constants import Hormone
from ..core.event import Event

logger = logging.getLogger(__name__)


class StateRepository:
    """
    Handles persistence of neurochemical states to PostgreSQL
    
    Future extensibility:
    - Can trigger knowledge base searches based on cortisol levels
    - Can log web search triggers for high-stress situations
    - Maintains search history linked to neurochemical states
    """
    
    def __init__(self, db_pool: asyncpg.Pool, config_path: Optional[str] = None):
        self.db_pool = db_pool
        self.config = self._load_config(config_path)
        self.batch_size = self.config.get('batch_size', 100)
        self.retention_days = self.config.get('retention_days', 30)
        
        # Future: Knowledge base triggers
        self.knowledge_triggers = {
            'web_search_cortisol_threshold': 70,
            'textbook_search_cortisol_threshold': 60,
            'specialist_db_adrenaline_threshold': 65
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load storage configuration"""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'neurochemical_config.yaml'
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('storage', {})
        return {}
    
    async def initialize_schema(self):
        """Create necessary database tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS neurochemical_states (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    
                    -- Hormone levels and baselines
                    dopamine_level FLOAT,
                    dopamine_baseline FLOAT,
                    cortisol_level FLOAT,
                    cortisol_baseline FLOAT,
                    adrenaline_level FLOAT,
                    adrenaline_baseline FLOAT,
                    serotonin_level FLOAT,
                    serotonin_baseline FLOAT,
                    oxytocin_level FLOAT,
                    oxytocin_baseline FLOAT,
                    
                    -- Behavioral parameters
                    planning_depth INTEGER,
                    risk_tolerance FLOAT,
                    processing_speed FLOAT,
                    confidence FLOAT,
                    creativity FLOAT,
                    empathy FLOAT,
                    patience FLOAT,
                    thoroughness FLOAT,
                    
                    -- Mood indicators
                    valence FLOAT,
                    arousal FLOAT,
                    dominance FLOAT,
                    
                    -- Future: Knowledge access triggers
                    triggered_web_search BOOLEAN DEFAULT FALSE,
                    triggered_textbook_search BOOLEAN DEFAULT FALSE,
                    search_query TEXT,
                    
                    -- Metadata
                    event_context JSONB,
                    
                    INDEX idx_user_timestamp (user_id, timestamp DESC)
                )
            ''')
            
            # Events table for learning
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS neurochemical_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    magnitude FLOAT,
                    
                    -- Event details
                    message TEXT,
                    complexity FLOAT,
                    urgency FLOAT,
                    emotional_content FLOAT,
                    
                    -- Neurochemical response
                    dopamine_response FLOAT,
                    cortisol_response FLOAT,
                    adrenaline_response FLOAT,
                    serotonin_response FLOAT,
                    oxytocin_response FLOAT,
                    
                    -- Outcome tracking
                    quality_score FLOAT,
                    user_satisfaction FLOAT,
                    
                    -- Future: Knowledge access
                    required_web_search BOOLEAN DEFAULT FALSE,
                    required_textbook_access BOOLEAN DEFAULT FALSE,
                    knowledge_sources_used TEXT[],
                    
                    metadata JSONB,
                    
                    INDEX idx_user_event_time (user_id, timestamp DESC),
                    UNIQUE (event_id)
                )
            ''')
            
            # Learning patterns table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS neurochemical_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data JSONB,
                    confidence FLOAT,
                    last_updated TIMESTAMP DEFAULT NOW(),
                    
                    -- Future: Knowledge preferences
                    preferred_knowledge_sources TEXT[],
                    search_success_rate FLOAT,
                    
                    UNIQUE (user_id, pattern_type)
                )
            ''')
            
            logger.info("Neurochemical database schema initialized")
    
    async def save_state(self, user_id: str, state: Dict[str, Any]):
        """Save current neurochemical state"""
        try:
            hormones = state.get('hormones', {})
            mood = state.get('mood', {})
            behavior = state.get('behavior', {})
            
            # Check if we should trigger knowledge searches
            triggers = await self._check_knowledge_triggers(hormones, behavior)
            
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO neurochemical_states (
                        user_id, timestamp,
                        dopamine_level, dopamine_baseline,
                        cortisol_level, cortisol_baseline,
                        adrenaline_level, adrenaline_baseline,
                        serotonin_level, serotonin_baseline,
                        oxytocin_level, oxytocin_baseline,
                        planning_depth, risk_tolerance,
                        processing_speed, confidence,
                        creativity, empathy, patience, thoroughness,
                        valence, arousal, dominance,
                        triggered_web_search, triggered_textbook_search,
                        event_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                             $11, $12, $13, $14, $15, $16, $17, $18, $19,
                             $20, $21, $22, $23, $24, $25, $26, $27)
                ''',
                    user_id, datetime.now(),
                    hormones.get('dopamine', {}).get('level'),
                    hormones.get('dopamine', {}).get('baseline'),
                    hormones.get('cortisol', {}).get('level'),
                    hormones.get('cortisol', {}).get('baseline'),
                    hormones.get('adrenaline', {}).get('level'),
                    hormones.get('adrenaline', {}).get('baseline'),
                    hormones.get('serotonin', {}).get('level'),
                    hormones.get('serotonin', {}).get('baseline'),
                    hormones.get('oxytocin', {}).get('level'),
                    hormones.get('oxytocin', {}).get('baseline'),
                    behavior.get('planning_depth'),
                    behavior.get('risk_tolerance'),
                    behavior.get('processing_speed'),
                    behavior.get('confidence'),
                    behavior.get('creativity'),
                    behavior.get('empathy'),
                    behavior.get('patience'),
                    behavior.get('thoroughness'),
                    mood.get('valence'),
                    mood.get('arousal'),
                    mood.get('dominance'),
                    triggers.get('web_search', False),
                    triggers.get('textbook_search', False),
                    json.dumps(state.get('context', {}))
                )
                
        except Exception as e:
            logger.error(f"Failed to save neurochemical state: {e}")
    
    async def save_event(self, user_id: str, event: Event, responses: Dict[str, float],
                        outcome: Optional[Dict[str, float]] = None):
        """Save event and neurochemical response"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO neurochemical_events (
                        user_id, event_id, timestamp, event_type, magnitude,
                        message, complexity, urgency, emotional_content,
                        dopamine_response, cortisol_response,
                        adrenaline_response, serotonin_response, oxytocin_response,
                        quality_score, user_satisfaction,
                        metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                             $11, $12, $13, $14, $15, $16, $17)
                    ON CONFLICT (event_id) DO UPDATE SET
                        quality_score = EXCLUDED.quality_score,
                        user_satisfaction = EXCLUDED.user_satisfaction
                ''',
                    user_id, event.event_id, event.timestamp,
                    event.type, event.magnitude,
                    event.metadata.get('message', ''),
                    event.complexity, event.urgency, event.emotional_content,
                    responses.get('dopamine', 0),
                    responses.get('cortisol', 0),
                    responses.get('adrenaline', 0),
                    responses.get('serotonin', 0),
                    responses.get('oxytocin', 0),
                    outcome.get('quality', 0) if outcome else None,
                    outcome.get('satisfaction', 0) if outcome else None,
                    json.dumps(event.metadata)
                )
        except Exception as e:
            logger.error(f"Failed to save event: {e}")
    
    async def get_recent_states(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Retrieve recent neurochemical states"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM neurochemical_states
                    WHERE user_id = $1 
                    AND timestamp > $2
                    ORDER BY timestamp DESC
                ''', user_id, datetime.now() - timedelta(hours=hours))
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve states: {e}")
            return []
    
    async def get_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Retrieve learned patterns for a user"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT pattern_type, pattern_data, confidence
                    FROM neurochemical_patterns
                    WHERE user_id = $1
                ''', user_id)
                
                patterns = {}
                for row in rows:
                    patterns[row['pattern_type']] = {
                        'data': json.loads(row['pattern_data']),
                        'confidence': row['confidence']
                    }
                return patterns
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return {}
    
    async def save_patterns(self, user_id: str, patterns: Dict[str, Any]):
        """Save learned patterns"""
        try:
            async with self.db_pool.acquire() as conn:
                for pattern_type, data in patterns.items():
                    await conn.execute('''
                        INSERT INTO neurochemical_patterns (
                            user_id, pattern_type, pattern_data, confidence
                        ) VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, pattern_type) DO UPDATE SET
                            pattern_data = EXCLUDED.pattern_data,
                            confidence = EXCLUDED.confidence,
                            last_updated = NOW()
                    ''',
                        user_id, pattern_type,
                        json.dumps(data.get('data', {})),
                        data.get('confidence', 0.5)
                    )
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    async def _check_knowledge_triggers(self, hormones: Dict, behavior: Dict) -> Dict[str, bool]:
        """
        Check if current state should trigger knowledge searches
        Future: This will actually trigger web search and textbook access
        """
        triggers = {
            'web_search': False,
            'textbook_search': False,
            'specialist_db': False
        }
        
        cortisol = hormones.get('cortisol', {}).get('level', 0)
        adrenaline = hormones.get('adrenaline', {}).get('level', 0)
        
        # High cortisol triggers careful research
        if cortisol > self.knowledge_triggers['web_search_cortisol_threshold']:
            triggers['web_search'] = True
            logger.info(f"High cortisol ({cortisol}) triggers web search")
        
        if cortisol > self.knowledge_triggers['textbook_search_cortisol_threshold']:
            triggers['textbook_search'] = True
            logger.info(f"High cortisol ({cortisol}) triggers textbook search")
        
        # High adrenaline triggers specialist knowledge
        if adrenaline > self.knowledge_triggers['specialist_db_adrenaline_threshold']:
            triggers['specialist_db'] = True
            logger.info(f"High adrenaline ({adrenaline}) triggers specialist DB")
        
        return triggers
    
    async def cleanup_old_data(self):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    DELETE FROM neurochemical_states
                    WHERE timestamp < $1
                ''', cutoff_date)
                
                await conn.execute('''
                    DELETE FROM neurochemical_events
                    WHERE timestamp < $1
                ''', cutoff_date)
                
                logger.info(f"Cleaned up data older than {cutoff_date}")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")