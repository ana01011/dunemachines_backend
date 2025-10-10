"""
Repository for persisting and retrieving neurochemical states
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import asyncpg
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

# from ..core.constants import Hormone  # Unused import
from ..core.event import Event

logger = logging.getLogger(__name__)


class StateRepository:
    """Handles persistence of neurochemical states to PostgreSQL"""
    
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
        """Schema already created via migration"""
        logger.info("Neurochemical database schema ready")
    
    async def save_state(self, user_id: str, state: Dict[str, Any]):
        """Save current neurochemical state"""
        try:
            hormones = state.get('hormones', {})
            mood = state.get('mood', {})
            behavior = state.get('behavior', {})
            
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
                        event_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                             $11, $12, $13, $14, $15, $16, $17, $18, $19,
                             $20, $21, $22, $23, $24)
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
                        'data': json.loads(row['pattern_data']) if row['pattern_data'] else {},
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
