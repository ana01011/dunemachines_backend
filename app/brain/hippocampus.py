"""
Hippocampus - Memory System for Neural Router
OMNIUS v2 - Stores and retrieves routing memories

Uses in-memory storage with optional ChromaDB backend.
Helps the router learn from past successful routings.
"""
import numpy as np
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from collections import defaultdict


@dataclass
class MemoryRecord:
    """A single memory stored in the hippocampus"""
    id: str
    user_id: str
    query_summary: str
    embedding: np.ndarray
    regions_used: List[str]
    was_successful: bool
    reward: float
    importance_score: float
    access_count: int
    created_at: datetime
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query_summary": self.query_summary,
            "regions_used": self.regions_used,
            "was_successful": self.was_successful,
            "reward": self.reward,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


@dataclass
class RetrievalResult:
    """Result from memory retrieval"""
    memories: List[MemoryRecord]
    similarity_scores: List[float]
    suggested_regions: List[str]
    confidence: float
    retrieval_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "memories": [m.to_dict() for m in self.memories],
            "similarity_scores": self.similarity_scores,
            "suggested_regions": self.suggested_regions,
            "confidence": self.confidence,
            "retrieval_time_ms": self.retrieval_time_ms
        }


@dataclass
class HippocampusConfig:
    """Configuration for the Hippocampus"""
    max_memories_per_user: int = 1000
    embedding_dim: int = 768
    similarity_threshold: float = 0.7
    importance_decay_rate: float = 0.01
    consolidation_threshold: float = 0.3
    use_chromadb: bool = False
    chromadb_path: str = "./chromadb_data"


class Hippocampus:
    """
    Memory system for the Neural Router.
    
    Stores successful routing decisions and retrieves similar
    past experiences to help guide future routing.
    
    Features:
    - In-memory storage with numpy-based similarity search
    - Optional ChromaDB backend for persistence
    - Memory consolidation (important memories persist)
    - Per-user memory isolation
    """
    
    def __init__(self, config: Optional[HippocampusConfig] = None):
        """Initialize the Hippocampus"""
        self.config = config or HippocampusConfig()
        
        # In-memory storage
        self.memories: Dict[str, List[MemoryRecord]] = defaultdict(list)
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.total_stores = 0
        self.total_retrievals = 0
        self.total_hits = 0
        
        # ChromaDB client (optional)
        self.chroma_client = None
        self.chroma_collection = None
        
        if self.config.use_chromadb:
            self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB backend"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.chromadb_path,
                anonymized_telemetry=False
            ))
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="neural_router_memories",
                metadata={"hnsw:space": "cosine"}
            )
            
            print("✅ ChromaDB initialized for Hippocampus")
        except ImportError:
            print("⚠️ ChromaDB not installed, using in-memory storage only")
            self.config.use_chromadb = False
        except Exception as e:
            print(f"⚠️ ChromaDB init failed: {e}, using in-memory storage")
            self.config.use_chromadb = False
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _truncate_query(self, query: str, max_length: int = 200) -> str:
        """Truncate query for storage"""
        if len(query) <= max_length:
            return query
        return query[:max_length-3] + "..."
    
    def store(
        self,
        user_id: str,
        query: str,
        embedding: np.ndarray,
        regions_used: List[str],
        was_successful: bool,
        reward: float = 0.0,
        importance_score: Optional[float] = None
    ) -> MemoryRecord:
        """
        Store a routing memory.
        
        Args:
            user_id: User identifier
            query: The original query text
            embedding: Query embedding vector
            regions_used: Which brain regions were used
            was_successful: Whether the routing was successful
            reward: Reward received (0-1)
            importance_score: Override importance (otherwise calculated)
        
        Returns:
            The stored MemoryRecord
        """
        # Calculate importance if not provided
        if importance_score is None:
            importance_score = self._calculate_importance(
                was_successful, reward, embedding
            )
        
        # Create memory record
        memory_id = str(uuid4())
        record = MemoryRecord(
            id=memory_id,
            user_id=user_id,
            query_summary=self._truncate_query(query),
            embedding=np.asarray(embedding).flatten(),
            regions_used=regions_used,
            was_successful=was_successful,
            reward=reward,
            importance_score=importance_score,
            access_count=0,
            created_at=datetime.now(),
            last_accessed=None
        )
        
        # Store in memory
        self.memories[user_id].append(record)
        
        # Store embedding for fast lookup
        embedding_key = f"{user_id}_{memory_id}"
        self.embeddings[embedding_key] = record.embedding
        
        # Enforce memory limit per user
        if len(self.memories[user_id]) > self.config.max_memories_per_user:
            self._consolidate_memories(user_id)
        
        # Store in ChromaDB if available
        if self.config.use_chromadb and self.chroma_collection is not None:
            try:
                self.chroma_collection.add(
                    ids=[memory_id],
                    embeddings=[record.embedding.tolist()],
                    metadatas=[{
                        "user_id": user_id,
                        "query_summary": record.query_summary,
                        "regions_used": ",".join(regions_used),
                        "was_successful": str(was_successful),
                        "reward": str(reward),
                        "importance_score": str(importance_score)
                    }]
                )
            except Exception as e:
                print(f"⚠️ ChromaDB store failed: {e}")
        
        self.total_stores += 1
        
        return record
    
    def retrieve(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        n_results: int = 5,
        min_similarity: Optional[float] = None
    ) -> RetrievalResult:
        """
        Retrieve similar memories for a query.
        
        Args:
            user_id: User identifier
            query_embedding: Query embedding to match
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            RetrievalResult with matching memories
        """
        start_time = time.time()
        
        if min_similarity is None:
            min_similarity = self.config.similarity_threshold
        
        query_embedding = np.asarray(query_embedding).flatten()
        
        # Get user's memories
        user_memories = self.memories.get(user_id, [])
        
        if not user_memories:
            return RetrievalResult(
                memories=[],
                similarity_scores=[],
                suggested_regions=[],
                confidence=0.0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
        
        # Calculate similarities
        similarities = []
        for memory in user_memories:
            sim = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((memory, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and limit
        filtered = [
            (m, s) for m, s in similarities 
            if s >= min_similarity
        ][:n_results]
        
        # Extract results
        memories = [m for m, s in filtered]
        scores = [s for m, s in filtered]
        
        # Update access counts
        for memory in memories:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        # Calculate suggested regions from successful memories
        suggested_regions = self._calculate_suggested_regions(memories)
        
        # Calculate confidence
        if scores:
            confidence = float(np.mean(scores))
            self.total_hits += 1
        else:
            confidence = 0.0
        
        self.total_retrievals += 1
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            memories=memories,
            similarity_scores=scores,
            suggested_regions=suggested_regions,
            confidence=confidence,
            retrieval_time_ms=retrieval_time_ms
        )
    
    def _calculate_importance(
        self,
        was_successful: bool,
        reward: float,
        embedding: np.ndarray
    ) -> float:
        """Calculate importance score for a memory"""
        base_importance = 0.5
        
        # Success bonus
        if was_successful:
            base_importance += 0.2
        
        # Reward bonus
        base_importance += reward * 0.3
        
        # Clip to valid range
        return float(np.clip(base_importance, 0.0, 1.0))
    
    def _calculate_suggested_regions(
        self,
        memories: List[MemoryRecord]
    ) -> List[str]:
        """Calculate suggested regions from memories"""
        if not memories:
            return []
        
        # Count region occurrences weighted by success and similarity
        region_scores = defaultdict(float)
        
        for memory in memories:
            weight = 1.0
            if memory.was_successful:
                weight *= 1.5
            weight *= (1.0 + memory.reward)
            
            for region in memory.regions_used:
                region_scores[region] += weight
        
        # Sort by score
        sorted_regions = sorted(
            region_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [region for region, score in sorted_regions]
    
    def _consolidate_memories(self, user_id: str):
        """
        Consolidate memories when limit is exceeded.
        Keeps important memories, removes less important ones.
        """
        memories = self.memories[user_id]
        
        # Sort by importance (higher = keep)
        memories.sort(key=lambda m: m.importance_score, reverse=True)
        
        # Keep top memories
        keep_count = int(self.config.max_memories_per_user * 0.8)
        
        # Remove excess memories
        removed = memories[keep_count:]
        self.memories[user_id] = memories[:keep_count]
        
        # Clean up embeddings
        for memory in removed:
            embedding_key = f"{user_id}_{memory.id}"
            if embedding_key in self.embeddings:
                del self.embeddings[embedding_key]
    
    def decay_importance(self, user_id: Optional[str] = None):
        """
        Apply importance decay to memories.
        Call periodically to let old memories fade.
        """
        decay_rate = self.config.importance_decay_rate
        
        users = [user_id] if user_id else list(self.memories.keys())
        
        for uid in users:
            for memory in self.memories.get(uid, []):
                memory.importance_score *= (1 - decay_rate)
                
                # Boost if frequently accessed
                if memory.access_count > 5:
                    memory.importance_score = min(1.0, memory.importance_score + 0.01)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user"""
        memories = self.memories.get(user_id, [])
        
        if not memories:
            return {
                "total_memories": 0,
                "successful_memories": 0,
                "avg_importance": 0.0,
                "regions_used": {}
            }
        
        successful = sum(1 for m in memories if m.was_successful)
        avg_importance = np.mean([m.importance_score for m in memories])
        
        region_counts = defaultdict(int)
        for memory in memories:
            for region in memory.regions_used:
                region_counts[region] += 1
        
        return {
            "total_memories": len(memories),
            "successful_memories": successful,
            "success_rate": successful / len(memories) if memories else 0,
            "avg_importance": float(avg_importance),
            "regions_used": dict(region_counts)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall hippocampus statistics"""
        total_memories = sum(len(m) for m in self.memories.values())
        
        return {
            "total_users": len(self.memories),
            "total_memories": total_memories,
            "total_stores": self.total_stores,
            "total_retrievals": self.total_retrievals,
            "total_hits": self.total_hits,
            "hit_rate": self.total_hits / max(1, self.total_retrievals),
            "storage_backend": "chromadb" if self.config.use_chromadb else "in_memory",
            "max_memories_per_user": self.config.max_memories_per_user
        }
    
    def clear_user(self, user_id: str):
        """Clear all memories for a user"""
        if user_id in self.memories:
            # Clean up embeddings
            for memory in self.memories[user_id]:
                embedding_key = f"{user_id}_{memory.id}"
                if embedding_key in self.embeddings:
                    del self.embeddings[embedding_key]
            
            del self.memories[user_id]
    
    def clear_all(self):
        """Clear all memories"""
        self.memories.clear()
        self.embeddings.clear()
        self.total_stores = 0
        self.total_retrievals = 0
        self.total_hits = 0
    
    def __repr__(self) -> str:
        total = sum(len(m) for m in self.memories.values())
        return f"Hippocampus(users={len(self.memories)}, memories={total})"


def create_hippocampus(
    max_memories: int = 1000,
    use_chromadb: bool = False,
    chromadb_path: str = "./chromadb_data"
) -> Hippocampus:
    """
    Factory function to create a Hippocampus.
    
    Args:
        max_memories: Maximum memories per user
        use_chromadb: Whether to use ChromaDB backend
        chromadb_path: Path for ChromaDB persistence
    
    Returns:
        Configured Hippocampus instance
    """
    config = HippocampusConfig(
        max_memories_per_user=max_memories,
        use_chromadb=use_chromadb,
        chromadb_path=chromadb_path
    )
    return Hippocampus(config)
