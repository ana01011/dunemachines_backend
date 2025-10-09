-- Add token tracking for neurochemistry feature
CREATE TABLE IF NOT EXISTS user_tokens (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    daily_tokens INTEGER DEFAULT 1000,
    tokens_used INTEGER DEFAULT 0,
    last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subscription_tier TEXT DEFAULT 'free',
    total_tokens_purchased INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for token reset checks
CREATE INDEX idx_user_tokens_reset ON user_tokens(last_reset);

-- Neurochemical states table
CREATE TABLE IF NOT EXISTS neurochemical_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
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
    
    -- Knowledge access triggers (for future web search feature)
    triggered_web_search BOOLEAN DEFAULT FALSE,
    triggered_textbook_search BOOLEAN DEFAULT FALSE,
    search_query TEXT,
    
    -- Metadata
    event_context JSONB,
    
    -- Add index for efficient queries
    CONSTRAINT neurochemical_states_user_timestamp_idx 
        UNIQUE (user_id, timestamp)
);

CREATE INDEX idx_neurochemical_states_user ON neurochemical_states(user_id, timestamp DESC);

-- Neurochemical events tracking
CREATE TABLE IF NOT EXISTS neurochemical_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_id TEXT NOT NULL UNIQUE,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
    
    -- Knowledge access (future feature)
    required_web_search BOOLEAN DEFAULT FALSE,
    required_textbook_access BOOLEAN DEFAULT FALSE,
    knowledge_sources_used TEXT[],
    
    metadata JSONB
);

CREATE INDEX idx_neurochemical_events_user ON neurochemical_events(user_id, timestamp DESC);

-- Learning patterns for each user
CREATE TABLE IF NOT EXISTS neurochemical_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pattern_type TEXT NOT NULL,
    pattern_data JSONB,
    confidence FLOAT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Future: Knowledge preferences
    preferred_knowledge_sources TEXT[],
    search_success_rate FLOAT,
    
    UNIQUE (user_id, pattern_type)
);

-- WebSocket sessions for real-time mood streaming
CREATE TABLE IF NOT EXISTS websocket_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    connection_id TEXT NOT NULL UNIQUE,
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    client_info JSONB
);

CREATE INDEX idx_websocket_sessions_user ON websocket_sessions(user_id, is_active);

-- Add neurochemistry fields to messages table if needed
ALTER TABLE messages 
ADD COLUMN IF NOT EXISTS neurochemical_state_id UUID REFERENCES neurochemical_states(id),
ADD COLUMN IF NOT EXISTS mood_indicators JSONB,
ADD COLUMN IF NOT EXISTS tokens_used INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS used_neurochemistry BOOLEAN DEFAULT FALSE;

-- Function to reset daily tokens
CREATE OR REPLACE FUNCTION reset_daily_tokens()
RETURNS void AS $$
BEGIN
    UPDATE user_tokens 
    SET tokens_used = 0, 
        last_reset = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE last_reset < CURRENT_DATE;
END;
$$ LANGUAGE plpgsql;

-- Create a view for user token status
CREATE OR REPLACE VIEW user_token_status AS
SELECT 
    u.id as user_id,
    u.email,
    u.username,
    COALESCE(ut.daily_tokens, 1000) as daily_tokens,
    COALESCE(ut.tokens_used, 0) as tokens_used,
    COALESCE(ut.daily_tokens - ut.tokens_used, 1000) as tokens_remaining,
    COALESCE(ut.subscription_tier, 'free') as subscription_tier,
    CASE 
        WHEN ut.last_reset < CURRENT_DATE THEN true
        ELSE false
    END as needs_reset
FROM users u
LEFT JOIN user_tokens ut ON u.id = ut.user_id;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO sarah_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sarah_user;