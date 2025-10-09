"""
Hormone-based capability triggering system
Different hormone combinations activate different AI capabilities
"""

from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class CapabilityTrigger:
    """Defines when and how a capability activates"""
    name: str
    description: str
    check_condition: Callable  # Function that checks hormone levels
    priority: int  # Higher priority triggers first
    tools: List[str]  # What tools/APIs to activate
    processing_style: Dict[str, any]  # How to process differently

# Define capability triggers based on hormone patterns
CAPABILITY_TRIGGERS = [
    
    # DEEP RESEARCH MODE
    # High cortisol (40-70) = Need careful analysis
    # Moderate adrenaline = Urgent but not panic
    CapabilityTrigger(
        name="deep_research",
        description="Thorough investigation with multiple sources",
        check_condition=lambda h: 40 <= h.cortisol <= 70 and h.adrenaline < 50,
        priority=8,
        tools=["web_search", "documentation_search", "fact_checker", "citation_finder"],
        processing_style={
            "search_depth": 10,  # Check top 10 results
            "verify_facts": True,
            "cross_reference": True,
            "include_citations": True,
            "summary_style": "comprehensive"
        }
    ),
    
    # CRISIS RESPONSE MODE
    # Very high cortisol + adrenaline = Emergency
    CapabilityTrigger(
        name="crisis_response",
        description="Immediate problem-solving, skip deep analysis",
        check_condition=lambda h: h.cortisol > 70 and h.adrenaline > 60,
        priority=10,  # Highest priority
        tools=["quick_search", "stackoverflow", "error_database", "quick_fix_finder"],
        processing_style={
            "search_depth": 3,  # Just top 3 results
            "verify_facts": False,  # No time
            "response_speed": "immediate",
            "solution_style": "quick_fixes",
            "include_warnings": True
        }
    ),
    
    # CREATIVE EXPLORATION MODE
    # High dopamine + serotonin, low cortisol = Playful exploration
    CapabilityTrigger(
        name="creative_exploration",
        description="Explore novel ideas and connections",
        check_condition=lambda h: h.dopamine > 60 and h.serotonin > 60 and h.cortisol < 40,
        priority=6,
        tools=["brainstorm_generator", "analogy_finder", "inspiration_search", "trend_explorer"],
        processing_style={
            "creativity_level": "high",
            "explore_tangents": True,
            "generate_alternatives": 5,
            "use_metaphors": True,
            "risk_tolerance": "high"
        }
    ),
    
    # CAREFUL LEARNING MODE
    # Low dopamine + moderate cortisol = Need to learn carefully
    CapabilityTrigger(
        name="careful_learning",
        description="Step-by-step learning with validation",
        check_condition=lambda h: h.dopamine < 40 and 40 <= h.cortisol <= 60,
        priority=7,
        tools=["tutorial_finder", "documentation", "example_finder", "concept_explainer"],
        processing_style={
            "explanation_depth": "beginner",
            "include_examples": True,
            "step_by_step": True,
            "validate_understanding": True,
            "pace": "slow"
        }
    ),
    
    # SOCIAL COLLABORATION MODE
    # High oxytocin + moderate dopamine = Team oriented
    CapabilityTrigger(
        name="social_collaboration",
        description="Collaborative problem solving with empathy",
        check_condition=lambda h: h.oxytocin > 60 and h.dopamine > 40,
        priority=5,
        tools=["team_solutions", "best_practices", "community_wisdom", "empathy_checker"],
        processing_style={
            "tone": "collaborative",
            "acknowledge_feelings": True,
            "suggest_teamwork": True,
            "include_encouragement": True,
            "solution_style": "we_together"
        }
    ),
    
    # ANALYTICAL PRECISION MODE
    # Moderate cortisol + low adrenaline + high serotonin = Methodical
    CapabilityTrigger(
        name="analytical_precision",
        description="Systematic, precise analysis",
        check_condition=lambda h: 30 <= h.cortisol <= 50 and h.adrenaline < 30 and h.serotonin > 50,
        priority=6,
        tools=["data_analyzer", "logic_checker", "proof_validator", "benchmark_runner"],
        processing_style={
            "analysis_type": "systematic",
            "include_metrics": True,
            "validate_logic": True,
            "show_reasoning": True,
            "precision": "high"
        }
    ),
    
    # DEFENSIVE PROTECTION MODE
    # High cortisol + low oxytocin = Defensive/suspicious
    CapabilityTrigger(
        name="defensive_protection",
        description="Cautious validation and security checking",
        check_condition=lambda h: h.cortisol > 60 and h.oxytocin < 30,
        priority=7,
        tools=["security_scanner", "vulnerability_checker", "risk_assessor", "trust_verifier"],
        processing_style={
            "trust_level": "low",
            "verify_everything": True,
            "include_warnings": True,
            "suggest_safeguards": True,
            "paranoia_level": "high"
        }
    ),
    
    # FLOW STATE MODE
    # Balanced hormones with slight elevation = Optimal performance
    CapabilityTrigger(
        name="flow_state",
        description="Optimal balanced processing",
        check_condition=lambda h: (
            40 <= h.dopamine <= 60 and
            25 <= h.cortisol <= 40 and
            20 <= h.adrenaline <= 40 and
            50 <= h.serotonin <= 70
        ),
        priority=9,
        tools=["all_tools"],  # Access to everything
        processing_style={
            "balance": "optimal",
            "adapt_to_user": True,
            "efficiency": "high",
            "quality": "high",
            "flexibility": "high"
        }
    ),
    
    # EXHAUSTED RECOVERY MODE
    # Very low everything = Need rest/simple tasks
    CapabilityTrigger(
        name="exhausted_recovery",
        description="Minimal processing, basic responses only",
        check_condition=lambda h: (
            h.dopamine < 30 and
            h.serotonin < 30 and
            h.adrenaline < 20
        ),
        priority=4,
        tools=["basic_search", "simple_answers"],
        processing_style={
            "complexity": "minimal",
            "response_length": "short",
            "avoid_deep_analysis": True,
            "suggest_break": True,
            "energy_conservation": True
        }
    )
]

class CapabilityOrchestrator:
    """Orchestrates capability activation based on hormonal state"""
    
    @staticmethod
    def get_active_capabilities(state) -> List[CapabilityTrigger]:
        """Get all currently active capabilities sorted by priority"""
        active = []
        
        for trigger in CAPABILITY_TRIGGERS:
            if trigger.check_condition(state):
                active.append(trigger)
        
        # Sort by priority (highest first)
        active.sort(key=lambda x: x.priority, reverse=True)
        return active
    
    @staticmethod
    def get_primary_mode(state) -> Optional[CapabilityTrigger]:
        """Get the highest priority active capability"""
        active = CapabilityOrchestrator.get_active_capabilities(state)
        return active[0] if active else None
    
    @staticmethod
    def get_tool_list(state) -> List[str]:
        """Get all tools that should be active"""
        active = CapabilityOrchestrator.get_active_capabilities(state)
        tools = set()
        
        for capability in active:
            tools.update(capability.tools)
        
        return list(tools)
    
    @staticmethod
    def get_processing_parameters(state) -> Dict:
        """Merge processing parameters from all active capabilities"""
        active = CapabilityOrchestrator.get_active_capabilities(state)
        
        if not active:
            return {"mode": "default"}
        
        # Start with highest priority
        params = {"mode": active[0].name}
        params.update(active[0].processing_style)
        
        # Layer in other active capabilities
        for capability in active[1:]:
            params[f"also_{capability.name}"] = True
            
        return params

def demonstrate_triggers():
    """Show how different hormone states trigger different capabilities"""
    
    print("=" * 70)
    print("🧠 HORMONE-TRIGGERED CAPABILITIES")
    print("=" * 70)
    
    # Test different hormone states
    test_states = [
        ("Normal", {"dopamine": 50, "cortisol": 30, "adrenaline": 20, "serotonin": 60, "oxytocin": 40}),
        ("Crisis", {"dopamine": 30, "cortisol": 80, "adrenaline": 75, "serotonin": 30, "oxytocin": 25}),
        ("Deep Focus", {"dopamine": 45, "cortisol": 55, "adrenaline": 30, "serotonin": 50, "oxytocin": 35}),
        ("Creative", {"dopamine": 75, "cortisol": 25, "adrenaline": 35, "serotonin": 70, "oxytocin": 50}),
        ("Learning", {"dopamine": 35, "cortisol": 50, "adrenaline": 20, "serotonin": 40, "oxytocin": 45}),
        ("Collaborative", {"dopamine": 55, "cortisol": 35, "adrenaline": 25, "serotonin": 60, "oxytocin": 75}),
        ("Defensive", {"dopamine": 25, "cortisol": 70, "adrenaline": 50, "serotonin": 25, "oxytocin": 20}),
        ("Flow", {"dopamine": 55, "cortisol": 35, "adrenaline": 30, "serotonin": 65, "oxytocin": 50}),
    ]
    
    for name, hormones in test_states:
        # Create mock state
        class State:
            pass
        
        state = State()
        for h, v in hormones.items():
            setattr(state, h, v)
        
        # Get active capabilities
        primary = CapabilityOrchestrator.get_primary_mode(state)
        tools = CapabilityOrchestrator.get_tool_list(state)
        params = CapabilityOrchestrator.get_processing_parameters(state)
        
        print(f"\n{'─' * 60}")
        print(f"📊 {name} State")
        print(f"   Hormones: D={hormones['dopamine']} C={hormones['cortisol']} "
              f"A={hormones['adrenaline']} S={hormones['serotonin']} O={hormones['oxytocin']}")
        
        if primary:
            print(f"\n   🎯 Primary Mode: {primary.name}")
            print(f"      Description: {primary.description}")
            print(f"      Priority: {primary.priority}")
            
        print(f"\n   🔧 Active Tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"      • {tool}")
        
        print(f"\n   ⚙️ Processing Style:")
        for key, value in list(params.items())[:4]:  # Show first 4
            print(f"      • {key}: {value}")

if __name__ == "__main__":
    demonstrate_triggers()
