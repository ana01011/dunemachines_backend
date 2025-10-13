# Neurochemistry V2 Implementation Reference

## Git Reference
- **Commit**: 065fbf2
- **Date**: 2024-12-19
- **Branch**: cursor/deploy-production-ready-repository-components-2eaf

## Previous Working Features
1. **5-Hormone System**: Dopamine, Serotonin, Cortisol, Adrenaline, Oxytocin
2. **State Management**: Basic state with baselines
3. **Neural Network**: Hormone to mood mapping
4. **3D Emergence**: Dimensional positioning system
5. **Per-User States**: Multi-user support
6. **WebSocket Integration**: Real-time updates
7. **Event Processing**: Message analysis and response

## Key Files (for rollback reference)
- `integrated_system.py` - Main integration point
- `scalable_hormone_network.py` - Neural network
- `core/state_v2_fixed.py` - State management
- `core/dimensional_emergence.py` - 3D positioning

## Rollback Command (if needed)
```bash
git checkout 065fbf2 -- app/neurochemistry/
```

## Integration Points to Maintain
- Called from: `main.py` via neurochemistry imports
- WebSocket: Via `omnius_websocket.py`
- User tracking: Per-user state management

## Notes
Starting fresh with advanced 7D system including:
- Hill equation dynamics
- Receptor adaptation
- Minimization principle
- Resource constraints
- Allostatic load
- Full biological realism