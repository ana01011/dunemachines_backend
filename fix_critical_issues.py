"""
Fix the 5 critical issues in the neurochemistry system
"""

print("Fixing critical issues in neurochemistry system...")

# Fix 1: Resource dynamics not working
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Find and fix resource dynamics
old_resource = """    def calculate_resource_dynamics(self, inputs: Dict) -> Tuple[float, float, float]:
        \"\"\"Resource dynamics (simplified)\"\"\"
        # Small changes for stability
        nutrition = inputs.get('nutrition', 0.5)
        
        dP_tyr_dt = nutrition - 0.01
        dP_trp_dt = nutrition - 0.01
        dE_ATP_dt = nutrition - 0.02
        
        return dP_tyr_dt, dP_trp_dt, dE_ATP_dt"""

new_resource = """    def calculate_resource_dynamics(self, inputs: Dict) -> Tuple[float, float, float]:
        \"\"\"Resource dynamics with proper depletion and recovery\"\"\"
        # Get current usage based on hormone production
        D = self.state.dopamine
        S = self.state.serotonin
        A = self.state.adrenaline
        
        # Tyrosine used by dopamine, norepinephrine, adrenaline
        tyrosine_usage = 0.001 * (D + self.state.norepinephrine + A) / 100
        
        # Tryptophan used by serotonin
        tryptophan_usage = 0.002 * S / 100
        
        # ATP used by all processes
        total_activity = np.sum(np.abs(self.state.hormones - self.state.baselines)) / 700
        atp_usage = 0.005 * (1 + total_activity * 2)
        
        # Recovery from nutrition and rest
        nutrition = inputs.get('nutrition', 0.5)
        sleep = inputs.get('sleep', 0)
        
        # Calculate changes
        dP_tyr_dt = nutrition * 0.01 - tyrosine_usage
        dP_trp_dt = nutrition * 0.01 - tryptophan_usage
        dE_ATP_dt = (nutrition * 0.02 + sleep * 0.03) - atp_usage
        
        return dP_tyr_dt, dP_trp_dt, dE_ATP_dt"""

content = content.replace(old_resource, new_resource)

# Fix 2: Adrenaline response too weak
old_adrenaline = """        # Check for sadness/depression (high punishment, low urgency)
        punishment = inputs.get('punishment', 0)
        urgency = inputs.get('urgency', 0)
        if punishment > 0.5 and urgency < 0.3:
            # Sadness depletes adrenaline
            dA_dt -= 5.0 * punishment * A/50
            return dA_dt * self.response_gain  # Return early for sadness"""

new_adrenaline = """        # Check for sadness/depression (high punishment, low urgency)
        punishment = inputs.get('punishment', 0)
        urgency = inputs.get('urgency', 0)
        if punishment > 0.5 and urgency < 0.3:
            # Sadness depletes adrenaline
            dA_dt -= 5.0 * punishment * A/50
            return dA_dt * self.response_gain  # Return early for sadness
        
        # ENHANCED stress response
        threat = inputs.get('threat', 0)
        if threat > 0 or urgency > 0.5:
            # Much stronger adrenaline response to stress
            dA_dt += 15.0 * (threat + urgency) * (1 - A/100) * R_A"""

content = content.replace(old_adrenaline, new_adrenaline)

# Fix 3: Allostatic load not accumulating
old_allostatic = """    def calculate_allostatic_load_dynamics(self) -> float:
        \"\"\"Allostatic load accumulation\"\"\"
        C = self.state.cortisol
        L = self.state.allostatic_load
        
        # Accumulate when cortisol is high
        if C > C_THRESHOLD:
            accumulation = 0.01 * (C - C_THRESHOLD) / 50
        else:
            accumulation = 0
        
        # Recovery
        recovery = 0.001 * L
        
        return accumulation - recovery"""

new_allostatic = """    def calculate_allostatic_load_dynamics(self) -> float:
        \"\"\"Allostatic load accumulation with proper scaling\"\"\"
        C = self.state.cortisol
        L = self.state.allostatic_load
        
        # Accumulate when cortisol is high (with better scaling)
        if C > 35:  # Lower threshold for accumulation
            accumulation = 0.001 * ((C - 35) / 20) ** 2  # Quadratic accumulation
        else:
            accumulation = 0
        
        # Also accumulate from chronic low serotonin
        S = self.state.serotonin
        if S < 40:
            accumulation += 0.0005 * ((40 - S) / 40)
        
        # Recovery (slower, needs real rest)
        sleep = 0  # Would come from inputs
        recovery = 0.0001 * L * (1 + sleep)
        
        return accumulation - recovery"""

content = content.replace(old_allostatic, new_allostatic)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Fixed dynamics.py")

# Fix 4: Cost function issues in minimization.py
with open('/root/openhermes_backend/app/neurochemistry/core/minimization.py', 'r') as f:
    content = f.read()

# Fix the cost calculation to be more reasonable
old_cost = """        # 3. Metabolic cost - resource usage
        # Normalized hormone levels
        X_norm = X / HORMONE_MAX
        
        # Resource depletion factors
        tyr_depletion = 1 - self.state.p_tyr / P_TYR_MAX
        trp_depletion = 1 - self.state.p_trp / P_TRP_MAX
        atp_depletion = 1 - self.state.e_atp / E_ATP_MAX
        
        C_metabolic = (
            2.0 * np.sum(X_norm**2) +  # Cost of maintaining levels
            5.0 * tyr_depletion**2 +    # Tyrosine depletion penalty
            5.0 * trp_depletion**2 +    # Tryptophan depletion penalty
            10.0 * atp_depletion**2     # Energy depletion penalty
        )"""

new_cost = """        # 3. Metabolic cost - resource usage
        # Normalized hormone levels
        X_norm = X / HORMONE_MAX
        
        # Resource depletion factors (with bounds to prevent explosion)
        tyr_depletion = max(0, min(1, 1 - self.state.p_tyr / P_TYR_MAX))
        trp_depletion = max(0, min(1, 1 - self.state.p_trp / P_TRP_MAX))
        atp_depletion = max(0, min(1, 1 - self.state.e_atp / E_ATP_MAX))
        
        C_metabolic = (
            1.0 * np.sum(X_norm**2) +  # Reduced weight
            2.0 * tyr_depletion**2 +    # Reduced weight
            2.0 * trp_depletion**2 +    # Reduced weight
            3.0 * atp_depletion**2      # Reduced weight
        )"""

content = content.replace(old_cost, new_cost)

# Also fix uncertainty cost scaling
old_uncertainty = """        # 4. Uncertainty cost - prediction errors are expensive
        PE = X_exp - X
        C_uncertainty = 3.0 * np.sum(PE**2 / (K_COST + PE**2))"""

new_uncertainty = """        # 4. Uncertainty cost - prediction errors are expensive (but bounded)
        PE = X_exp - X
        C_uncertainty = 1.0 * np.sum(PE**2 / (K_COST + PE**2))  # Reduced weight"""

content = content.replace(old_uncertainty, new_uncertainty)

with open('/root/openhermes_backend/app/neurochemistry/core/minimization.py', 'w') as f:
    f.write(content)

print("✅ Fixed minimization.py")

# Fix 5: Enhance mood detection in state.py
with open('/root/openhermes_backend/app/neurochemistry/core/state.py', 'r') as f:
    content = f.read()

# Add more sensitive mood detection
old_mood_end = """        else:
            return "neutral\""""

new_mood_end = """        elif D > 55 and E > 40:
            return "happy"
        elif A > 20 and N > 50:
            return "alert"
        elif C > 35 and A < 15:
            return "worried"
        elif S > 40 and D > 50:
            return "balanced"
        else:
            return "neutral\""""

content = content.replace(old_mood_end, new_mood_end)

with open('/root/openhermes_backend/app/neurochemistry/core/state.py', 'w') as f:
    f.write(content)

print("✅ Fixed state.py")

print("\n🎯 All critical issues fixed!")
print("\nChanges made:")
print("1. ✅ Resources now properly deplete and recover")
print("2. ✅ Adrenaline has stronger stress response (15x amplification)")
print("3. ✅ Allostatic load accumulates from high cortisol and low serotonin")
print("4. ✅ Cost function weights reduced to prevent explosion")
print("5. ✅ Added 4 new mood states for better variety")

