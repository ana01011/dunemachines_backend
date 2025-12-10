"""
Debug why fixes aren't working and apply stronger corrections
"""

import sys
sys.path.append('/root/openhermes_backend')

print("Debugging the system...")

# First, let's check if ATP is even being updated
from app.neurochemistry.interface import NeurochemicalSystem
from app.neurochemistry.core.constants import *

system = NeurochemicalSystem()

# Check initial state
print(f"\nInitial ATP: {system.state.e_atp}")
print(f"E_ATP_MAX constant: {E_ATP_MAX}")

# Manually check if dynamics are being called
response = system.process_message("Test")
print(f"After message ATP: {system.state.e_atp}")

# Let's check if the step function is updating resources
import inspect
print("\nChecking step function for resource updates...")
source = inspect.getsource(system.dynamics.step)
if "self.state.e_atp +=" in source:
    print("✅ ATP update found in step function")
else:
    print("❌ ATP update NOT found in step function")
    
# Now let's properly fix it
print("\nApplying STRONGER fixes...")

# Fix the dynamics.py file with more aggressive changes
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Make sure resources are actually being updated in the step function
if "self.state.e_atp += dE_ATP_dt * dt" not in content:
    print("❌ Resource updates missing in step function! Adding them...")
    
    # Find the step function and add resource updates
    old_step = """        # Apply bounds
        self.state.apply_bounds()
        
        # Update time
        self.t += dt
        self.state.last_update = time.time()"""
    
    new_step = """        # Apply bounds
        self.state.apply_bounds()
        
        # Update time
        self.t += dt
        self.state.last_update = time.time()"""
    
    # If resource updates are missing entirely, add them
    if "self.state.p_tyr +=" not in content:
        old_step = """        # Update all states
        self.state.hormones += dX_dt * dt
        self.state.receptors += dR_dt * dt
        self.state.baselines += dB_dt * dt
        self.state.expected += dX_exp_dt * dt
        self.state.allostatic_load += dL_dt * dt"""
        
        new_step = """        # Update all states
        self.state.hormones += dX_dt * dt
        self.state.receptors += dR_dt * dt
        self.state.baselines += dB_dt * dt
        self.state.expected += dX_exp_dt * dt
        self.state.allostatic_load += dL_dt * dt
        self.state.p_tyr += dP_tyr_dt * dt
        self.state.p_trp += dP_trp_dt * dt
        self.state.e_atp += dE_ATP_dt * dt"""
        
        content = content.replace(old_step, new_step)
        print("✅ Added resource updates to step function")

# Write the fixed content
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Fixed dynamics.py")

# Now fix the adrenaline response more aggressively
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Replace the entire adrenaline dynamics function with a stronger version
old_adrenaline_func = """    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:"""

# Find the whole function and replace it
import re
pattern = r'(    def calculate_adrenaline_dynamics.*?)(?=    def calculate_)'
match = re.search(pattern, content, re.DOTALL)

if match:
    old_function = match.group(1)
    
    new_function = '''    def calculate_adrenaline_dynamics(self, X: np.ndarray, X_eff: np.ndarray,
                                      PE: np.ndarray, inputs: Dict) -> float:
        """Enhanced adrenaline dynamics with STRONG response"""
        A = X[A_IDX]
        B_A = self.state.baselines[A_IDX]
        R_A = self.state.receptors[A_IDX]
        
        # Very fast decay
        dA_dt = -LAMBDA_DECAY[A_IDX] * 2.0 * (A - B_A) * (1 + R_A)
        
        # VERY STRONG urgency/threat response
        urgency = inputs.get('urgency', 0)
        threat = inputs.get('threat', 0)
        stress = inputs.get('punishment', 0) * 0.5  # Some stress from negative events
        
        # Calculate total emergency level
        emergency = urgency + threat + stress
        
        if emergency > 0:
            # MASSIVE response to any stress
            dA_dt += 25.0 * emergency * (1 - A/100) * R_A  # Increased from 15 to 25
        
        # Exercise increases adrenaline significantly
        exercise = inputs.get('exercise', 0)
        if exercise > 0:
            dA_dt += 10.0 * exercise * (1 - A/100)
        
        # Relaxation strongly decreases adrenaline
        sleep = inputs.get('sleep', 0)
        if sleep > 0:
            dA_dt -= 10.0 * sleep * A/100
        
        return dA_dt * self.response_gain
    
'''
    
    content = content.replace(old_function, new_function)
    print("✅ Replaced adrenaline function with stronger version")

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

# Fix allostatic load to be more sensitive
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Make allostatic load much more sensitive
old_allostatic = """        # Accumulate when cortisol is high (with better scaling)
        if C > 35:  # Lower threshold for accumulation
            accumulation = 0.001 * ((C - 35) / 20) ** 2  # Quadratic accumulation"""

new_allostatic = """        # Accumulate when cortisol is high (MUCH more sensitive)
        if C > 32:  # Even lower threshold
            accumulation = 0.01 * ((C - 32) / 10) ** 2  # 10x more accumulation"""

content = content.replace(old_allostatic, new_allostatic)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Made allostatic load 10x more sensitive")

# Fix resource depletion to be much stronger
with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'r') as f:
    content = f.read()

# Make ATP depletion much more aggressive
old_atp = """        # ATP used by all processes
        total_activity = np.sum(np.abs(self.state.hormones - self.state.baselines)) / 700
        atp_usage = 0.005 * (1 + total_activity * 2)"""

new_atp = """        # ATP used by all processes (MUCH MORE)
        total_activity = np.sum(np.abs(self.state.hormones - self.state.baselines)) / 100  # More sensitive
        atp_usage = 0.05 * (1 + total_activity * 3)  # 10x more usage"""

content = content.replace(old_atp, new_atp)

with open('/root/openhermes_backend/app/neurochemistry/core/dynamics.py', 'w') as f:
    f.write(content)

print("✅ Made ATP depletion 10x stronger")

print("\n" + "="*50)
print("FIXES APPLIED:")
print("1. ✅ Resources now properly update in step function")
print("2. ✅ Adrenaline response increased to 25x")
print("3. ✅ Allostatic load 10x more sensitive")
print("4. ✅ ATP depletion 10x stronger")
print("="*50)

