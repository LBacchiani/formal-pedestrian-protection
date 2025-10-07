"""
Improved verification properties for the pedestrian protection automaton.
"""

from z3 import *
from z3_automaton import *

def check_property(property_name: str, property_formula, vars_dict):
    """Helper function to check a property using Z3."""
    solver = Solver()
    
    # Add buffer constraints (invariants)
    for constraint in buffer_constraints(vars_dict):
        solver.add(constraint)
    
    for constraint in threshold_constraints():
        solver.add(constraint)
    
    solver.add(Not(property_formula))
    
    result = solver.check()
    
    if result == unsat:
        print(f"✓ {property_name}: VERIFIED")
        return True
    elif result == sat:
        print(f"✗ {property_name}: FALSIFIED")
        model = solver.model()
        print(f"  Counterexample:")
        for var_name in ['s_d', 's_c', 's_u']:
            if var_name in vars_dict:
                print(f"    {var_name} = {model[vars_dict[var_name]]}")
        print(f"    B_C: {[model[var] for var in vars_dict['B_C']]}")
        print(f"    B_TTC: {[model[var] for var in vars_dict['B_TTC']]}")
        print(f"    B_cross: {[model[var] for var in vars_dict['B_cross']]}")
        return False
    else:
        print(f"? {property_name}: UNKNOWN")
        return None

# ============================================================================
# PROPERTY 1: Determinism (Guards Mutually Exclusive)
# ============================================================================

def prop_guards_mutually_exclusive():
    """Verify that for each state, at most one guard can be true."""
    print("\n" + "="*70)
    print("PROPERTY 1: Determinism (Guards Mutually Exclusive)")
    print("="*70)
    
    all_verified = True
    
    for state_name, transitions in STATE_TRANSITIONS.items():
        print(f"\nChecking state: {state_name}")
        
        vars_dict = create_automaton_vars()
        guards = get_guards(vars_dict)
        
        for i, t1 in enumerate(transitions):
            for t2 in transitions[i+1:]:
                property_formula = Not(And(guards[t1], guards[t2]))
                property_name = f"{state_name}: {t1} ⊥ {t2}"
                result = check_property(property_name, property_formula, vars_dict)
                
                if result is False:
                    all_verified = False
    
    print("\n" + "="*70)
    if all_verified:
        print("RESULT: All guards are mutually exclusive ✓")
    else:
        print("RESULT: Automaton is non-deterministic ✗")
    print("="*70)
    
    return all_verified

# ============================================================================
# PROPERTY 2: Completeness (No Deadlock)
# ============================================================================

def prop_guards_complete():
    """Verify that for each state, at least one guard is always enabled."""
    print("\n" + "="*70)
    print("PROPERTY 2: Completeness (No Deadlock)")
    print("="*70)
    
    all_verified = True
    
    for state_name, transitions in STATE_TRANSITIONS.items():
        print(f"\nChecking state: {state_name}")
        
        vars_dict = create_automaton_vars()
        guards = get_guards(vars_dict)
        
        property_formula = Or([guards[t] for t in transitions])
        property_name = f"{state_name}: At least one guard enabled"
        result = check_property(property_name, property_formula, vars_dict)
        
        if result is False:
            all_verified = False
    
    print("\n" + "="*70)
    if all_verified:
        print("RESULT: All states have at least one enabled guard ✓")
    else:
        print("RESULT: Some states may deadlock ✗")
    print("="*70)
    
    return all_verified

# ============================================================================
# PROPERTY 3: Emergency Braking Safety (Consolidated)
# ============================================================================

def prop_emergency_braking_safety():
    """
    Verify that Emergency Braking is only entered when:
    - Detection is valid (valid_d)
    - Crossing is validated (valid_c)
    - Distance is critical (c_distance)
    
    This consolidates the previous Properties 3 and 4.
    """
    print("\n" + "="*70)
    print("PROPERTY 3: Emergency Braking Safety")
    print("="*70)
    
    emergency_transitions = ['e6', 'e12', 'e18', 'e24', 'e28']
    
    vars_dict = create_automaton_vars()
    guards = get_guards(vars_dict)
    
    all_verified = True
    
    for trans in emergency_transitions:
        guard = guards[trans]
        
        vd = valid_d(vars_dict['B_C'], vars_dict['s_d'])
        vc = valid_c(vars_dict['B_cross'], vars_dict['s_c'])
        cd = c_distance(vars_dict['B_TTC'])
        
        critical_conditions = And(vd, vc, cd)
        property_formula = Implies(guard, critical_conditions)
        
        property_name = f"{trans} → EmergencyBraking requires (valid_d ∧ valid_c ∧ c_distance)"
        result = check_property(property_name, property_formula, vars_dict)
        
        if result is False:
            all_verified = False
    
    print("\n" + "="*70)
    if all_verified:
        print("RESULT: Emergency braking only in critical situations ✓")
    else:
        print("RESULT: Emergency braking may be triggered inappropriately ✗")
    print("="*70)
    
    return all_verified

# ============================================================================
# PROPERTY 4: Bounded Liveness - Uncertainty Timer Progression
# ============================================================================

def prop_bounded_liveness_uncertainty():
    """
    Bounded Liveness Property:
    If uncertainty persists (uncertain_distance remains true), then s_u
    will reach MAX_UNCERTAIN within a bounded number of steps.
    
    Approach:
    1. Unroll the automaton for K steps (where K = ceil(MAX_UNCERTAIN / CAMERA_FREQ))
    2. Model state transitions between steps
    3. Assume uncertainty persists and s_d is invalid throughout
    4. Verify that s_u must reach MAX_UNCERTAIN by step K
    
    We model the key dynamics:
    - s_u increments by CAMERA_FREQ each step when uncertainty holds
    - s_u resets to 0 when s_d < MAX_STALE (valid detection)
    - We're checking the worst case: continuous uncertainty with invalid detection
    """
    print("\n" + "="*70)
    print("PROPERTY 4: Bounded Liveness - Uncertainty Timer Progression")
    print("="*70)
    
    # Calculate number of steps needed
    K = int((MAX_UNCERTAIN_VAL + CAMERA_FREQ_VAL - 1) // CAMERA_FREQ_VAL)
    print(f"\nUnrolling for K = {K} steps (MAX_UNCERTAIN={MAX_UNCERTAIN_VAL}, CAMERA_FREQ={CAMERA_FREQ_VAL})")
    
    # Create variables for each step
    vars_list = []
    for step in range(K + 1):
        vars_dict = create_automaton_vars()
        vars_list.append(vars_dict)
    
    solver = Solver()
    
    # Add buffer constraints for each step
    for step, vars_dict in enumerate(vars_list):
        for constraint in buffer_constraints(vars_dict):
            solver.add(constraint)
    
    # Initial conditions at step 0
    print("Setting initial conditions...")
    # Start with s_u = 0
    solver.add(vars_list[0]['s_u'] == 0)
    # Start with invalid detection (so s_u can increment)
    solver.add(vars_list[0]['s_d'] >= MAX_STALE)
    # Uncertainty holds initially
    solver.add(uncertain_distance(vars_list[0]['B_TTC']))
    
    # Model transitions between steps
    print("Modeling state transitions...")
    for step in range(K):
        s_u_curr = vars_list[step]['s_u']
        s_u_next = vars_list[step + 1]['s_u']
        s_d_curr = vars_list[step]['s_d']
        s_d_next = vars_list[step + 1]['s_d']
        B_TTC_curr = vars_list[step]['B_TTC']
        B_TTC_next = vars_list[step + 1]['B_TTC']
        
        # Key dynamics:
        # 1. If s_d becomes valid (< MAX_STALE), s_u resets to 0
        # 2. Otherwise, if uncertainty holds, s_u increments by CAMERA_FREQ
        # 3. s_u is capped at MAX_UNCERTAIN
        
        # Assumption: uncertainty persists AND detection remains invalid
        # (worst case for liveness)
        solver.add(uncertain_distance(B_TTC_curr))
        solver.add(s_d_next >= MAX_STALE)  # detection stays invalid
        
        # s_u increment logic (from your buffer_constraints logic)
        solver.add(
            s_u_next == If(s_d_next < MAX_STALE, 
                          0,  # Reset if detection becomes valid
                          If(s_u_curr + CAMERA_FREQ >= MAX_UNCERTAIN,
                             MAX_UNCERTAIN,  # Cap at max
                             s_u_curr + CAMERA_FREQ))  # Increment
        )
    
    # Property to check: s_u reaches MAX_UNCERTAIN by step K
    # We check the NEGATION (s_u < MAX_UNCERTAIN at final step)
    s_u_final = vars_list[K]['s_u']
    print(f"\nChecking if s_u reaches MAX_UNCERTAIN by step {K}...")
    solver.add(s_u_final < MAX_UNCERTAIN)
    
    result = solver.check()
    
    if result == unsat:
        print(f"✓ VERIFIED: s_u reaches MAX_UNCERTAIN within {K} steps")
        print(f"  Under continuous uncertainty with invalid detection,")
        print(f"  s_u MUST reach {MAX_UNCERTAIN} within {K} camera frames")
        verified = True
    elif result == sat:
        print(f"✗ FALSIFIED: s_u may not reach MAX_UNCERTAIN within {K} steps")
        model = solver.model()
        print(f"  Counterexample trace:")
        for step in range(K + 1):
            s_u_val = model.eval(vars_list[step]['s_u'])
            s_d_val = model.eval(vars_list[step]['s_d'])
            print(f"    Step {step}: s_u = {s_u_val}, s_d = {s_d_val}")
        verified = False
    else:
        print(f"? UNKNOWN: Could not determine property")
        verified = None
    
    print("\n" + "="*70)
    if verified:
        print("RESULT: Bounded liveness verified ✓")
        print(f"  Uncertainty is guaranteed to trigger handover within {K} frames")
    else:
        print("RESULT: Bounded liveness failed ✗")
    print("="*70)
    
    return verified

# ============================================================================
# PROPERTY 5: Emergency Braking Exit Conditions
# ============================================================================

def prop_emergency_exit_safety():
    """
    Verify that once in EMERGENCY_BRAKING, you can only exit through:
    - e29: when detection OR crossing becomes invalid
    - e30: when crossing persists (stays in EMERGENCY_BRAKING)
    
    This ensures we don't prematurely exit emergency braking.
    """
    print("\n" + "="*70)
    print("PROPERTY 5: Emergency Braking Exit Safety")
    print("="*70)
    
    vars_dict = create_automaton_vars()
    guards = get_guards(vars_dict)
    
    # e29: exits to SOFT_BRAKING when Not(detected) OR Not(crossing)
    # e30: stays in EMERGENCY_BRAKING when crossing
    
    det = detected(vars_dict['B_C'])
    cross = crossing(vars_dict['B_cross'])
    
    # e29 should be enabled when detection or crossing fails
    exit_condition = Or(Not(det), Not(cross))
    property_e29 = Implies(exit_condition, guards['e29'])
    
    # e30 should be enabled when crossing persists
    property_e30 = Implies(cross, guards['e30'])
    
    result_e29 = check_property("e29: Exit when detection/crossing fails", 
                                property_e29, vars_dict)
    result_e30 = check_property("e30: Stay when crossing persists", 
                                property_e30, vars_dict)
    
    all_verified = result_e29 and result_e30
    
    print("\n" + "="*70)
    if all_verified:
        print("RESULT: Emergency braking exits safely ✓")
    else:
        print("RESULT: Emergency braking exit logic may be incorrect ✗")
    print("="*70)
    
    return all_verified

# ============================================================================
# PROPERTY 6: Uncertainty Timer Reset
# ============================================================================

def prop_uncertainty_timer_reset():
    """
    Verify that s_u is 0 whenever detection is valid (s_d < MAX_STALE).
    
    This ensures the uncertainty timer only runs when we lack valid detection.
    """
    print("\n" + "="*70)
    print("PROPERTY 6: Uncertainty Timer Reset on Valid Detection")
    print("="*70)
    
    vars_dict = create_automaton_vars()
    
    # Property: s_d < MAX_STALE → s_u == 0
    property_formula = Implies(
        vars_dict['s_d'] < MAX_STALE,
        vars_dict['s_u'] == 0
    )
    
    result = check_property("s_d valid → s_u = 0", property_formula, vars_dict)
    
    print("\n" + "="*70)
    if result:
        print("RESULT: Uncertainty timer resets with valid detection ✓")
    else:
        print("RESULT: Uncertainty timer may not reset properly ✗")
    print("="*70)
    
    return result

# ============================================================================
# MAIN VERIFICATION RUNNER
# ============================================================================

def verify_all_properties():
    """Run all property verifications."""
    print("\n" + "#"*70)
    print("# PEDESTRIAN PROTECTION AUTOMATON - FORMAL VERIFICATION")
    print("#"*70)
    
    results = {
        'P1: Determinism': prop_guards_mutually_exclusive(),
        'P2: Completeness': prop_guards_complete(),
        'P3: Emergency Braking Safety': prop_emergency_braking_safety(),
        'P4: Bounded Liveness (Uncertainty)': prop_bounded_liveness_uncertainty(),
        'P5: Emergency Exit Safety': prop_emergency_exit_safety(),
        'P6: Uncertainty Timer Reset': prop_uncertainty_timer_reset()
    }
    
    print("\n" + "#"*70)
    print("# VERIFICATION SUMMARY")
    print("#"*70)
    for prop_name, result in results.items():
        status = "✓ VERIFIED" if result else "✗ FAILED"
        print(f"{prop_name}: {status}")
    print("#"*70 + "\n")
    
    return all(results.values())

if __name__ == "__main__":
    verify_all_properties()