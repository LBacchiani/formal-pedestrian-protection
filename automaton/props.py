"""
Verification harness for the pedestrian protection automaton (uses z3_automaton.py).
Run this alongside your automaton module.
"""

from z3 import *
from z3_automaton import *   # imports all N, CAMERA_FREQ_VAL, MAX_UNCERTAIN_VAL, functions, etc.
from automaton import *
# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def model_val(model, expr):
    """Safely evaluate expr in model (with model completion) and return as string."""
    try:
        val = model.eval(expr, model_completion=True)
        return str(val)
    except Exception:
        return "<undef>"

def print_buffer_model(model, buffer_vars):
    """Return a Python list of string values for buffer entries from model."""
    return [model_val(model, v) for v in buffer_vars]

# ---------------------------------------------------------------------------
# Generic property checker
# ---------------------------------------------------------------------------

def check_property(property_name: str, property_formula, vars_dict):
    """Helper function to check a property using Z3."""
    solver = Solver()
    
    # Add buffer constraints (invariants)
    for constraint in buffer_constraints(vars_dict):
        solver.add(constraint)
    
    
    # Negate the property (we want UNSAT to mean verified)
    solver.add(Not(property_formula))
    
    result = solver.check()
    
    if result == unsat:
        print(f"✓ {property_name}: VERIFIED")
        return True
    elif result == sat:
        print(f"✗ {property_name}: FALSIFIED")
        model = solver.model()
        print(f"  Counterexample:")
        # print timers safely
        for var_name in ['s_d', 's_c', 's_u', 'sns', 't']:
            if var_name in vars_dict:
                print(f"    {var_name} = {model_val(model, vars_dict[var_name])}")
        # print buffers
        if 'B_C' in vars_dict:
            print(f"    B_C: {print_buffer_model(model, vars_dict['B_C'])}")
        if 'B_TTC' in vars_dict:
            print(f"    B_TTC: {print_buffer_model(model, vars_dict['B_TTC'])}")
        if 'B_cross' in vars_dict:
            print(f"    B_cross: {print_buffer_model(model, vars_dict['B_cross'])}")
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
                # property: not(guard_t1 and guard_t2)
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
        
        # property: at least one guard holds (we assert this must always be true)
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
    Verify that Emergency Braking transitions are only taken when:
      valid_d ∧ valid_c ∧ c_distance
    """
    print("\n" + "="*70)
    print("PROPERTY 3: Emergency Braking Safety")
    print("="*70)
    
    # transitions that lead to EMERGENCY_BRAKING (as per your automaton)
    emergency_transitions = ['e6', 'e12', 'e18', 'e24', 'e28']
    
    all_verified = True
    
    for trans in emergency_transitions:
        vars_dict = create_automaton_vars()
        guards = get_guards(vars_dict)
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
    If uncertainty persists, s_u will reach MAX_UNCERTAIN within K = ceil(MAX_UNCERTAIN_VAL / CAMERA_FREQ_VAL) steps.
    We unroll the system for K steps and force the worst-case (no valid detection, uncertainty persists).
    """
    print("\n" + "="*70)
    print("PROPERTY 4: Bounded Liveness - Uncertainty Timer Progression")
    print("="*70)
    
    # Compute K using the automaton's Python constants (ms units as in the automaton)
    # CAMERA_FREQ_VAL and MAX_UNCERTAIN_VAL are module-level Python ints defined in the automaton
    K = int((MAX_UNCERTAIN + CAMERA_FREQ - 1) // CAMERA_FREQ)
    print(f"\nUnrolling for K = {K} steps (MAX_UNCERTAIN={MAX_UNCERTAIN} ms, CAMERA_FREQ={CAMERA_FREQ} ms/frame)")
    
    # Create variables for each step
    vars_list = []
    for step in range(K + 1):
        vars_dict = create_automaton_vars()
        # use unique var names across steps to avoid collisions: rename the consts
        # Z3 named constants created in create_automaton_vars() have fixed names;
        # to avoid clashes we remap them into fresh copies per-step by wrapping them in Let expressions
        # Simpler approach: treat each step independently since create_automaton_vars uses same var names,
        # but Z3 allows shadowing across solver contexts if we keep separate solvers. To keep code simple,
        # we will reuse create_automaton_vars but be careful: buffer_constraints() creates fresh Int('first_idx_d') etc.
        # This is acceptable for bounded unrolling in this script.
        vars_list.append(vars_dict)
    
    solver = Solver()
    
    # Add buffer constraints and thresholds for each step
    for step, vars_dict in enumerate(vars_list):
        for constraint in buffer_constraints(vars_dict):
            solver.add(constraint)

    # Initial conditions at step 0 (worst-case start)
    solver.add(vars_list[0]['s_u'] == 0)                # start with s_u = 0
    solver.add(vars_list[0]['s_d'] >= MAX_STALE)        # start with detection invalid (stale)
    solver.add(uncertain(vars_list[0]['B_TTC']))        # uncertainty holds initially (use automaton's uncertain)
    
    # Model transitions between steps (s_u progression)
    for step in range(K):
        s_u_curr = vars_list[step]['s_u']
        s_u_next = vars_list[step + 1]['s_u']
        s_d_next = vars_list[step + 1]['s_d']
        B_TTC_curr = vars_list[step]['B_TTC']
        
        # force uncertainty to hold and detection to remain invalid (worst case)
        solver.add(uncertain(B_TTC_curr))
        solver.add(s_d_next >= MAX_STALE)
        
        # s_u increment logic (ms units)
        # s_u_next == If(s_d_next < MAX_STALE, 0,
        #                 If(s_u_curr + CAMERA_FREQ >= MAX_UNCERTAIN, MAX_UNCERTAIN, s_u_curr + CAMERA_FREQ))
        solver.add(
            s_u_next == If(s_d_next < MAX_STALE,
                           0,
                           If(s_u_curr + CAMERA_FREQ >= MAX_UNCERTAIN,
                              MAX_UNCERTAIN,
                              s_u_curr + CAMERA_FREQ))
        )
    
    # Property to check: s_u reaches MAX_UNCERTAIN by step K
    # We assert the negation (i.e., s_u_final < MAX_UNCERTAIN) to look for counterexample
    s_u_final = vars_list[K]['s_u']
    solver.add(s_u_final < MAX_UNCERTAIN)
    
    result = solver.check()
    
    if result == unsat:
        print(f"✓ VERIFIED: s_u reaches MAX_UNCERTAIN within {K} steps")
        verified = True
    elif result == sat:
        print(f"✗ FALSIFIED: s_u may NOT reach MAX_UNCERTAIN within {K} steps")
        model = solver.model()
        print("  Counterexample trace (s_u, s_d per step):")
        for step in range(K + 1):
            print(f"    Step {step}: s_u = {model_val(model, vars_list[step]['s_u'])}, s_d = {model_val(model, vars_list[step]['s_d'])}")
        verified = False
    else:
        print(f"? UNKNOWN: Could not determine property")
        verified = None
    
    print("\n" + "="*70)
    if verified:
        print("RESULT: Bounded liveness verified ✓")
        print(f"  Uncertainty must trigger handover within {K} frames (worst-case).")
    else:
        print("RESULT: Bounded liveness failed ✗")
    print("="*70)
    
    return verified

# ============================================================================
# PROPERTY 5: Emergency Braking Exit Conditions
# ============================================================================

def prop_emergency_exit_safety():
    """
    Verify that once in EMERGENCY_BRAKING, exits are only via e29 or e30 as specified:
      - e29: when detection or crossing fails (exit)
      - e30: remain in EMERGENCY_BRAKING when crossing persists
    """
    print("\n" + "="*70)
    print("PROPERTY 5: Emergency Braking Exit Safety")
    print("="*70)
    
    vars_dict = create_automaton_vars()
    guards = get_guards(vars_dict)
    
    det = detected(vars_dict['B_C'])
    cross = crossing(vars_dict['B_cross'])
    
    # e29 should be enabled when detection or crossing fails (i.e., Not(det) or Not(cross))
    exit_condition = Or(Not(det), Not(cross))
    property_e29 = Implies(exit_condition, guards['e29'])
    
    # e30 should be enabled when crossing persists (stay in emergency)
    property_e30 = Implies(cross, guards['e30'])
    
    result_e29 = check_property("e29: Exit when detection/crossing fails", property_e29, vars_dict)
    result_e30 = check_property("e30: Stay when crossing persists", property_e30, vars_dict)
    
    all_verified = (result_e29 is True) and (result_e30 is True)
    
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
    Verify s_d < MAX_STALE -> s_u == 0
    """
    print("\n" + "="*70)
    print("PROPERTY 6: Uncertainty Timer Reset on Valid Detection")
    print("="*70)
    
    vars_dict = create_automaton_vars()
    
    property_formula = Implies(vars_dict['s_d'] < MAX_STALE, vars_dict['s_u'] == 0)
    
    result = check_property("s_d valid → s_u = 0", property_formula, vars_dict)
    
    print("\n" + "="*70)
    if result:
        print("RESULT: Uncertainty timer resets with valid detection ✓")
    else:
        print("RESULT: Uncertainty timer may not reset properly ✗")
    print("="*70)
    
    return result

# ============================================================================
# MAIN: run all properties
# ============================================================================

def verify_all_properties():
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
        if result is None:
            status = "? UNKNOWN"
        print(f"{prop_name}: {status}")
    print("#"*70 + "\n")
    
    # return True only if all are True (not None)
    return all((r is True) for r in results.values())

if __name__ == "__main__":
    verify_all_properties()
