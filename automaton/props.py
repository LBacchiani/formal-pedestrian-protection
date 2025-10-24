from z3 import *
from z3_automaton import * # Assuming this module defines constants, State, and helper functions

# ---------------------------------------------------------------------------
# Small helpers for nicer output
# ---------------------------------------------------------------------------

def model_val(model, expr):
    """Safely evaluate expr in model (with model completion) and return as string."""
    try:
        # Use simple string conversion for common types for cleaner output
        return str(model.eval(expr, model_completion=True)) 
    except Exception:
        return "<undef>"

def print_buffer_model(model, buffer_vars):
    """Return a Python list of string values for buffer entries from model."""
    return [model_val(model, v) for v in buffer_vars]

def print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs):
    """Utility function to print common continuous state variables"""
    print(f"    s_d = {model_val(m, s_d)}, s_c = {model_val(m, s_c)}, s_u = {model_val(m, s_u)}, t = {model_val(m, t)}")
    print(f"    B_C: {print_buffer_model(m, B_C)}")
    print(f"    B_TTC: {print_buffer_model(m, B_TTC)}")
    print(f"    B_cs: {print_buffer_model(m, B_cs)}")

# ---------------------------------------------------------------------------
# Determinism property check
# ---------------------------------------------------------------------------

def prop_guards_mutually_exclusive():
    """
    PROPERTY: Determinism (Guards Mutually Exclusive)
    Verify that for each discrete state, at most one transition can fire.
    """
    print("\n" + "="*70)
    print("PROPERTY: Determinism (Guards Mutually Exclusive)")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)
    states = [Normal, SafeWarning, Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, s_u, t) for qn in states ]

        for i in range(len(trans_constraints)):
            for j in range(i + 1, len(trans_constraints)):
                s.push()
                s.add(q == q_val)
                s.add(trans_constraints[i])
                s.add(trans_constraints[j])
                s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))

                if s.check() == sat:
                    m = s.model()
                    all_verified = False
                    print(f"\n✗ Non-determinism detected for q = {q_val}")
                    print(f"  Conflicting transitions to: {states[i]} and {states[j]}")
                    print("  Counterexample full state:")
                    print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs)
                s.pop()

    print(f"\n{'✓' if all_verified else '✗'} All guards are mutually exclusive: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Completeness property check
# ---------------------------------------------------------------------------

def prop_guards_complete():
    """
    PROPERTY: Completeness (No Deadlock)
    Verify: For each discrete state q, if no transition is fireable, then the invariant must hold.
    """
    print("\n" + "="*70)
    print("PROPERTY: Completeness (No Deadlock)")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)
    states = [Normal, SafeWarning, Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, s_u, t) for qn in states ]
    
        s.push()
        s.add(q == q_val)
        s.add(Not(Or(trans_constraints)))      # No transition is fireable
        s.add(Not(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t)))  # Invariant fails (VIOLATION)

        if s.check() == sat:
            m = s.model()
            all_verified = False
            print(f"\n✗ Completeness violation detected for q = {q_val}")
            print("  No fireable transition AND invariant does NOT hold (Deadlock State Found)!")
            print("  Counterexample full state:")
            print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs)
        s.pop()

    print(f"\n{'✓' if all_verified else '✗'} Completeness {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Safety property check (Distance)
# ---------------------------------------------------------------------------

def prop_distance_safety():
    """
    Safety: The car's braking state must match the closest distance threshold (criticality).
    Models the two-step transition: X -> X_next (sense) -> X_reset (transition/reset).
    Guard check is on X_next, invariant check on X_reset.
    """
    print("\n" + "="*70)
    print("PROPERTY: Distance Safety (Minimum Braking Response)")
    print("="*70)

    s = Solver()
    all_verified = True

    # State variables (pre-sense, post-sense, post-reset)
    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars("_next")
    q_next = Const('q_next', State)
    B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset = declare_continuous_vars("_reset")
    
    # Define generic sensor values for sense() for simplicity
    C_new_generic = RealVal(1)
    TTC_new_risky = RealVal(TH_TTC_R)
    cs_new_crossing = IntVal(1)

    def check_safety_violation(solver, condition_trigger, unsafe_state, prop_name, c_new, ttc_new, cs_new):
        """Helper to check one safety level violation"""
        nonlocal all_verified
        solver.push()
        
        # 1. Sense Jump (X -> X_next)
        solver.add(t >= CAMERA_FREQ) # Ensures sense fires
        solver.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, C_new=c_new, TTC_new=ttc_new, cs_new=cs_new))
        
        # 2. Trigger Condition: Distance requirement is met on the fresh data (X_next)
        solver.add(condition_trigger)
        
        # 3. Transition Check (q -> q_next)
        solver.add(transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
        
        # 4. Timer Reset (X_next -> X_reset)
        solver.add(reset_timers(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        
        # 5. Violation: System transitions to an unsafe discrete state q_next
        solver.add(Not(q_next == unsafe_state))
        
        # 6. Consistency: The resulting X_reset must satisfy the invariant of the violating q_next
        solver.add(invariant(q_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        
        if solver.check() == sat:
            all_verified = False
            m = solver.model()
            print(f"\n✗ Distance safety violated: {prop_name}")
            print(f"  q_pre = {model_val(m, q)}, q_next = {model_val(m, q_next)}")
            print("  Pre-sense state:")
            print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs)
            print("  Post-sense (Guard) state:")
            print_ce_vars(m, s_d_next, s_c_next, s_u_next, t_next, B_C_next, B_TTC_next, B_cs_next)
        solver.pop()

    # --- Critical distance → EmergencyBraking ---
    check_safety_violation(s, 
        And(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            c_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)),
        EmergencyBraking,
        "Critical distance requires EmergencyBraking",
        RealVal(1), RealVal(0.1), IntVal(1) # New sensor data is critical
    )

    # --- Risky distance → SoftBraking ---
    check_safety_violation(s, 
        And(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            r_c_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)), 
        SoftBraking,
        "Risky distance requires SoftBraking",
        RealVal(1), TTC_new_risky, IntVal(1) # New sensor data is risky
    )
    
    # --- Short distance → Throttling ---
    check_safety_violation(s, 
        And(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
            s_r_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)),
        Throttling,
        "Short distance requires Throttling",
        RealVal(1), RealVal(TH_TTC_S), IntVal(1) # New sensor data is short
    )

    print(f"\n{'✓' if all_verified else '✗'} Distance Safety {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Safety property check (Staleness)
# ---------------------------------------------------------------------------

def prop_safe_transition_on_staleness():
    """
    Safety: If in a braking state, losing 'valid_d' or 'valid_c' must lead to a regression
    (Normal or SafeWarning).
    """
    print("\n" + "="*70)
    print("PROPERTY: Safe Transition on Data Staleness")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars("_next")
    q = Const('q', State)
    q_next = Const('q_next', State)

    for current_state in [Throttling, SoftBraking, EmergencyBraking]:
        s.push()
        s.add(q == current_state)
        s.add(t >= CAMERA_FREQ)
        s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
        
        # Sense: Sensor sees NO pedestrian (C_new=0), causing timers to advance/reset
        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, C_new=RealVal(0), TTC_new=RealVal(0), cs_new=IntVal(0)))
        
        # Condition for Staleness: Data invalid after sensor shift
        if current_state == EmergencyBraking:
            # EB depends only on valid_d. Staleness of s_d must lead to Normal.
            s.add(s_d_next > TH_D_STALE)
            s.add(transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
            s.add(q_next != Normal)
        else:
            # Throttling and SoftBraking depend on valid_d AND valid_c. Losing either should regress.
            s.add(Or(s_d_next > TH_D_STALE, s_c_next > TH_C_STALE))

            s.add(transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
            
            # Violation: q_next is still in a braking state (Throttling, Soft, Emergency)
            s.add(Or(q_next == Throttling, q_next == SoftBraking, q_next == EmergencyBraking))

        if s.check() == sat:
            all_verified = False
            m = s.model()
            print(f"\n✗ Staleness violation from q = {current_state}")
            print(f"  q_next = {model_val(m, q_next)}")
            print(f"  s_d_next = {model_val(m, s_d_next)}, s_c_next = {model_val(m, s_c_next)}")
            print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs)
        s.pop()

    print(f"\n{'✓' if all_verified else '✗'} Safe Transition on Data Staleness {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Liveness property check (Sudden Pedestrian)
# ---------------------------------------------------------------------------

def prop_sudden_pedestrian_reaction():
    """
    PROPERTY: Sudden pedestrian appears → EmergencyBraking within k steps (Liveness)
    Checks if, starting from an arbitrary state, a critical threat always leads to EmergencyBraking.
    The transition logic must use X_next for guards and X_reset for the next step.
    """
    print("\n" + "="*70)
    print("PROPERTY: Sudden Pedestrian Reaction (Liveness)")
    print("="*70)

    s = Solver()
    all_verified = True

    # Initial variables (q is unbounded)
    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)
    s.add(initial_state(q,B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    

    max_steps = RT_HALF_FRAMES # The maximum allowed steps
    
    # Critical input that persists across all steps
    C_new_critical = RealVal(1) 
    TTC_new_critical = RealVal(TH_TTC_C / 2) # Make it very critical
    cs_new_crossing = IntVal(1) 

    # Unroll transitions
    current_B_C, current_B_TTC, current_B_cs = B_C, B_TTC, B_cs
    current_s_d, current_s_c, current_s_u, current_t = s_d, s_c, s_u, t
    current_q = q
    
    s.add(invariant(current_q, current_B_C, current_B_TTC, current_B_cs, current_s_d, current_s_c, current_s_u, current_t))
    s.add(current_t >= CAMERA_FREQ) # First step must be a sense

    for step in range(max_steps):
        B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars(f"_next{step}")
        B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset = declare_continuous_vars(f"_reset{step}")
        q_next = Const(f'q_next{step}', State)

        # 1. Sense Jump (X -> X_next): Applies critical input
        s.add(sense(current_B_C, current_B_TTC, current_B_cs, current_s_d, current_s_c, current_s_u, current_t,
                    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next,
                    C_new_critical, TTC_new_critical, cs_new_crossing))
        
        # 2. Transition (q -> q_next) - GUARD CHECK ON X_NEXT
        s.add(transition(current_q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))

        # 3. Timer Reset (X_next -> X_reset)
        s.add(reset_timers(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        
        # 4. Invariant must hold on the next state (q_next, X_reset)
        s.add(invariant(q_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))

        # Update current step for next iteration (q, X) -> (q_next, X_reset)
        current_B_C, current_B_TTC, current_B_cs = B_C_reset, B_TTC_reset, B_cs_reset
        current_s_d, current_s_c, current_s_u, current_t = s_d_reset, s_c_reset, s_u_reset, t_reset
        current_q = q_next

    # Liveness Violation: By the end, state must NOT be EmergencyBraking
    s.add(current_q != EmergencyBraking)

    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("\n✗ Sudden pedestrian safety violation: did NOT reach EmergencyBraking in time")
        print(f"  q_final = {model_val(m, current_q)}")
        print(f"  q_start = {model_val(m, q)}")
        # print state trace for context
        for step in range(max_steps):
            print(f"    Step {step+1}: q = {model_val(m, Const(f'q_next{step}', State))}, t_next={model_val(m, Const(f't_next{step}', Real))}")
        print(f"  Last buffer B_TTC: {print_buffer_model(m, current_B_TTC)}")
    else:
        print("\n✓ Sudden pedestrian → EmergencyBraking within required steps: VERIFIED")

    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Safety property check (Monotonicity)
# ---------------------------------------------------------------------------

def prop_monotonic_safety_response():
    """
    PROPERTY: Monotonicity of Safety Response (No Unsafe Degradation)
    
    If we transition from an aggressive state (q_from) to a less aggressive state (q_to)
    while fresh sensor data (X_next) indicates a continuing threat, a violation occurs.
    The guard check is on X_next.
    """
    print("\n" + "="*70)
    print("PROPERTY: Monotonicity of Safety Response (No Unsafe Degradation)")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars("_next")
    B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset = declare_continuous_vars("_reset")
    q = Const('q', State)
    q_next = Const('q_next', State)

    # Test degradation scenarios (q_from is more aggressive than q_to)
    degradation_pairs = [
        (EmergencyBraking, SoftBraking), (EmergencyBraking, Throttling), (EmergencyBraking, SafeWarning),
        (SoftBraking, Throttling), (SoftBraking, SafeWarning), (SoftBraking, Normal),
        (Throttling, SafeWarning), (Throttling, Normal),
        (SafeWarning, Normal)
    ]

    for q_from, q_to in degradation_pairs:
        s.push()
        
        # Initial state constraints
        s.add(q == q_from)
        s.add(t >= CAMERA_FREQ) # Must be a sense event
        s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
        
        # Sensor input variables
        C_new = Real('C_new')
        TTC_new = Real('TTC_new')
        cs_new = Int('cs_new')
        
        # 1. Sense Jump (X -> X_next)
        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t,
                    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next,
                    C_new, TTC_new, cs_new))
        
        # 2. Transition (q -> q_next) - GUARD CHECK ON X_NEXT
        s.add(transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
        s.add(q_next == q_to) # VIOLATION: Transitioned to a less aggressive state
        
        # 3. Timer Reset (X_next -> X_reset)
        s.add(reset_timers(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next,
                          B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        
        # 4. Consistency: Invariant must hold on the resulting state
        s.add(invariant(q_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        
        # 5. Trigger: Fresh data (C_new, TTC_new, cs_new) still shows threat matching or exceeding q_from's conditions
        s.add(C_new >= TH_C)  # High confidence detection
        s.add(cs_new == 1)     # Pedestrian is crossing
        
        # Distance should still be dangerous for the original state
        if q_from == EmergencyBraking:
            s.add(TTC_new <= TH_TTC_C)
        elif q_from == SoftBraking:
            s.add(TTC_new <= TH_TTC_R)
        elif q_from == Throttling:
            s.add(TTC_new <= TH_TTC_S)
        # Note: SafeWarning requires TTC <= TH_TTC_F. This is implicitly handled if the guard allows the regression.

        if s.check() == sat:
            all_verified = False
            m = s.model()
            print(f"\n✗ Monotonicity violation: {model_val(m, q)} → {model_val(m, q_next)}")
            print(f"  Fresh sensor data shows continuing threat: C_new = {model_val(m, C_new)}, TTC_new = {model_val(m, TTC_new)}")
            print("  Pre-sense state:")
            print_ce_vars(m, s_d, s_c, s_u, t, B_C, B_TTC, B_cs)
        
        s.pop()

    print(f"\n{'✓' if all_verified else '✗'} Monotonicity of Safety Response {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prop_guards_mutually_exclusive()
    prop_guards_complete()
    prop_distance_safety()
    prop_safe_transition_on_staleness()
    prop_sudden_pedestrian_reaction()