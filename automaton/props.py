from z3 import *
from z3_automaton import *

# ---------------------------------------------------------------------------
# Small helpers for nicer output
# ---------------------------------------------------------------------------

def model_val(model, expr):
    """Safely evaluate expr in model (with model completion) and return as string."""
    try:
        return str(model.eval(expr, model_completion=True))
    except Exception:
        return "<undef>"

def print_buffer_model(model, buffer_vars):
    """Return a Python list of string values for buffer entries from model."""
    return [model_val(model, v) for v in buffer_vars]

# def buffer_invariants(B_C, B_TTC, B_cs, s_d, s_c, s_u, t):
#     inv = []
#     N = len(B_C)
#     for i in range(N):
#         inv.append(And(B_C[i] >= 0, B_C[i] <= 1))
#         inv.append(Or(B_TTC[i] == NO_TTC, B_TTC[i] > 0))
#         inv.append(Or(B_cs[i] == 0, B_cs[i] == 1))
#         inv.append(Implies(B_C[i] < TH_C, And(B_cs[i] == 0,B_TTC[i] == NO_TTC)))
#     inv.append(And(s_d >= 0, s_d <= TH_D_STALE, s_c >= 0, s_c <= TH_C_STALE, s_u >= 0, s_u <= MAX_UNCERTAIN, t >= 0, t <= CAMERA_FREQ))
#     inv.append(Implies(detected(B_C, B_TTC, B_cs, s_d, s_c, s_u, t), s_d == 0))
#     inv.append(Implies(crossing(B_C, B_TTC, B_cs, s_d, s_c, s_u, t), s_c == 0))
#     inv.append(Implies(uncertain(B_C, B_TTC, B_cs, s_d, s_c, s_u, t), s_u == 0))
#     return And(inv)

# ---------------------------------------------------------------------------
# Determinism property check
# ---------------------------------------------------------------------------

def prop_guards_mutually_exclusive():
    """
    Verify that for each discrete state, at most one transition can fire
    at any time (mutually exclusive guards).
    """
    print("\n" + "="*70)
    print("PROPERTY: Determinism (Guards Mutually Exclusive)")
    print("="*70)

    s = Solver()
    all_verified = True

    # Declare automaton variables
    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)

    states = [Normal, SafeWarning, Throttling, SoftBraking, EmergencyBraking]

    # Check all pairs of transitions for each state
    for q_val in states:
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, s_u, t) for qn in states ]

        for i in range(len(trans_constraints)):
            for j in range(i+1, len(trans_constraints)):
                s.push()
                s.add(q == q_val)
                s.add(trans_constraints[i])
                s.add(trans_constraints[j])
                s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))

                if s.check() == sat:
                    m = s.model()
                    all_verified = False
                    print(f"\n✗ Non-determinism detected for q = {q_val}")
                    print(f"  Conflicting transitions: {i} and {j}")
                    print("  Counterexample full state:")
                    print(f"    s_d = {model_val(m, s_d)}, s_c = {model_val(m, s_c)}, s_u = {model_val(m, s_u)}, t = {model_val(m, t)}")
                    print(f"    B_C: {print_buffer_model(m, B_C)}")
                    print(f"    B_TTC: {print_buffer_model(m, B_TTC)}")
                    print(f"    B_cs: {print_buffer_model(m, B_cs)}")
                s.pop()

    if all_verified:
        print("\n✓ All guards are mutually exclusive: VERIFIED")
    else:
        print("\n✗ Some guards are not mutually exclusive: NON-DETERMINISTIC")

    print("="*70)
    return all_verified

# ---------------------------------------------------------------------------
# Completeness property check
# ---------------------------------------------------------------------------

def prop_guards_complete():
    """
    PROPERTY: Completeness (No Deadlock)
    Meaning: For each discrete state q,
             if no transition is fireable, then the invariant must hold.
    """
    print("\n" + "="*70)
    print("PROPERTY: Completeness (No Deadlock / Invariant implies no transitions)")
    print("="*70)

    s = Solver()
    all_verified = True

    # Declare automaton variables
    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)

    states = [Normal, SafeWarning, Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        # Collect all transitions from this state
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, s_u, t) for qn in states ]
    

        # Counterexample: no transition is fireable AND invariant does NOT hold
        s.push()
        s.add(q == q_val)
        s.add(Not(Or(trans_constraints)))  # no transition is fireable
        s.add(Not(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t)))  # invariant fails

        if s.check() == sat:
            m = s.model()
            all_verified = False
            print(f"\n✗ Completeness violation detected for q = {q_val}")
            print("  No fireable transition AND invariant does NOT hold!")
            print(f"    s_d = {m.eval(s_d, model_completion=True)}, "
                  f"s_c = {m.eval(s_c, model_completion=True)}, "
                  f"s_u = {m.eval(s_u, model_completion=True)}, "
                  f"t = {m.eval(t, model_completion=True)}")
            print(f"    B_C: {[m.eval(v, model_completion=True) for v in B_C]}")
            print(f"    B_TTC: {[m.eval(v, model_completion=True) for v in B_TTC]}")
            print(f"    B_cs: {[m.eval(v, model_completion=True) for v in B_cs]}")
        s.pop()

    if all_verified:
        print("\n✓ Completeness VERIFIED: whenever no transition is fireable, the invariant holds")
    else:
        print("\n✗ Completeness FAILED: possible violation found")

    print("="*70)
    return all_verified


def prop_distance_safety():
    """
    Safety: The car's braking state must match the closest distance threshold
    """
    print("\n" + "="*70)
    print("PROPERTY: Distance Safety")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)

    # Critical distance → EmergencyBraking
    s.push()
    s.add(c_dist(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(q != EmergencyBraking)
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(valid_d(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(valid_c(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    if s.check() == sat:
        all_verified = False
        m = s.model()
        print("\n✗ Distance safety violated: Critical distance not in EmergencyBraking")
        print(f"  q = {model_val(m, q)}")
    s.pop()

    # Risky distance → at least SoftBraking
    s.push()
    s.add(valid_d(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(valid_c(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(Or(q == Normal, q == SafeWarning, q == Throttling))
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    if s.check() == sat:
        all_verified = False
        m = s.model()
        print("\n✗ Distance safety violated: Risky distance not in SoftBraking/Emergency")
        print(f"  q = {model_val(m, q)}")
    s.pop()

    # Short distance → at least Throttling
    s.push()
    s.add(valid_d(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(valid_c(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    s.add(Or(q == Normal, q == SafeWarning))
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
    if s.check() == sat:
        all_verified = False
        m = s.model()
        print("\n✗ Distance safety violated: Reduced safe distance not in Throttling+")
        print(f"  q = {model_val(m, q)}")
    s.pop()

    if all_verified:
        print("\n✓ Distance Safety VERIFIED")
    else:
        print("\n✗ Distance Safety FAILED")
    print("="*70)
    return all_verified

def prop_braking_monotonicity():
    """
    Safety: Braking state cannot regress under hazard conditions
    """
    print("\n" + "="*70)
    print("PROPERTY: Braking Monotonicity")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars("_next")
    q = Const('q', State)
    q_next = Const('q_next', State)

    # Consider hazard condition (e.g., c_dist, r_c_dist, s_r_dist)
    s.push()
    s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, C_new=RealVal(0), TTC_new=RealVal(0), cs_new=IntVal(0)))
    s.add(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
    s.add(valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))
    s.add(Or(
        c_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
        r_c_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
        s_r_dist(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)
    ))
    # Unsafe if automaton regresses
    s.add(transition(q, q_next, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))    
    s.add(Or(
        And(q == EmergencyBraking, q_next != EmergencyBraking),
        And(q == SoftBraking, Or(q_next == Normal, q_next == SafeWarning, q_next == Throttling)),
        And(q == Throttling, Or(q_next == Normal, q_next == SafeWarning))    
    ))

    if s.check() == sat:
        all_verified = False
        m = s.model()
        print("\n✗ Braking regression violation")
        print(f"  q = {model_val(m, q)}, q_next = {model_val(m, q_next)}")
        print(f"  s_d = {model_val(m, s_d)}, s_c = {model_val(m, s_c)}, s_u = {model_val(m, s_u)}, t = {model_val(m, t)}")
        print(f"  B_C:  {print_buffer_model(m, B_C)}")
        print(f"  B_TTC: {print_buffer_model(m, B_TTC)}")
        print(f"  B_cs: {print_buffer_model(m, B_cs)}")
    s.pop()

    if all_verified:
        print("\n✓ Braking Monotonicity VERIFIED")
    else:
        print("\n✗ Braking Monotonicity FAILED")
    print("="*70)
    return all_verified

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
        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, C_new=RealVal(0), TTC_new=RealVal(0), cs_new=IntVal(0)))
        s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
        if current_state == EmergencyBraking:
            s.add(Not(valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)))
            s.add(And(
                transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
                q_next != Normal
            ))
        else:
            # Throttling and SoftBraking depend on valid_d AND valid_c
            s.add(Not(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)))
            # Check for a transition that *doesn't* lead to Normal or SafeWarning
            s.add(And(
                transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
                q_next != Normal,
                q_next != SafeWarning
            ))

        if s.check() == sat:
            all_verified = False
            m = s.model()
            print(f"\n✗ Staleness violation from q = {current_state}")
            print(f"  q_next = {model_val(m, q_next)}")
        s.pop()

    if all_verified:
        print("\n✓ Safe Transition on Data Staleness VERIFIED")
    else:
        print("\n✗ Safe Transition on Data Staleness FAILED")
    print("="*70)
    return all_verified


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
        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, C_new=RealVal(0), TTC_new=RealVal(0), cs_new=IntVal(0)))
        s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
        if current_state == EmergencyBraking:
            s.add(Not(valid_c(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)))
            s.add(And(
                transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
                q_next != Normal
            ))
        else:
            # Throttling and SoftBraking depend on valid_d AND valid_c
            s.add(Not(valid_d(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next)))
            # Check for a transition that *doesn't* lead to Normal or SafeWarning
            s.add(And(
                transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next),
                q_next != Normal,
                q_next != SafeWarning
            ))

        if s.check() == sat:
            all_verified = False
            m = s.model()
            print(f"\n✗ Staleness violation from q = {current_state}")
            print(f"  q_next = {model_val(m, q_next)}")
        s.pop()

    if all_verified:
        print("\n✓ Safe Transition on Data Staleness VERIFIED")
    else:
        print("\n✗ Safe Transition on Data Staleness FAILED")
    print("="*70)
    return all_verified

def prop_sudden_pedestrian_reaction():
    """
    PROPERTY: Sudden pedestrian appears → EmergencyBraking within k steps
    """
    print("\n" + "="*70)
    print("PROPERTY: Sudden Pedestrian Reaction")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, s_u, t = declare_continuous_vars()
    q = Const('q', State)

    # Step variables
    max_steps = RT_HALF_FRAMES
    buffers = [(B_C, B_TTC, B_cs, s_d, s_c, s_u, t)]
    states = [q]

    # Initial state = Normal
    #s.add(initial_state(q, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))

    for step in range(max_steps):
        B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next = declare_continuous_vars(f"_next{step}")
        B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset = declare_continuous_vars(f"_reset{step}")
        q_next = Const(f'q_next{step}', State)
        C_new = RealVal(1) 
        TTC_new = RealVal(TH_TTC_C) 
        cs_new = IntVal(1) 

        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, s_u, t,
                    B_C_next, B_TTC_next, B_cs_next,
                    s_d_next, s_c_next, s_u_next, t_next,
                    C_new, TTC_new, cs_new))
        s.add(transition(q, q_next, B_C, B_TTC, B_cs, s_d, s_c, s_u, t))
        s.add(invariant(q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next))

        s.add(reset_timers(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, s_u_next, t_next, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset))
        # Update current step
        B_C, B_TTC, B_cs, s_d, s_c, s_u, t = B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, s_u_reset, t_reset
        q = q_next

    # By the end, state must be EmergencyBraking
    s.add(q != EmergencyBraking)

    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("\n✗ Sudden pedestrian safety violation: did NOT reach EmergencyBraking in time")
        print(f"  q_final = {model_val(m, q)}")
        print(f"  Last buffer B_TTC: {print_buffer_model(m, B_TTC)}")
    else:
        print("\n✓ Sudden pedestrian → EmergencyBraking within required steps: VERIFIED")

    print("="*70)
    return all_verified



# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prop_guards_mutually_exclusive()
    prop_guards_complete()
    prop_distance_safety()
    prop_braking_monotonicity()
    prop_safe_transition_on_staleness()
    prop_sudden_pedestrian_reaction()
