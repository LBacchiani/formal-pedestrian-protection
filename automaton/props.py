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

def print_ce_vars(m, s_d, s_c, t, B_C, B_TTC, B_cs):
    """Utility function to print common continuous state variables"""
    print(f"    s_d = {model_val(m, s_d)}, s_c = {model_val(m, s_c)}, t = {model_val(m, t)}")
    print(f"    B_C: {print_buffer_model(m, B_C)}")
    print(f"    B_TTC: {print_buffer_model(m, B_TTC)}")
    print(f"    B_cs: {print_buffer_model(m, B_cs)}")

def buffer_constraints(B_C, B_TTC, B_cs, s_d, s_c, t):
    constraints = []
    for i in range(N):
        constraints.append(And(B_C[i] >= 0, B_C[i] <= 1))
        constraints.append(And(B_TTC[i] >= 0, B_TTC[i] <= NO_TTC))
        constraints.append(Or(B_cs[i] == 0, B_cs[i] == 1))
        constraints.append(Implies(B_C[i] <= TH_C, And(B_TTC[i] == NO_TTC, B_cs[i] == 0)))
    constraints.append(Implies(detected(B_C, B_TTC, B_cs, s_d, s_c, t), s_d == 0))
    constraints.append(Implies(crossing(B_C, B_TTC, B_cs, s_d, s_c, t), s_c == 0))

    ####s_d constraints####
    first_idx_d = Int('first_idx_d')
    constraints.append(Or([first_idx_d == i for i in range(N)] + [first_idx_d == -1]))
    for i in range(N):
        constraints.append(Implies(first_idx_d == i, And(B_C[i] >= TH_C, And([B_C[j] < TH_C for j in range(i)]), detected(B_C, B_TTC, B_cs, s_d, s_c, t) )))
    constraints.append(Implies(first_idx_d == -1, And([B_C[i] < TH_C for i in range(N)])))
    constraints.append(s_d == If(first_idx_d >= 0, CAMERA_FREQ * first_idx_d, TH_D_STALE))
    
    ####s_c constraints####
    first_idx_c = Int('first_idx_c')
    constraints.append(Or([first_idx_c == i for i in range(N)] + [first_idx_c == -1]))
    for i in range(N):
        constraints.append(Implies(first_idx_c == i, And(B_cs[i] == 1, And([B_cs[j] == 0 for j in range(i)]), crossing(B_C, B_TTC, B_cs, s_d, s_c, t))))
    constraints.append(Implies(first_idx_c == -1, And([B_cs[i] == 0 for i in range(N)])))
    constraints.append(s_c == If(first_idx_c >= 0, CAMERA_FREQ * first_idx_c, TH_C_STALE))

    return And(constraints)

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

    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    states = [Normal, 
    # SafeWarning, 
    Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, t) for qn in states ]

        for i in range(len(trans_constraints)):
            for j in range(i + 1, len(trans_constraints)):
                s.push()
                s.add(buffer_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))
                s.add(q == q_val)
                s.add(trans_constraints[i])
                s.add(trans_constraints[j])
                s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))

                if s.check() == sat:
                    m = s.model()
                    all_verified = False
                    print(f"\n✗ Non-determinism detected for q = {q_val}")
                    print(f"  Conflicting transitions to: {states[i]} and {states[j]}")
                    print("  Counterexample full state:")
                    print_ce_vars(m, s_d, s_c, t, B_C, B_TTC, B_cs)
                s.pop()

    print(f"\n{'✓' if all_verified else '✗'} All guards are mutually exclusive: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

def prop_invariant_vs_transitions_exclusive():
    """
    PROPERTY: Invariant vs Transitions Mutual Exclusivity
    Verify that in any discrete state, either the invariant holds or transitions can fire, never both.
    """
    print("\n" + "="*70)
    print("PROPERTY: Invariant vs Transitions Mutual Exclusivity")
    print("="*70)

    s = Solver()
    all_verified = True

    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    states = [Normal, 
    # SafeWarning, 
    Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        # Collect all transitions from q_val
        trans_constraints = [transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, t) for qn in states]

        s.push()
        s.add(buffer_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))
        s.add(q == q_val)
        s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))
        s.add(Or(trans_constraints))  # At least one transition is fireable

        if s.check() == sat:
            m = s.model()
            all_verified = False
            # Identify which transitions are true in this model
            firing_transitions = []
            for idx, trans in enumerate(trans_constraints):
                s.push()
                s.add(trans)
                if s.check() == sat:
                    firing_transitions.append(states[idx])
                s.pop()

            print(f"\n✗ Property violated for q = {q_val}")
            print(f"  Invariant holds AND transitions fireable: {firing_transitions}")
            print("  Counterexample full state:")
            print_ce_vars(m, s_d, s_c, t, B_C, B_TTC, B_cs)
        s.pop()

    print(f"\n{'✓' if all_verified else '✗'} Property check: {'VERIFIED' if all_verified else 'FAILED'}")
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

    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    states = [Normal, 
    # SafeWarning, 
    Throttling, SoftBraking, EmergencyBraking]

    for q_val in states:
        trans_constraints = [ transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, t) for qn in states ]
    
        s.push()
        s.add(buffer_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))
        s.add(q == q_val)
        s.add(Not(Or(trans_constraints)))      # No transition is fireable
        s.add(Not(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t)))  # Invariant fails (VIOLATION)

        if s.check() == sat:
            m = s.model()
            all_verified = False
            print(f"\n✗ Completeness violation detected for q = {q_val}")
            print("  No fireable transition AND invariant does NOT hold (Deadlock State Found)!")
            print("  Counterexample full state:")
            print_ce_vars(m, s_d, s_c, t, B_C, B_TTC, B_cs)
        s.pop()

    print(f"\n{'✓' if all_verified else '✗'} Completeness {'VERIFIED' if all_verified else 'FAILED'}")
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
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    q_start = Const('q_start', State)
    s.add(q == Normal)
    s.add(q_start == Normal)
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))
    s.add(invariant(q_start, B_C, B_TTC, B_cs, s_d, s_c, t))
    max_steps = RT_HALF_FRAMES # The maximum allowed steps
    t_cont = Int(f't_cont{0}')
    s.add(t_cont >= CAMERA_FREQ)
    #s.add(t >= CAMERA_FREQ)

    

    for step in range(max_steps):
        B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars(f"_next{step}")
        #B_C_reset, B_TTC_reset, B_cs_reset, , s_c_reset, t_reset = declare_continuous_vars(f"_reset{step}")
        C_new_critical = Real('C_new_critical') 
        s.add(C_new_critical >= TH_C)
        TTC_new_critical = Real('TTC_new_critical')
        s.add(TTC_new_critical < TH_TTC_R)
        cs_new_crossing = Int('cs_new_crossing')
        s.add(cs_new_crossing == 1)
        q_next = Const(f'q_next{step}', State)
        t_cont_next = Int(f't_cont{step+1}')

        s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, t_cont, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_cont_next, C_new_critical, TTC_new_critical, cs_new_crossing))
        s.add(Or(And(invariant(q, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_cont_next)), 
        And(reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t_cont, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_cont_next), q == q_next, transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_cont))))




        #s.add(If(t >= CAMERA_FREQ, sense(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next, C_new_critical, TTC_new_critical, cs_new_crossing), reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next)))

        # 1. Sense Jump (X -> X_next): Applies critical input
        # s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, t_cont, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next, C_new_critical, TTC_new_critical, cs_new_crossing))
        # s.add(det_coll(B_C_next, B_TTC_next, B_cs_next))

        # 2. Transition (q -> q_next) - GUARD CHECK ON X_NEXT
        #s.add(Or(invariant(q, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next), And(q == q_next, transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next))))

        B_C, B_TTC, B_cs, s_d, s_c, t = B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next
 

    
    
    s.add(q != EmergencyBraking)
    s.add(q != SoftBraking)
    
    
    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("\n✗ Sudden pedestrian safety violation: did NOT reach SoftBraking/EmergencyBraking in time")
        print(f"  q_init = {model_val(m,q_start)}")
        print(f"  q_final = {model_val(m,q)}")
            # print state trace for context
        print(f"  B_C: {print_buffer_model(m, B_C)}")
        print(f"  B_TTC: {print_buffer_model(m, B_TTC)}")
        print(f"  B_cs: {print_buffer_model(m, B_cs)}")
        print(f"  t: {model_val(m, t_next)}")
        print(f"TRANS: {m.eval(transition(q_start, q_next, B_C, B_TTC, B_cs, s_d, s_c, t))}")
        print(f"INV: {m.eval(invariant(q_next, B_C, B_TTC, B_cs, s_d, s_c, t))}")
        print(m.eval(s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, t)))
    else:
        print("\n✓ Sudden pedestrian → EmergencyBraking within required steps: VERIFIED")


    print("="*70)
    return all_verified


# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   prop_guards_mutually_exclusive()
   prop_invariant_vs_transitions_exclusive()
   prop_guards_complete()
   prop_sudden_pedestrian_reaction()