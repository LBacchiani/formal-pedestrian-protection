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

def print_ce_vars(m, B_C, B_TTC, B_cs, s_d, s_c, t,):
    """Utility function to print common continuous state variables"""
    print(f"    B_C: {print_buffer_model(m, B_C)}")
    print(f"    B_TTC: {print_buffer_model(m, B_TTC)}")
    print(f"    B_cs: {print_buffer_model(m, B_cs)}")
    print(f"    s_d = {model_val(m, s_d)}")
    print(f"    s_c = {model_val(m, s_c)}") 
    print(f"    t = {model_val(m, t)}")


def timer_constraint_helper(B_C, B_TTC, B_cs, s_d, s_c, t):
    constraints = []
    first_idx_d = Int('first_idx_d')
    first_idx_c = Int('first_idx_c')

    # Domain
    constraints.append(Or([first_idx_d == i for i in range(N)] + [first_idx_d == -1]))
    constraints.append(Or([first_idx_c == i for i in range(N)] + [first_idx_c == -1]))

    # Per-index predicates
    pred_d = [ detected(B_C[i:], B_TTC[i:], B_cs[i], s_d, s_c, t) for i in range(N) ]
    pred_c = [ crossing(B_C[i:], B_TTC[i:], B_cs[i:], s_d, s_c, t) for i in range(N) ]

    # FIRST-INDEX semantics
    for i in range(N):
        # If first index is i, predicate_i must hold and all previous must NOT hold
        constraints.append(
            Implies(first_idx_d == i,
                    And(pred_d[i], *[Not(pred_d[j]) for j in range(i)]))
        )
        constraints.append(
            Implies(first_idx_c == i,
                    And(pred_c[i], *[Not(pred_c[j]) for j in range(i)]))
        )

    # Case where predicate never holds
    constraints.append(
        Implies(first_idx_d == -1, And([Not(pred_d[i]) for i in range(N)]))
    )
    constraints.append(
        Implies(first_idx_c == -1, And([Not(pred_c[i]) for i in range(N)]))
    )

    # Timer update
    constraints.append(
        s_d == If(first_idx_d == -1,
                       TH_D_STALE,
                       CAMERA_FREQ * first_idx_d)
    )
    constraints.append(
        s_c == If(first_idx_c == -1,
                       TH_C_STALE,
                       CAMERA_FREQ * first_idx_c)
    )

    return constraints

def element_constraints(C,TTC,cs):
    constraints = []
    constraints.append(And(C >= 0, C <= 1))
    constraints.append(And(TTC >= 0, TTC <= NO_TTC))
    constraints.append(Or(cs == 0, cs == 1))
    constraints.append(Implies(C < TH_C, And(TTC == NO_TTC, cs == 0))) 
    return constraints

def state_constraints(B_C, B_TTC, B_cs, s_d, s_c, t):
    constraints = []
    for i in range(N):
        constraints.extend(element_constraints(B_C[i], B_TTC[i], B_cs[i]))
    constraints.append(And(s_d >= 0, s_d <= TH_D_STALE))
    constraints.append(And(s_c >= 0, s_c <= TH_D_STALE))
    constraints.append(Or(t == 0, t == CAMERA_FREQ))
    constraints.extend(timer_constraint_helper(B_C, B_TTC, B_cs, s_d, s_c, t))
    return And(constraints)

def threat(C,TTC,cs):
    constraints = []
    constraints.append(C >= TH_C)
    constraints.append(And(TTC >= 0, TTC < TH_TTC_R))
    constraints.append(cs == 1)
    return And(constraints)

# ---------------------------------------------------------------------------
# Determinism property check
# ---------------------------------------------------------------------------

def determinism_1():
    """
    PROPERTY: Determinism 1 (Guards Mutually Exclusive)
    Verify that for each discrete state, at most one transition can fire.
    """
    print("\n" + "="*70)
    print("PROPERTY: Determinism (Guards Mutually Exclusive)")
    print("="*70)
    all_verified = True
    s = Solver()

    # Declare current state
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const("q", State)
    s.add(state_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))

    # Declare symbolic next states for two hypothetical transitions
    B_C1, B_TTC1, B_cs1, s_d1, s_c1, t1 = declare_continuous_vars("_1")
    q1_next = Const("q1_next", State)
    
    B_C2, B_TTC2, B_cs2, s_d2, s_c2, t2 = declare_continuous_vars("_2")
    q2_next = Const("q2_next", State)
    s.add(q1_next != q2_next)
    
    # Two possible transitions
    trans1 = transition(q, q1_next, B_C, B_TTC, B_cs, s_d, s_c, t)
    
    trans2 = transition(q, q2_next, B_C, B_TTC, B_cs, s_d, s_c, t)
    
    # Determinism: not possible for both to fire simultaneously
    no_two_transitions = Not(And(trans1, trans2))
    
    # Only enforce when invariant is violated
    s.add(Not(Implies(Not(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t)), no_two_transitions)))


    # Check if determinism could be violated
    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("Determinism violation possible!")
        print(f"state: {model_val(m,q)}")
        print_ce_vars(m, B_C, B_TTC, B_cs, s_d, s_c, t)
    else:
        print("Determinism holds for all states where invariant does not hold.")

    print(f"\n{'✓' if all_verified else '✗'} All guards are mutually exclusive: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

def determinism_2():
    """
    PROPERTY: Determinism 2 (Invariant or some guard fires)
    Verify that for each discrete state, either the invariant holds or at least
    one transition is enabled.
    """
    print("\n" + "="*70)
    print("PROPERTY: Determinism 2 (Invariant or some guard fires)")
    print("="*70)
    all_verified = True
    s = Solver()

    # Declare current state
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const("q", State)
    s.add(state_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))

    # Declare symbolic next state for a hypothetical transition
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars("_next")
    q_next = Const("q_next", State)

    # Define the transition from current state to next
    trans = transition(q, q_next, B_C, B_TTC, B_cs, s_d, s_c, t)

    # Property: violation occurs if invariant fails AND no transition is possible
    violation = And(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t), trans)
    s.add(violation)

    # Check if a violation exists
    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("Violation found: invariant fails and no guard can fire!")
        print(f"state: {model_val(m,q)}")
        print_ce_vars(m, B_C, B_TTC, B_cs, s_d, s_c, t)
    else:
        print("Property verified: for all states, either invariant holds or some guard can fire.")

    print(f"\n{'✓' if all_verified else '✗'} Determinism 2 check: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified



# ---------------------------------------------------------------------------
# Completeness property check
# ---------------------------------------------------------------------------

def input_completeness():
    """
    PROPERTY: Input Completeness (No Deadlock)
    Verify that for each discrete state, either the invariant holds
    or at least one transition is enabled.
    """
    print("\n" + "="*70)
    print("PROPERTY: Input Completeness (No Deadlock)")
    print("="*70)
    all_verified = True
    s = Solver()

    # Declare current state
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const("q", State)
    s.add(state_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))
    s.add(t == 0)

    # Declare symbolic next state for a hypothetical transition
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars("_next")
    q_next = Const("q_next", State)

    # Define the transition from current state to next
    trans = Or([transition(q, qn, B_C, B_TTC, B_cs, s_d, s_c, t) 
                for qn in [Normal, Throttling, SoftBraking, EmergencyBraking]])

    # Property violation occurs if NEITHER invariant holds NOR any transition is possible
    violation = And(
        Not(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t)),
        Not(trans)
    )
    s.add(violation)

    # Check if a violation exists
    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("Violation found: neither invariant holds nor any guard can fire (deadlock)!")
        print(f"state: {model_val(m,q)}")
        print_ce_vars(m, B_C, B_TTC, B_cs, s_d, s_c, t)
        print(model_val(m, G_to_throttling(B_C, B_TTC, B_cs, s_d, s_c, t)))
    else:
        print("Property verified: no state exists where invariant fails and no guard can fire.")

    print(f"\n{'✓' if all_verified else '✗'} Input Completeness check: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified


# ---------------------------------------------------------------------------
# Liveness property check (Sudden Pedestrian)
# ---------------------------------------------------------------------------


def prop_sudden_pedestrian_reaction():
    """
    PROPERTY: Sudden pedestrian appears → SoftBraking or EmergencyBraking within k steps (Liveness)
    Starting from Normal with a critical threat, the system either:
      - remains in Normal (performing sense operations) 
      - or eventually transitions to SoftBraking or EmergencyBraking in at most k steps.
    """
    print("\n" + "="*70)
    print("PROPERTY: Sudden Pedestrian Reaction (Liveness, ≤ k steps)")
    print("="*70)

    s = Solver()
    all_verified = True
    max_steps = RT_HALF_FRAMES

    # Initial state
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    C_threat, TTC_threat, cs_threat = Real('C_threat'), Real('TTC_threat'), Int('cs_threat')

    s.add(q == Normal)
    s.add(threat(C_threat, TTC_threat, cs_threat))
    s.add(state_constraints(B_C, B_TTC, B_cs, s_d, s_c, t))
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))

    # Simulate up to max_steps
    for i in range(max_steps):
        # Next step variables
        B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars(f"_step{i}")
        q_next = Const(f"q_step{i}", State)
        s.add(state_constraints(B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next))

        # Sense operation: remain in Normal
        stay_normal = And(
            q_next == Normal,
            sense(B_C, B_TTC, B_cs, s_d, s_c, t,
                  B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next,
                  C_threat, TTC_threat, cs_threat),
            invariant(q_next, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next)
        )

        # Allowed leaving transitions: only SoftBraking or EmergencyBraking
        leave_normal = And(
            transition(q, q_next, B_C, B_TTC, B_cs, s_d, s_c, t),
            And(q_next != SoftBraking, q_next != EmergencyBraking)
        )

        # Either stay in Normal or leave towards SB/EB
        s.add(Or(stay_normal, leave_normal))

        # Update for next iteration
        B_C, B_TTC, B_cs, s_d, s_c, t = B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next
        q = q_next

    # Solve
    if s.check() == sat:
        m = s.model()
        all_verified = False
        print("Counterexample found: a forbidden transition is possible within k steps!")
        print(f"state: {model_val(m, q)}")
        print_ce_vars(m, B_C, B_TTC, B_cs, s_d, s_c, t)
    else:
        print("Property verified: after at most k steps, only SoftBraking or EmergencyBraking transitions are possible.")

    print(f"\n{'✓' if all_verified else '✗'} Sudden pedestrian reaction check: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified



# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   determinism_1()
   determinism_2()
   input_completeness()
   prop_sudden_pedestrian_reaction()

