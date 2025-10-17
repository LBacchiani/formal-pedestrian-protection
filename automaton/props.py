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

def buffer_invariants(B_C, B_TTC, B_cs, s_d, s_c, s_u, t):
    inv = []
    N = len(B_C)
    for i in range(N):
        inv.append(And(B_C[i] >= 0, B_C[i] <= 1))
        inv.append(Or(B_TTC[i] == NO_TTC, B_TTC[i] > 0))
        inv.append(Or(B_cs[i] == 0, B_cs[i] == 1))
        inv.append(Implies(B_C[i] < TH_C, And(B_cs[i] == 0,B_TTC[i] == NO_TTC)))
    inv.append(And(s_d > 0, s_d <= TH_D_STALE, s_c > 0, s_c <= TH_C_STALE, s_u > 0, s_u <= MAX_UNCERTAIN, t >= 0, t <= CAMERA_FREQ))
    return And(inv)

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
    buffer_invariants(B_C, B_TTC, B_cs, s_d, s_c, s_u, t)
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

# ---------------------------------------------------------------------------
# Completeness property check (corrected)
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
    buffer_invariants(B_C, B_TTC, B_cs, s_d, s_c, s_u, t)
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



# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prop_guards_mutually_exclusive()
    prop_guards_complete()
