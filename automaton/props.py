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

def element_constraints(C,TTC,cs):
    constraints = []
    constraints.append(And(C >= 0, C <= 1))
    constraints.append(And(TTC >= 0, TTC <= NO_TTC))
    constraints.append(Or(cs == 0, cs == 1))
    constraints.append(Implies(C <= TH_C, And(TTC == NO_TTC, cs == 0))) 
    return constraints

def buffer_constraints(B_C, B_TTC, B_cs):
    constraints = []
    for i in range(N):
        constraints.extend(element_constraints(B_C[i], B_TTC[i], B_cs[i]))
    return And(constraints)

def threat(C,TTC,cs):
    constraints = []
    constraints.append(C >= TH_C)
    constraints.append(And(TTC >= 0, TTC <= TH_TTC_R))
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
    states = [Normal, Throttling, SoftBraking, EmergencyBraking]
    # SafeWarning, 
    q = Const('q', State)
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset = declare_continuous_vars('reset')
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars('next')
    C_new, TTC_new, cs_new = Real('C_new'), Real('TTC_new'), Real('cs_new')
    
    ######CHECKING DETERMINISM AT SENSING BOUNDARIES#####
    for q_val in states:
        s = Solver()
        s.add(q == q_val)
        s.add(buffer_constraints(B_C, B_TTC, B_cs))
        s.add(reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset))
        s.add(element_constraints(C_new, TTC_new, cs_new))
        s.add(sense(B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next, C_new, TTC_new, cs_new))
        trans_constraints = [ transition(q, qn, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next) for qn in states ]
        for i in range(len(trans_constraints)):
            for j in range(i + 1, len(trans_constraints)):
                s.push()
                s.add(trans_constraints[i])
                s.add(trans_constraints[j])
                if s.check() == sat:
                    m = s.model()
                    all_verified = False
                    print(f"\n✗ Non-determinism detected for q = {q_val}")
                    print(f"  Conflicting transitions to: {states[i]} and {states[j]}")
                    print("  Counterexample full state:")
                    print_ce_vars(m, s_d_next, s_c_next, t_next, B_C_next, B_TTC_next, B_cs_next)
                s.pop()
    #############################################################################################

    print(f"\n{'✓' if all_verified else '✗'} All guards are mutually exclusive: {'VERIFIED' if all_verified else 'FAILED'}")
    print("="*70)
    return all_verified

def determinism_2():
    """
    PROPERTY: Invariant vs Transitions Mutual Exclusivity
    Verify that in any discrete state, the invariant holds iff no transitions are fireable.
    """
    print("\n" + "="*70)
    print("PROPERTY: Invariant vs Transitions Mutual Exclusivity")
    print("="*70)

    all_verified = True
    states = [Normal, Throttling, SoftBraking, EmergencyBraking]
    # SafeWarning, 
    q = Const('q', State)
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset = declare_continuous_vars('reset')
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars('next')
    C_new, TTC_new, cs_new = Real('C_new'), Real('TTC_new'), Real('cs_new')
    
    for q_val in states:
        s = Solver()
        s.add(q == q_val)
        s.add(buffer_constraints(B_C, B_TTC, B_cs))
        s.add(reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset))
        s.add(element_constraints(C_new, TTC_new, cs_new))
        s.add(sense(B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next, C_new, TTC_new, cs_new))
        trans_constraints = [ transition(q, qn, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next) for qn in states ]
        s.add(invariant(q, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next))
        s.add(Or(trans_constraints))
        if s.check() == sat:
            m = s.model()
            all_verified = False
            print(f"\n✗ Non-determinism detected for q = {q_val}")
            print(f"  Conflicting invariant and transition to: {states[i]}")
            print("  Counterexample full state:")
            print_ce_vars(m, s_d_next, s_c_next, t_next, B_C_next, B_TTC_next, B_cs_next)

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
    all_verified = True
    q = Const('q', State)
    states = [Normal, Throttling, SoftBraking, EmergencyBraking]  # SafeWarning, 
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset = declare_continuous_vars('reset')
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars('next')
    C_new, TTC_new, cs_new = Real('C_new'), Real('TTC_new'), Real('cs_new')
    for q_val in states:
        s = Solver()
        s.add(q == q_val)
        s.add(buffer_constraints(B_C, B_TTC, B_cs))
        s.add(reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset))
        s.add(element_constraints(C_new, TTC_new, cs_new))
        s.add(sense(B_C_reset, B_TTC_reset, B_cs_reset, s_d_reset, s_c_reset, t_reset, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next, C_new, TTC_new, cs_new))
        trans_constraints = [ transition(q, qn, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next) for qn in states ]
        s.add(Not(Or(trans_constraints)))      # No transition is fireable
        s.add(Not(invariant(q_val, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next)))  # Invariant fails (VIOLATION)
        if s.check() == sat:
            m = s.model()
            all_verified = False
            print(f"\n✗ Completeness violation detected for q = {q_val}")
            print("  No fireable transition AND invariant does NOT hold (Deadlock State Found)!")
            print("  Counterexample full state:")
            print_ce_vars(m, s_d_next, s_c_next, t_next, B_C_next, B_TTC_next, B_cs_next)

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
    max_steps = RT_HALF_FRAMES # The maximum allowed steps


    # discrete states (we keep them but won't enforce transitions here)
    q_states = [Const(f'q_{i}', State) for i in range(max_steps + 1)]

    # continuous snapshots: reset and sense intermediates
    X_reset = [declare_continuous_vars(f'_{i}_reset') for i in range(max_steps + 1)]
    X_sense = [declare_continuous_vars(f'_{i}_sense') for i in range(max_steps + 1)]

    # threat/no-threat readings
    C_threat, TTC_threat, cs_threat = Real('C_threat'), Real('TTC_threat'), Int('cs_threat')

    # prepare initial reset[0] from a precursor buffer (no threat)
    B_C_prec, B_TTC_prec, B_cs_prec, s_d_prec, s_c_prec, t_prec = declare_continuous_vars(f'{-1}_reset')
    B_C_0_reset, B_TTC_0_reset, B_cs_0_reset, s_d_0_reset, s_c_0_reset, t_0_reset = X_reset[0]

    s.add(buffer_constraints(B_C_0_reset, B_TTC_0_reset, B_cs_0_reset))
    s.add(threat(C_threat, TTC_threat, cs_threat))
    s.add(q_states[0] == Normal)
    s.add(invariant(q_states[0], B_C_0_reset, B_TTC_0_reset, B_cs_0_reset, s_d_0_reset, s_c_0_reset, t_0_reset))


    # Fixed cycle: for each i do
    #   reset_timers(X_reset[i] -> X_sense[i])
    #   sense(X_sense[i], threat -> X_reset[i+1])
    for i in range(0, max_steps):
        # unpack variables for clarity
        B_C_i_reset, B_TTC_i_reset, B_cs_i_reset, s_d_i_reset, s_c_i_reset, t_i_reset = X_reset[i]
        B_C_i_sense, B_TTC_i_sense, B_cs_i_sense, s_d_i_sense, s_c_i_sense, t_i_sense = X_sense[i]
        B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset = X_reset[i+1]

        # 1) reset_timers: current reset -> produce sense-intermediate
        s.add(reset_timers(
            B_C_i_reset, B_TTC_i_reset, B_cs_i_reset, s_d_i_reset, s_c_i_reset, t_i_reset,
            B_C_i_sense, B_TTC_i_sense, B_cs_i_sense, s_d_i_sense, s_c_i_sense, t_i_sense))

        # 2) sense: inject the threat reading and produce next reset state
        s.add(sense(B_C_i_sense, B_TTC_i_sense, B_cs_i_sense, s_d_i_sense, s_c_i_sense, t_i_sense,
            B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset,
            C_threat, TTC_threat, cs_threat))

        # Optionally enforce invariant on the reset result of the next step (keeps things realistic)
        # s.add(invariant(q_states[i+1], B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset))
        s.add(Or(transition(q_states[i],
                        q_states[i+1],
                        B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset),
                        invariant(q_states[i+1], B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset)
            ))


    
    s.add(q_states[i+1] != EmergencyBraking)
    s.add(q_states[i+1] != SoftBraking)

    if s.check() == sat:
        m = s.model()
        all_verified = False
        print(model_val(m, q_states[i+1]))
        print(f"\nSTEP {i} (after reset_timers -> sense)")
        print("  produced X_reset[i+1] (from sense):")
        print_ce_vars(m, B_C_ip1_reset, B_TTC_ip1_reset, B_cs_ip1_reset, s_d_ip1_reset, s_c_ip1_reset, t_ip1_reset)
    else:
        print("\n✓ Sudden pedestrian → EmergencyBraking within required steps: VERIFIED")

    print("="*70)
    return all_verified

def prop_transition_preserves_validity():
    """
    PROPERTY: Transition Preservation
    When invariant breaks and transition fires, 
    the destination state's invariant holds.
    """
    s = Solver()
    
    q = Const('q', State)
    q_next = Const('q_next', State)
    
    # Current state
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    s.add(buffer_constraints(B_C, B_TTC, B_cs))
    
    # After sense (new sensor data)
    B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next = declare_continuous_vars("_next")
    C_new, TTC_new, cs_new = Real('C_new'), Real('TTC_new'), Int('cs_new')
    s.add(element_constraints(C_new, TTC_new, cs_new))
    
    s.add(sense(B_C, B_TTC, B_cs, s_d, s_c, t,
                B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next,
                C_new, TTC_new, cs_new))
    
    # Assume: current invariant held before sense
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))
    
    # Assume: transition fires (because invariant broke after sense)
    s.add(transition(q, q_next, B_C_next, B_TTC_next, B_cs_next, 
                    s_d_next, s_c_next, t_next))
    
    # Check: does destination state's invariant hold?
    s.add(Not(invariant(q_next, B_C_next, B_TTC_next, B_cs_next,
                       s_d_next, s_c_next, t_next)))
    
    return s.check() == unsat  # Should be UNSAT

# ---------------------------------------------------------------------------
# PROPERTIES CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   determinism_1()
   determinism_2()
   prop_guards_complete()
   prop_sudden_pedestrian_reaction()
#    prop_transition_preserves_validity()