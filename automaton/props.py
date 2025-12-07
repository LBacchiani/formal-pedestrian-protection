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

    # Initial variables (q is unbounded)
    B_C, B_TTC, B_cs, s_d, s_c, t = declare_continuous_vars()
    q = Const('q', State)
    q_start = Const('q_start', State)
    s.add(q == Normal)
    s.add(q_start == Normal)
    s.add(invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t))
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