"""
Z3 setup and helper functions for pedestrian protection automaton verification.

This module provides the Z3 encoding of the automaton's state variables,
helper predicates, guard conditions, and global invariants enforcement.
"""

from z3 import *
from automaton import *


# ============================================================================
# Z3 VARIABLE DECLARATIONS
# ============================================================================

def create_buffer_vars(name_prefix: str, size: int, sort):
    """Create Z3 variables for a buffer of given size and sort."""
    return [Const(f"{name_prefix}_{i}", sort) for i in range(size)]


def create_automaton_vars():
    """
    Create all Z3 variables representing the automaton state.
    
    Returns:
        Dictionary containing all Z3 variables
    """
    # Buffers
    B_C = create_buffer_vars("B_C", N, RealSort())
    B_TTC = create_buffer_vars("B_TTC", N, RealSort())
    B_cross = create_buffer_vars("B_cross", N, IntSort())
    
    # Staleness timers
    s_d = Real('s_d')
    s_c = Real('s_c')
    s_u = Real('s_u')  # Uncertainty timer
    
    # Timing variables
    sns = Int('sns')  # Sensor flag: 0 or 1
    t = Real('t')  # Time since last sensor value
    
    return {
        'B_C': B_C,
        'B_TTC': B_TTC,
        'B_cross': B_cross,
        's_d': s_d,
        's_c': s_c,
        's_u': s_u,
        'sns': sns,
        't': t
    }


# ============================================================================
# GLOBAL INVARIANTS / BUFFER CONSTRAINTS
# ============================================================================

def buffer_constraints(vars_dict):
    """
    Create Z3 constraints enforcing global invariants (I(...)).
    """
    constraints = []
    B_C = vars_dict['B_C']
    B_TTC = vars_dict['B_TTC']
    B_cross = vars_dict['B_cross']
    s_d = vars_dict['s_d']
    s_c = vars_dict['s_c']
    s_u = vars_dict['s_u']
    sns = vars_dict['sns']
    t = vars_dict['t']

    # Sensor flag is boolean
    constraints.append(Or(sns == 0, sns == 1))
    
    # Time is non-negative
    constraints.append(And(t >= 0, t < CAMERA_FREQ))

    constraints.append(Implies(sns == 1, t == 0))
    constraints.append(Implies(sns == 0, t >= CAMERA_FREQ))

    # Element bounds and B_TTC → B_C linkage
    for i in range(N):
        # Confidence values are probabilities
        constraints.append(And(B_C[i] >= 0, B_C[i] <= 1))
        
        # TTC values are positive
        constraints.append(B_TTC[i] > 0)
        
        # Crossing is boolean
        constraints.append(Or(B_cross[i] == 0, B_cross[i] == 1))
        
        # Linkage: high confidence iff TTC is not NO_TTC
        # If C_i < TH_C then TTC_i = NO_TTC
        # If C_i >= TH_C then TTC_i < NO_TTC
        constraints.append(If(B_C[i] < TH_C, B_TTC[i] == NO_TTC, B_TTC[i] < NO_TTC) )
        
        # If crossing, must have high confidence
        constraints.append(Implies(B_cross[i] == 1, B_C[i] >= TH_C))
    
    # Staleness timers computation for detection
    first_idx_d = Int('first_idx_d')
    constraints.append(Or([first_idx_d == i for i in range(N)] + [first_idx_d == -1]))
    
    for i in range(N):
        constraints.append(Implies(first_idx_d == i, And(B_C[i] >= TH_C, And([B_C[j] < TH_C for j in range(i)]), detected(B_C) )))
    
    constraints.append(Implies(first_idx_d == -1, And([B_C[i] < TH_C for i in range(N)])))
    constraints.append(s_d == If(first_idx_d >= 0, CAMERA_FREQ * first_idx_d, MAX_STALE))

    # Staleness timers computation for crossing
    first_idx_c = Int('first_idx_c')
    constraints.append(Or([first_idx_c == i for i in range(N)] + [first_idx_c == -1]))
    
    for i in range(N):
        constraints.append(Implies(first_idx_c == i, And(B_cross[i] == 1, And([B_cross[j] == 0 for j in range(i)]), crossing(B_cross))))
    
    constraints.append(Implies(first_idx_c == -1, And([B_cross[i] == 0 for i in range(N)])))
    constraints.append(s_c == If(first_idx_c >= 0, CAMERA_FREQ * first_idx_c, MAX_STALE))

    # Uncertainty timer constraints
    constraints.append(s_u >= 0)
    constraints.append(Implies(s_d < MAX_STALE, s_u == 0))

    return constraints


# ============================================================================
# HELPER PREDICATES (Z3 encoding)
# ============================================================================

def detected(B_C):
    """
    Z3 encoding of detected() predicate incorporating human reaction time.
    A detection is valid if, within half a human reaction window's worth of frames,
    enough frames show confidence >= TH_C.
    """
    limit = int(min(N, RT_WINDOW_FRAMES))
    count = Sum([If(B_C[i] >= TH_C, 1, 0) for i in range(limit)])
    return count >= RT_HALF_FRAMES


def crossing(B_cross):
    """
    Z3 encoding of crossing() predicate incorporating human reaction time.
    A crossing is valid if, within half a human reaction window's worth of frames,
    enough frames indicate crossing (B_cross == 1).
    """
    limit = int(min(N, RT_WINDOW_FRAMES))
    count = Sum([B_cross[i] for i in range(limit)])
    return count >= RT_HALF_FRAMES


def valid_d(B_C, s_d):
    """Z3 encoding of valid_d() predicate."""
    return Or(detected(B_C), s_d < TH_D_STALE)


def valid_c(B_cross, s_c):
    """Z3 encoding of valid_c() predicate."""
    return Or(crossing(B_cross), s_c < TH_C_STALE)


def s_distance(B_TTC):
    """
    Symbolic safe distance check.
    TTC_i > TH_TTC_s for at least CONSENSUS fraction of the first k frames.
    """
    k = int(S_DISTANCE_CONSENSUS * N)
    count = Sum([If(B_TTC[i] > TH_TTC_S, 1, 0) for i in range(k)])
    threshold = k * CONSENSUS
    return count >= threshold


def s_r_distance(B_TTC):
    """
    Check if distance is safe-to-risky.
    TH_TTC_r < TTC_i <= TH_TTC_s for at least CONSENSUS fraction of the first k frames.
    """
    k = int(SR_DISTANCE_CONSENSUS * N)
    count = Sum([If(And(B_TTC[i] > TH_TTC_R, B_TTC[i] <= TH_TTC_S), 1, 0) for i in range(k)])
    threshold = k * CONSENSUS
    return count >= threshold


def r_c_distance(B_TTC):
    """
    Check if distance is risky-to-critical.
    TH_TTC_c < TTC_i <= TH_TTC_r for at least CONSENSUS fraction of the first k frames.
    """
    k = int(RC_DISTANCE_CONSENSUS * N)
    count = Sum([If(And(B_TTC[i] > TH_TTC_C, B_TTC[i] <= TH_TTC_R), 1, 0) for i in range(k)])
    threshold = k * CONSENSUS
    return count >= threshold


def c_distance(B_TTC):
    """
    Check if distance is critical.
    TTC_i <= TH_TTC_c for at least CONSENSUS fraction of the first k frames.
    Note: The formal definition uses >= for the comparison, which seems inconsistent.
    Using <= here as critical means TTC is at or below the critical threshold.
    """
    k = int(C_DISTANCE_CONSENSUS * N)
    count = Sum([If(B_TTC[i] <= TH_TTC_C, 1, 0) for i in range(k)])
    threshold = k * CONSENSUS
    return count >= threshold


def uncertain(B_TTC):
    return And(
        Not(s_distance(B_TTC)),
        Not(s_r_distance(B_TTC)),
        Not(r_c_distance(B_TTC)),
        Not(c_distance(B_TTC))
    )


# ============================================================================
# GUARD CONDITIONS FOR EACH TRANSITION
# ============================================================================

def get_guards(vars_dict):
    """
    Create Z3 expressions for all transition guards, matching the formal definition exactly.
    """
    B_C = vars_dict['B_C']
    B_TTC = vars_dict['B_TTC']
    B_cross = vars_dict['B_cross']
    s_d = vars_dict['s_d']
    s_c = vars_dict['s_c']
    s_u = vars_dict['s_u']
    sns = vars_dict['sns']
    t = vars_dict['t']
    
    # Compute predicates
    det = detected(B_C)
    cross = crossing(B_cross)
    vd = valid_d(B_C, s_d)
    vc = valid_c(B_cross, s_c)
    sd = s_distance(B_TTC)
    srd = s_r_distance(B_TTC)
    rcd = r_c_distance(B_TTC)
    cd = c_distance(B_TTC)
    unc = uncertain(B_TTC)
    
    guards = {}
    
    # From NORMAL
    guards['e1'] = Or(
        And(Or(Not(vd), sd, unc), sns == 1, t == 0),
        And(sns == 0, t >= CAMERA_FREQ)
    )
    guards['e2'] = And(vd, Not(vc), srd, sns == 1, t == 0)
    guards['e3'] = And(
        vd,
        Or(And(Not(vc), rcd), And(vc, srd)),
        sns == 1, t == 0
    )
    guards['e4'] = And(vd, Not(vc), cd, sns == 1, t == 0)
    guards['e5'] = And(vd, vc, rcd, sns == 1, t == 0)
    guards['e6'] = And(vd, vc, cd, sns == 1, t == 0)
    
    # From SAFE_WARNING
    guards['e7'] = And(
        Or(Not(vd), sd, s_u >= MAX_UNCERTAIN),
        sns == 1, t == 0
    )
    guards['e8'] = Or(
        And(
            Or(And(vd, Not(vc), srd), And(unc, s_u < MAX_UNCERTAIN)),
            sns == 1, t == 0
        ),
        And(sns == 0, t >= CAMERA_FREQ)
    )
    guards['e9'] = guards['e3']
    guards['e10'] = guards['e4']
    guards['e11'] = guards['e5']
    guards['e12'] = guards['e6']
    
    # From THROTTLING
    guards['e13'] = guards['e7']
    guards['e14'] = guards['e2']
    guards['e15'] = Or(
        And(
            vd,
            Or(
                And(Not(vc), rcd),
                And(vc, srd),
                And(unc, s_u < MAX_UNCERTAIN)
            ),
            sns == 1, t == 0
        ),
        And(sns == 0, t >= CAMERA_FREQ)
    )
    guards['e16'] = guards['e4']
    guards['e17'] = guards['e5']
    guards['e18'] = guards['e6']
    
    # From CRITICAL_SLOWDOWN
    guards['e19'] = And(
        Or(Not(vd), sd, s_u >= MAX_UNCERTAIN),
        sns == 1, t == 0
    )
    guards['e20'] = guards['e2']
    guards['e21'] = guards['e3']
    guards['e22'] = Or(
        And(
            Or(
                And(vd, Not(vc), cd),
                And(vd, unc, s_u < MAX_UNCERTAIN)
            ),
            sns == 1, t == 0
        ),
        And(sns == 0, t >= CAMERA_FREQ)
    )
    guards['e23'] = guards['e5']
    guards['e24'] = guards['e6']
    
    # From SOFT_BRAKING
    guards['e25'] = And(Or(Not(vd), Not(vc), sd, s_u >= MAX_UNCERTAIN), sns == 1, t == 0)
    guards['e26'] = And(vd, vc, srd, sns == 1, t == 0)
    guards['e27'] = Or(And(vd, vc, Or(rcd, And(unc, s_u < MAX_UNCERTAIN)), sns == 1, t == 0), And(sns == 0, t >= CAMERA_FREQ))
    guards['e28'] = guards['e6']
    
    # From EMERGENCY_BRAKING
    guards['e29'] = And(Or(Not(vd), Not(vc)), sns == 1, t == 0)
    guards['e30'] = Or(And(vc, sns == 1, t == 0), And(sns == 0, t >= CAMERA_FREQ))
    
    return guards


# ============================================================================
# TRANSITION GROUPS BY STATE
# ============================================================================

# Mapping of states to their outgoing transitions
STATE_TRANSITIONS = {
    'NORMAL': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'],
    'SAFE_WARNING': ['e7', 'e8', 'e9', 'e10', 'e11', 'e12'],
    'THROTTLING': ['e13', 'e14', 'e15', 'e16', 'e17', 'e18'],
    'CRITICAL_SLOWDOWN': ['e19', 'e20', 'e21', 'e22', 'e23', 'e24'],
    'SOFT_BRAKING': ['e25', 'e26', 'e27', 'e28'],
    'EMERGENCY_BRAKING': ['e29', 'e30']
}


# ============================================================================
# INVARIANTS
# ============================================================================

def get_invariants(vars_dict):
    """
    Return the invariant conditions for each discrete state.
    """
    B_C = vars_dict['B_C']
    B_TTC = vars_dict['B_TTC']
    B_cross = vars_dict['B_cross']
    s_d = vars_dict['s_d']
    s_c = vars_dict['s_c']
    s_u = vars_dict['s_u']
    
    vd = valid_d(B_C, s_d)
    vc = valid_c(B_cross, s_c)
    sd = s_distance(B_TTC)
    srd = s_r_distance(B_TTC)
    rcd = r_c_distance(B_TTC)
    cd = c_distance(B_TTC)
    unc = uncertain(B_TTC)
    
    invariants = {}
    
    invariants['NORMAL'] = Or(Not(vd), sd, unc)
    
    invariants['SAFE_WARNING'] = Or(
        And(vd, Not(vc), srd),
        And(unc, s_u < MAX_UNCERTAIN)
    )
    
    invariants['THROTTLING'] = And(
        vd,
        Or(
            And(Not(vc), rcd),
            And(vc, srd),
            And(unc, s_u < MAX_UNCERTAIN)
        )
    )
    
    invariants['CRITICAL_SLOWDOWN'] = Or(
        And(vd, Not(vc), cd),
        And(vd, unc, s_u < MAX_UNCERTAIN)
    )
    
    invariants['SOFT_BRAKING'] = Or(
        And(vd, vc, rcd),
        And(vd, vc, unc, s_u < MAX_UNCERTAIN)
    )
    
    invariants['EMERGENCY_BRAKING'] = vc
    
    return invariants