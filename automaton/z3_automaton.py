"""
Z3 setup and helper functions for pedestrian protection automaton verification.

This module provides the Z3 encoding of the automaton's state variables,
helper predicates, guard conditions, and global invariants enforcement.
"""

from z3 import *

# ============================================================================
# STRUCTURAL CONSTANTS (Python - define the algorithm structure)
# ============================================================================

N = 10  # Buffer size

# Consensus thresholds (percentage of buffer that must agree)
DETECTION_CONSENSUS = 0.7
CROSSING_CONSENSUS = 0.7
S_DISTANCE_CONSENSUS = 0.8
SR_DISTANCE_CONSENSUS = 0.6
RC_DISTANCE_CONSENSUS = 0.4
C_DISTANCE_CONSENSUS = 0.2

# ============================================================================
# SAFETY THRESHOLDS (Z3 - can be verified parametrically)
# ============================================================================

# Python values (for computation)
TH_C_VAL = 0.5
TH_D_STALE_VAL = 200
TH_C_STALE_VAL = 200
TH_TTC_S_VAL = 5.0
TH_TTC_R_VAL = 3.0
TH_TTC_C_VAL = 1.5
MIN_VAL_VAL = 0.8
MAX_STALE_VAL = 1000
MAX_UNCERTAIN_VAL = 2000
NO_TTC_VAL = 10000000
CAMERA_FREQ_VAL = 100

# Z3 constants (for SMT solving)
TH_C = RealVal(TH_C_VAL)  # Threshold for certain detection (confidence)
TH_D_STALE = RealVal(TH_D_STALE_VAL)  # Threshold for detection stale data (ms)
TH_C_STALE = RealVal(TH_C_STALE_VAL)  # Threshold for crossing stale data (ms)

# Time-to-collision thresholds (seconds)
TH_TTC_S = RealVal(TH_TTC_S_VAL)  # Safe TTC threshold
TH_TTC_R = RealVal(TH_TTC_R_VAL)  # Risky TTC threshold
TH_TTC_C = RealVal(TH_TTC_C_VAL)  # Critical TTC threshold
MIN_VAL = RealVal(MIN_VAL_VAL)  # Minimum validation threshold

# Staleness upper bounds
MAX_STALE = RealVal(MAX_STALE_VAL)  # Maximum staleness (ms)
MAX_UNCERTAIN = RealVal(MAX_UNCERTAIN_VAL)  # Handover timeout (ms)

# Sensor parameters
NO_TTC = RealVal(NO_TTC_VAL)  # No TTC value (pedestrian not detected)
CAMERA_FREQ = RealVal(CAMERA_FREQ_VAL)  # Camera frequency (ms)

# ============================================================================
# THRESHOLD SANITY CONSTRAINTS
# ============================================================================

def threshold_constraints():
    """
    Return Z3 constraints ensuring thresholds are well-formed.
    These can be added to the solver to ensure threshold consistency.
    """
    return [
        # TTC thresholds are ordered
        TH_TTC_C < TH_TTC_R,
        TH_TTC_R < TH_TTC_S,
        TH_TTC_C > 0,
        
        # Confidence threshold is a probability
        TH_C >= 0,
        TH_C <= 1,
        
        # MIN_VAL is a probability
        MIN_VAL >= 0,
        MIN_VAL <= 1,
        
        # Staleness thresholds are positive
        TH_D_STALE > 0,
        TH_C_STALE > 0,
        MAX_STALE > 0,
        MAX_UNCERTAIN > 0,
        
        # Uncertainty timeout should be longer than staleness thresholds
        MAX_UNCERTAIN > TH_D_STALE,
        MAX_UNCERTAIN > TH_C_STALE,
        
        # MAX_STALE should be longer than detection staleness threshold
        MAX_STALE >= TH_D_STALE,
        MAX_STALE >= TH_C_STALE,
        
        # Camera frequency is positive
        CAMERA_FREQ > 0,
        
        # NO_TTC is a large sentinel value
        NO_TTC > TH_TTC_S * 1000,
    ]


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
    
    return {
        'B_C': B_C,
        'B_TTC': B_TTC,
        'B_cross': B_cross,
        's_d': s_d,
        's_c': s_c,
        's_u': s_u
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

    # Element bounds and B_TTC → B_C linkage
    for i in range(N):
        # Confidence values are probabilities
        constraints.append(And(B_C[i] >= 0, B_C[i] <= 1))
        
        # TTC values are positive
        constraints.append(B_TTC[i] > 0)
        
        # Crossing is boolean
        constraints.append(Or(B_cross[i] == 0, B_cross[i] == 1))
        
        # Linkage: high confidence iff TTC is not NO_TTC
        constraints.append((B_C[i] >= TH_C) == (B_TTC[i] < NO_TTC))
        
        # If crossing, must have high confidence
        constraints.append(Implies(B_cross[i] == 1, B_C[i] > TH_C))
    
    # Staleness timers computation for detection
    first_idx_d = Int('first_idx_d')
    constraints.append(Or([first_idx_d == i for i in range(N)] + [first_idx_d == -1]))
    
    for i in range(N):
        constraints.append(
            Implies(
                first_idx_d == i,
                And(
                    B_C[i] >= TH_C,
                    And([B_C[j] < TH_C for j in range(i)]),
                    detected(B_C)
                )
            )
        )
    
    constraints.append(Implies(first_idx_d == -1, And([B_C[i] < TH_C for i in range(N)])))
    constraints.append(s_d == If(first_idx_d >= 0, CAMERA_FREQ * first_idx_d, MAX_STALE))

    # Staleness timers computation for crossing
    first_idx_c = Int('first_idx_c')
    constraints.append(Or([first_idx_c == i for i in range(N)] + [first_idx_c == -1]))
    
    for i in range(N):
        constraints.append(
            Implies(
                first_idx_c == i,
                And(
                    B_cross[i] == 1,
                    And([B_cross[j] == 0 for j in range(i)]),
                    crossing(B_cross)
                )
            )
        )
    
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
    """Z3 encoding of detected() predicate."""
    count = Sum([If(c > TH_C, 1, 0) for c in B_C])
    return count >= DETECTION_CONSENSUS * N


def crossing(B_cross):
    """Z3 encoding of crossing() predicate."""
    count = Sum(B_cross)
    return count >= CROSSING_CONSENSUS * N


def valid_d(B_C, s_d):
    """Z3 encoding of valid_d() predicate."""
    return Or(detected(B_C), s_d < TH_D_STALE)


def valid_c(B_cross, s_c):
    """Z3 encoding of valid_c() predicate."""
    return Or(crossing(B_cross), s_c < TH_C_STALE)


def s_distance(B_TTC):
    """Check if distance is safe."""
    k = int(S_DISTANCE_CONSENSUS * N)
    count = Sum([If(B_TTC[i] >= TH_TTC_S, 1, 0) for i in range(k)])
    return count >= MIN_VAL * k


def s_r_distance(B_TTC):
    """Check if distance is safe-to-risky."""
    k = int(SR_DISTANCE_CONSENSUS * N)
    count = Sum([If(And(B_TTC[i] >= TH_TTC_R, B_TTC[i] < TH_TTC_S), 1, 0) for i in range(k)])
    return count >= MIN_VAL * k


def r_c_distance(B_TTC):
    """Check if distance is risky-to-critical."""
    k = int(RC_DISTANCE_CONSENSUS * N)
    count = Sum([If(And(B_TTC[i] >= TH_TTC_C, B_TTC[i] < TH_TTC_R), 1, 0) for i in range(k)])
    return count >= MIN_VAL * k


def c_distance(B_TTC):
    """Check if distance is critical."""
    limit = int(C_DISTANCE_CONSENSUS * N)
    return And([B_TTC[i] < TH_TTC_C for i in range(limit)])


def uncertain_distance(B_TTC):
    """Check if distance classification is uncertain (none of the categories apply)."""
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
    Create Z3 expressions for all transition guards, now including the
    uncertainty timer s_u to enforce handover when uncertain.
    """
    B_C = vars_dict['B_C']
    B_TTC = vars_dict['B_TTC']
    B_cross = vars_dict['B_cross']
    s_d = vars_dict['s_d']
    s_c = vars_dict['s_c']
    s_u = vars_dict['s_u']
    
    # Compute predicates
    det = detected(B_C)
    cross = crossing(B_cross)
    vd = valid_d(B_C, s_d)
    vc = valid_c(B_cross, s_c)
    sd = s_distance(B_TTC)
    srd = s_r_distance(B_TTC)
    rcd = r_c_distance(B_TTC)
    cd = c_distance(B_TTC)
    unc_d = uncertain_distance(B_TTC)
    
    guards = {}
    
    # From NORMAL
    guards['e1'] = Or(Not(vd), sd, unc_d)
    guards['e2'] = And(vd, Not(vc), srd)
    guards['e3'] = And(vd, Or(And(Not(vc), rcd), And(vc, srd)))
    guards['e4'] = And(vd, Not(vc), cd)
    guards['e5'] = And(vd, vc, rcd)
    guards['e6'] = And(vd, vc, cd)
    
    # From SAFE_WARNING
    guards['e7'] = Or(Not(vd), sd, s_u >= MAX_UNCERTAIN)  # handover if uncertain too long
    guards['e8'] = Or(And(vd, Not(vc), srd), And(vd, unc_d, s_u < MAX_UNCERTAIN))
    guards['e9'] = And(vd, Or(And(Not(vc), rcd), And(vc, srd)))
    guards['e10'] = And(vd, Not(vc), cd)
    guards['e11'] = And(vd, vc, rcd)
    guards['e12'] = And(vd, vc, cd)
    
    # From THROTTLING
    guards['e13'] = Or(Not(vd), sd, s_u >= MAX_UNCERTAIN)
    guards['e14'] = And(vd, Not(vc), srd)
    guards['e15'] = And(vd, Or(And(Not(vc), rcd), And(vc, srd), And(unc_d, s_u < MAX_UNCERTAIN)))
    guards['e16'] = And(vd, Not(vc), cd)
    guards['e17'] = And(vd, vc, rcd)
    guards['e18'] = And(vd, vc, cd)
    
    # From CRITICAL_SLOWDOWN
    guards['e19'] = Or(Not(vd), sd, s_u >= MAX_UNCERTAIN)
    guards['e20'] = And(vd, Not(vc), srd)
    guards['e21'] = Or(And(Not(vc), rcd), And(vc, srd))
    guards['e22'] = Or(And(vd, Not(vc), cd), And(vd, And(unc_d, s_u < MAX_UNCERTAIN)))
    guards['e23'] = And(vd, vc, rcd)
    guards['e24'] = And(vd, vc, cd)
    
    # From SOFT_BRAKING
    guards['e25'] = Or(Not(vd), Not(vc), sd, s_u >= MAX_UNCERTAIN)
    guards['e26'] = And(vd, vc, srd)
    guards['e27'] = Or(And(vd, vc, rcd), And(vc, vd, And(unc_d, s_u < MAX_UNCERTAIN)))
    guards['e28'] = And(vd, vc, cd)
    
    # From EMERGENCY_BRAKING
    guards['e29'] = Or(Not(det), Not(cross))
    guards['e30'] = cross
    
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