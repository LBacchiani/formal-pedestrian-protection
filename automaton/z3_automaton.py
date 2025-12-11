from z3 import *
from automaton import *
import math

# ============================================================================
# DISCRETE STATES (Q)
# ============================================================================
State, (Normal, 
# SafeWarning, 
Throttling, SoftBraking, EmergencyBraking) = EnumSort(
    'State', 
    ['Normal', 
    # 'SafeWarning', 
    'Throttling', 'SoftBraking', 'EmergencyBraking']
)

# ============================================================================
# CONTINUOUS VARIABLES (X)
# ============================================================================
def declare_continuous_vars(suffix=""):
    """
    Declare continuous variables X = {C_0, ..., C_{n-1}, TTC_0, ..., TTC_{n-1}, 
                                      cs_0, ..., cs_{n-1}, s_d, s_c, t}
    """
    # Confidence levels B_C = C_0, ..., C_{n-1}
    B_C = [Real(f'C_{i}{suffix}') for i in range(N)]
    
    # Time-to-collision B_TTC = TTC_0, ..., TTC_{n-1}
    B_TTC = [Real(f'TTC_{i}{suffix}') for i in range(N)]
    
    # Crossing states B_cs = cs_0, ..., cs_{n-1}
    B_cs = [Int(f'cs_{i}{suffix}') for i in range(N)]
    
    # Staleness timers
    s_d = Real(f's_d{suffix}')  # Detection staleness
    s_c = Real(f's_c{suffix}')  # Crossing staleness
    t = Real(f't{suffix}')      # Time since last sensor value
    
    return B_C, B_TTC, B_cs, s_d, s_c, t

# ============================================================================
# INITIAL STATES (Init)
# ============================================================================
def initial_state(q, B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    Init = {(Normal, X_0)} where X_0 = {C_i = 0, TTC_i = NO_TTC, cs_i = 0, 
                                         s_d = 0, s_c = 0, s_u = 0}
    """
    constraints = [q == Normal]
    
    for i in range(N):
        constraints.append(B_C[i] == 0)
        constraints.append(B_TTC[i] == NO_TTC)
        constraints.append(B_cs[i] == 0)
    
    constraints.extend([
        s_d == 300,
        s_c == 300,
        t == 0
    ])
    
    return And(constraints)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def P(condition):
    """
    P(cond) = 1 if cond is true, 0 otherwise
    """
    return If(condition, 1, 0)

def detected(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    detected(X) = sum_{i=0}^{n-1} P(C_i >= TH_C) >= ceil(RT_H / (2 * CAM_FREQ))
    """
    limit = int(min(len(B_C), RT_WINDOW_FRAMES))    
    count = Sum([P(B_C[i] >= TH_C) for i in range(limit)])
    return count >= RT_HALF_FRAMES

def valid_d(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    valid_d(X) = detected(X) ∨ s_d < TH_D_stale
    """
    return s_d < TH_D_STALE# Or(detected(B_C, B_TTC, B_cs, s_d, s_c, t), s_d < TH_D_STALE)

def s_dist(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    s_dist(X) = sum_{i=0}^{k} P(TTC_i > TH_TTC_s) >= ceil(k * CONSENSUS)
    where k = ceil(n * S_DISTANCE_CONSENSUS)
    """
    k = math.ceil(N * S_DISTANCE_CONSENSUS)
    count = Sum([P(B_TTC[i] > TH_TTC_S) for i in range(k)])
    threshold = math.ceil(k * CONSENSUS)
    return count >= threshold

def s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    s_r_dist(X) = sum_{i=0}^{k} P(TH_TTC_r < TTC_i <= TH_TTC_s) >= ceil(k * CONSENSUS)
    where k = ceil(n * SR_DISTANCE_CONSENSUS)
    """
    k = math.ceil(N * SR_DISTANCE_CONSENSUS)
    count = Sum([P(B_TTC[i] <= TH_TTC_S) for i in range(k)])
    threshold = math.ceil(k * CONSENSUS)
    return count >= threshold

def r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    r_c_dist(X) = sum_{i=0}^{k} P(TH_TTC_c < TTC_i <= TH_TTC_r) >= ceil(k * CONSENSUS)
    where k = ceil(n * RC_DISTANCE_CONSENSUS)
    """
    k = math.ceil(N * RC_DISTANCE_CONSENSUS)
    count = Sum([P(B_TTC[i] <= TH_TTC_R) for i in range(k)])
    threshold = math.ceil(k * CONSENSUS)
    return count >= threshold

def c_dist(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    c_dist(X) = sum_{i=0}^{k} P(TTC_i <= TH_TTC_c) >= ceil(k * CONSENSUS)
    where k = ceil(n * C_DISTANCE_CONSENSUS)
    """
    k = math.ceil(N * C_DISTANCE_CONSENSUS)
    count = Sum([P(B_TTC[i] <= TH_TTC_C) for i in range(k)])
    return count >= k

def crossing(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    crossing(X) = sum_{i=0}^{n-1} P(cs_i = 1) >= ceil(RT_H / (2 * CAM_FREQ))
    """
    limit = int(min(len(B_cs), RT_WINDOW_FRAMES))
    count = Sum([P(B_cs[i] == 1) for i in range(limit)])
    return count >= RT_HALF_FRAMES

def valid_c(B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    valid_c(X) = crossing(X) ∨ s_c < TH_C_stale
    """
    return s_c < TH_C_STALE #Or(crossing(B_C, B_TTC, B_cs, s_d, s_c, t), s_c < TH_C_STALE)

def uncertain(B_C, B_TTC, B_cs, s_d, s_c, t):
    return And(
        Not(s_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        Not(s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        Not(r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        Not(c_dist(B_C, B_TTC, B_cs, s_d, s_c, t))
    )
# ============================================================================
# INVARIANTS (Inv)
# ============================================================================

def invariant(q, B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    Invariant definition by design:
    - Invariants partition the continuous space by TTC-based safety level.
    - Transitions emerge when an invariant ceases to hold.
    """

    Inv_normal = And(
        Or(
            Not(valid_d(B_C, B_TTC, B_cs, s_d, s_c, t)),
            # And(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t), Or(s_dist(B_C, B_TTC, B_cs, s_d, s_c, t), uncertain(B_C, B_TTC, B_cs, s_d, s_c, t))),
            Not(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t)),
            s_dist(B_C, B_TTC, B_cs, s_d, s_c, t),
            uncertain(B_C, B_TTC, B_cs, s_d, s_c, t)
        ),
        t < CAMERA_FREQ
    )

    # Inv_safe_warning = And(
    #     valid_d(B_C, B_TTC, B_cs, s_d, s_c, t),
    #     Or(Not(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t)),uncertain(B_C, B_TTC, B_cs, s_d, s_c, t)),
    #     t < CAMERA_FREQ
    # )

    Inv_throttling = And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        Or(And(s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, t), Not(r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t)), Not(c_dist(B_C, B_TTC, B_cs, s_d, s_c, t))), uncertain(B_C, B_TTC, B_cs, s_d, s_c, t)),
        t < CAMERA_FREQ
    )

    Inv_soft_braking = And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        Or(And(r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t), Not(c_dist(B_C, B_TTC, B_cs, s_d, s_c, t))), uncertain(B_C, B_TTC, B_cs, s_d, s_c, t)),
        t < CAMERA_FREQ
    )

    Inv_emergency = And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        t < CAMERA_FREQ
    )

    return If(
        q == Normal, Inv_normal,
        # If(q == SafeWarning, Inv_safe_warning,
        If(q == Throttling, Inv_throttling,
        If(q == SoftBraking, Inv_soft_braking,
        If(q == EmergencyBraking, Inv_emergency, False)))#)
    )


# ============================================================================
# GUARDS (G: E → 2^X)
# ============================================================================

# Common guard conditions
def G_sense(B_C, B_TTC, B_cs, s_d, s_c, t):
    """Guard for sensing transitions (t == CAM_FREQ)"""
    return t == CAMERA_FREQ

def G_to_normal(B_C, B_TTC, B_cs, s_d, s_c, t):
    """Guard for transitions to Normal"""
    return And(
        Or(
            Not(valid_d(B_C, B_TTC, B_cs, s_d, s_c, t)),
            Not(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t)),
            s_dist(B_C, B_TTC, B_cs, s_d, s_c, t)
        ),
        t < CAMERA_FREQ
    )


# def G_to_safe_warning(B_C, B_TTC, B_cs, s_d, s_c, t):
#     """Guard for transitions to SafeWarning"""
#     return And(
#         valid_d(B_C, B_TTC, B_cs, s_d, s_c, t),
#         Not(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t)),
#         t < CAMERA_FREQ
#     )

def G_to_throttling(B_C, B_TTC, B_cs, s_d, s_c, t):
    return And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        s_r_dist(B_C, B_TTC, B_cs, s_d, s_c, t),
        Not(r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        Not(c_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        t < CAMERA_FREQ
    )

def G_to_soft_braking(B_C, B_TTC, B_cs, s_d, s_c, t):
    return And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        r_c_dist(B_C, B_TTC, B_cs, s_d, s_c, t),
        Not(c_dist(B_C, B_TTC, B_cs, s_d, s_c, t)),
        t < CAMERA_FREQ
    )

def G_to_emergency_braking(B_C, B_TTC, B_cs, s_d, s_c, t):
    return And(
        valid_c(B_C, B_TTC, B_cs, s_d, s_c, t),
        c_dist(B_C, B_TTC, B_cs, s_d, s_c, t),
        t < CAMERA_FREQ
    )


def G_from_emergency(B_C, B_TTC, B_cs, s_d, s_c, t):
    """Guard for transitions from EmergencyBraking"""
    return And(Not(valid_c(B_C, B_TTC, B_cs, s_d, s_c, t)),t < CAMERA_FREQ)


# ============================================================================
# RESETS (R: E × X → 2^X)
# ============================================================================

def sense(B_C, B_TTC, B_cs, s_d, s_c, t,
          B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next,
          C_new, TTC_new, cs_new):
    """
    sense(X) shifts buffer and adds new sensor values:
        t' = 0, C_i' = C_{i-1}, TTC_i' = TTC_{i-1}, cs_i' = cs_{i-1}
        C_0' = C_new, TTC_0' = TTC_new, cs_0' = cs_new
    """
    constraints = [t_next == 0]
    
    # Shift buffer: element i gets value from element i-1
    for i in range(1, N):
        constraints.append(B_C_next[i] == B_C[i-1])
        constraints.append(B_TTC_next[i] == B_TTC[i-1])
        constraints.append(B_cs_next[i] == B_cs[i-1])
    
    # New sensor values at position 0
    constraints.append(B_C_next[0] == C_new)
    constraints.append(B_TTC_next[0] == TTC_new)
    constraints.append(B_cs_next[0] == cs_new)
    
    # Sensor constraints
    constraints.append(And(C_new >= 0, C_new <= 1))
    constraints.append(Or(
        TTC_new == NO_TTC,
        TTC_new > 0
    ))
    constraints.append(Or(cs_new == 0, cs_new == 1))
    constraints.append(Implies(C_new < TH_C, And(TTC_new == NO_TTC, cs_new == 0)))
    
    # Staleness timers remain unchanged
    constraints.extend([
        s_d_next == s_d,
        s_c_next == s_c
    ])
    return And(constraints)


def reset_timers(B_C, B_TTC, B_cs, s_d, s_c, t, B_C_next, B_TTC_next, B_cs_next, s_d_next, s_c_next, t_next):
    """
    reset(X) resets staleness timers based on current state:
        s_d' = 0 if detected(X) else s_d
        s_c' = 0 if crossing(X) else s_c
    """
    constraints = []
    
    # Buffers remain unchanged
    for i in range(N):
        constraints.append(B_C_next[i] == B_C[i])
        constraints.append(B_TTC_next[i] == B_TTC[i])
        constraints.append(B_cs_next[i] == B_cs[i])
    constraints.append(t_next == CAMERA_FREQ)
    constraint.append(If(detected(B_C, B_TTC, B_cs, s_d, s_c, t), s_d_next == 0, s_d_next == s_d))
    constraint.append(If(crossing(B_C, B_TTC, B_cs, s_d, s_c, t), s_c_next == 0, s_c_next == s_c))

    return And(constraints)

# ============================================================================
# TRANSITION RELATION
# ============================================================================

def transition(q, q_next, B_C, B_TTC, B_cs, s_d, s_c, t):
    """
    Encodes all edges E and their guards G with reset R
    """
    # Encode all edges with their guards
    transition_cases = Or(
        # From Normal (e1-e5)
        And(q == Normal, q_next == Normal, G_sense(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == Normal, q_next == SafeWarning, G_to_safe_warning(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Normal, q_next == Throttling, G_to_throttling(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Normal, q_next == SoftBraking, G_to_soft_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Normal, q_next == EmergencyBraking, G_to_emergency_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        
        # From SafeWarning (e6-e10)
        # And(q == SafeWarning, q_next == Normal, G_to_normal(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == SafeWarning, q_next == SafeWarning, G_sense(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == SafeWarning, q_next == Throttling, G_to_throttling(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == SafeWarning, q_next == SoftBraking, G_to_soft_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == SafeWarning, q_next == EmergencyBraking, G_to_emergency_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        
        # From Throttling (e11-e15)
        And(q == Throttling, q_next == Normal, G_to_normal(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == Throttling, q_next == SafeWarning, G_to_safe_warning(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Throttling, q_next == Throttling, G_sense(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Throttling, q_next == SoftBraking, G_to_soft_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == Throttling, q_next == EmergencyBraking, G_to_emergency_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        
        # From SoftBraking (e16-e20)
        And(q == SoftBraking, q_next == Normal, G_to_normal(B_C, B_TTC, B_cs, s_d, s_c, t)),
        # And(q == SoftBraking, q_next == SafeWarning, G_to_safe_warning(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == SoftBraking, q_next == Throttling, G_to_throttling(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == SoftBraking, q_next == SoftBraking, G_sense(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == SoftBraking, q_next == EmergencyBraking, G_to_emergency_braking(B_C, B_TTC, B_cs, s_d, s_c, t)),
        
        # From EmergencyBraking (e21-e22)
        And(q == EmergencyBraking, q_next == Normal, G_from_emergency(B_C, B_TTC, B_cs, s_d, s_c, t)),
        And(q == EmergencyBraking, q_next == EmergencyBraking, G_sense(B_C, B_TTC, B_cs, s_d, s_c, t))
    )
    
    return transition_cases