from enum import Enum
from typing import Optional
from collections import deque
import time
import math

# ============================================================================
# GLOBAL THRESHOLDS
# ============================================================================

N = 10  # Buffer size
RT_H = 700  # Reaction time (ms)
CAMERA_FREQ = 100  # Camera frequency (ms)
RT_WINDOW_FRAMES = RT_H / 100
RT_HALF_FRAMES = max(1, math.ceil(RT_WINDOW_FRAMES / 2))  # Half reaction window (minimum 1)

# Consensus thresholds (percentage of buffer that must agree)
S_DISTANCE_CONSENSUS = 0.8
SR_DISTANCE_CONSENSUS = 0.6
RC_DISTANCE_CONSENSUS = 0.4
C_DISTANCE_CONSENSUS = 0.2
CONSENSUS = 0.8

# ============================================================================
# SAFETY THRESHOLDS (Z3 - can be verified parametrically)
# ============================================================================

TH_C = 0.4
TH_D_STALE = 300
TH_C_STALE = 300

# Time-to-collision thresholds (seconds)
TH_TTC_S = 3500   # Safe
TH_TTC_R = 2000   # Risky
TH_TTC_C = 1000   # Critical

# Staleness upper bounds
MAX_UNCERTAIN = 300 # Handover timeout (ms)

# Sensor parameters
NO_TTC = TH_TTC_S + 1


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class State(Enum):
    """States of the pedestrian protection automaton"""
    NORMAL = "Normal"
    SAFE_WARNING = "SafeWarning"
    THROTTLING = "Throttling"
    #CRITICAL_SLOWDOWN = "CriticalSlowdown"
    SOFT_BRAKING = "SoftBraking"
    EMERGENCY_BRAKING = "EmergencyBraking"


class Action(Enum):
    """Actions that can be emitted by the automaton"""
    # BRAKE = "brake"
    # STOP = "stop"
    # THROTTLE_ACCELERATION = "throttle_acceleration"
    # ALERTING_DRIVER = "alerting_driver"
    # REMOVE_ALERT = "remove_alert"
    # STOP_THROTTLING = "stop_throttling"
    # STOP_BRAKING = "stop_braking"
    # BRAKE_TO_THROTTLE = "brake_to_throttle"
    # NONE = "_"  # No action

    BRAKE = "brake"
    STOP = "emergency_brake"
    THROTTLE_ACCELERATION = "mild_brake" #mild brake con 0 di frenata e 0 di gas 
    ALERTING_DRIVER = "warning"
    REMOVE_ALERT = "normal"
    STOP_THROTTLING = "normal" #  
    STOP_BRAKING = "normal"
    BRAKE_TO_THROTTLE = "mild_brake" #mild brake con 0 di frenata e 0 di gas
    NONE = "_"  # No action continui a fare l'azione che stavi facendo 


class NoTransitionAvailableError(Exception):
    """Raised when no valid transition exists from the current state"""
    pass


# ============================================================================
# PEDESTRIAN PROTECTION AUTOMATON
# ============================================================================

class PedestrianProtectionAutomaton:
    """
    Hybrid automaton for pedestrian protection system.
    
    Manages state transitions based on:
    - Detection confidence (B_C)
    - Time-to-collision (B_TTC)
    - Crossing status (B_cross)
    - Staleness timers (s_d, s_c)
    """
    
    def __init__(self):
        """Initialize the automaton with default state and empty buffers"""
        # Current state
        self.state = State.NORMAL
        
        # Buffers (most recent element at index 0)
        self.B_C: deque = deque(maxlen=N)  # Confidence buffer [0,1]
        self.B_TTC: deque = deque(maxlen=N)  # Time-to-collision buffer (s)
        self.B_cross: deque = deque(maxlen=N)  # Crossing status buffer {0,1}
        
        # Staleness timers (ms)
        self.s_d = TH_D_STALE  # Detection staleness
        self.s_c = TH_C_STALE  # Crossing staleness
        
        # Last step call timestamp (seconds)
        self.last_step_call: Optional[float] = None
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _detected(self) -> bool:
        """Check if pedestrian is detected based on confidence buffer"""
        limit = min(N, RT_HALF_FRAMES)
        if len(self.B_C) < limit:
            return False
        count = sum(1 for i in range(limit) if self.B_C[i] >= TH_C)
        return count >= RT_HALF_FRAMES
    
    def _crossing(self) -> bool:
        """Check if pedestrian is crossing based on crossing buffer"""
        limit = min(N, RT_HALF_FRAMES)
        if len(self.B_cross) < limit: 
            return False
        count = sum(self.B_cross[i] for i in range(limit))
        return count >= RT_HALF_FRAMES
    
    def _valid_d(self) -> bool:
        """Check if detection data is valid (fresh or recently detected)"""
        return self._detected() or self.s_d < TH_D_STALE
    
    def _valid_c(self) -> bool:
        """Check if crossing data is valid (fresh or recently crossing)"""
        return self._crossing() or self.s_c < TH_C_STALE
    
    def _s_distance(self) -> bool:
        """Check if distance is safe (Z3-matching implementation)"""
        k = math.ceil(S_DISTANCE_CONSENSUS * N)
        if len(self.B_TTC) < k:
            return False
        count = sum(1 for i in range(k) if self.B_TTC[i] > TH_TTC_S)
        threshold = math.ceil(CONSENSUS * k)
        return count >= threshold
    
    def _s_r_distance(self) -> bool:
        """Check if distance is safe-to-risky (Z3-matching implementation)"""
        k = math.ceil(SR_DISTANCE_CONSENSUS * N)
        if len(self.B_TTC) < k:
            return False
        count = sum(1 for i in range(k) if self.B_TTC[i] <= TH_TTC_S)
        threshold = math.ceil(CONSENSUS * k)
        return count >= threshold
    
    def _r_c_distance(self) -> bool:
        """Check if distance is risky-to-critical"""
        k = math.ceil(RC_DISTANCE_CONSENSUS * N)
        if len(self.B_TTC) < k:
            return False
        count = sum(1 for i in range(k) if self.B_TTC[i] <= TH_TTC_R)
        threshold = math.ceil(CONSENSUS * k)
        return count >= threshold
    
    def _c_distance(self) -> bool:
        """Check if distance is critical"""
        k = math.ceil(C_DISTANCE_CONSENSUS * N)
        if len(self.B_TTC) < k:
            return False
        return all(self.B_TTC[i] <= TH_TTC_C for i in range(k))
    
    def _uncertain_distance(self) -> bool:
        """Check if distance classification is uncertain"""
        return not (self._s_distance() or self._s_r_distance() or 
                   self._r_c_distance() or self._c_distance())

    
    # ========================================================================
    # STATE TRANSITION LOGIC
    # ========================================================================
    
    def _check_transitions(self) -> tuple[State, Action]:
        """
        Check all possible transitions from current state.
        Returns the new state and action to take.
        Raises NoTransitionAvailableError if no valid transition exists.
        """
        valid_d = self._valid_d()
        valid_c = self._valid_c()
        det = self._detected()
        cross = self._crossing()
        s_dist = self._s_distance()
        sr_dist = self._s_r_distance()
        rc_dist = self._r_c_distance()
        c_dist = self._c_distance()
        unc_dist = self._uncertain_distance()

        inv_normal = not valid_d or (valid_c and (s_dist or unc_dist))
        inv_safe_warning = valid_d and (not valid_c or unc_dist)
        inv_throttling = valid_c and ((sr_dist and not rc_dist and not c_dist) or unc_dist)
        inv_soft_braking = valid_c and ((rc_dist and not c_dist) or unc_dist)
        inv_emergency_braking = valid_c

        G_to_normal = not valid_d or (valid_c and s_dist)
        G_to_safe_warning = valid_d and not valid_c
        G_to_throttling = valid_c and sr_dist and not rc_dist and not c_dist
        G_to_soft_braking = valid_c and rc_dist and not c_dist
        G_to_emergency_braking = valid_c and c_dist
        
        # Transition logic based on current state
        if self.state == State.NORMAL:
            if inv_normal: 
                return State.NORMAL, Action.NONE
            if G_to_safe_warning:
                return State.SAFE_WARNING, Action.ALERTING_DRIVER
            if G_to_throttling:
                return State.THROTTLING, Action.THROTTLE_ACCELERATION
            if G_to_soft_braking:
                return State.SOFT_BRAKING, Action.BRAKE
            if G_to_emergency_braking:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.SAFE_WARNING:
            if G_to_normal:
                return State.NORMAL, Action.NONE
            if inv_safe_warning:
                return State.SAFE_WARNING, Action.NONE
            if G_to_throttling:
                return State.THROTTLING, Action.THROTTLE_ACCELERATION
            if G_to_soft_braking:
                return State.SOFT_BRAKING, Action.BRAKE
            if G_to_emergency_braking:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.THROTTLING:
            if G_to_normal:
                return State.NORMAL, Action.STOP_THROTTLING
            if G_to_safe_warning:
                return State.SAFE_WARNING, Action.STOP_THROTTLING
            if inv_throttling:
                return State.THROTTLING, Action.NONE
            if G_to_soft_braking:
                return State.SOFT_BRAKING, Action.BRAKE
            if G_to_emergency_braking:
                return State.EMERGENCY_BRAKING, Action.STOP

        elif self.state == State.SOFT_BRAKING:
            if G_to_normal:
                return State.NORMAL, Action.STOP_BRAKING
            if G_to_safe_warning:
                return State.SAFE_WARNING, Action.STOP_BRAKING
            if G_to_throttling:
                return State.THROTTLING, Action.BRAKE_TO_THROTTLE
            if inv_soft_braking:
                return State.SOFT_BRAKING, Action.NONE
            if G_to_emergency_braking:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.EMERGENCY_BRAKING:
            if not valid_c:
                return State.NORMAL, Action.STOP_BRAKING
            if inv_emergency_braking:
                return State.EMERGENCY_BRAKING, Action.NONE
        
        # No transition available - this should not happen with proper invariants
        raise NoTransitionAvailableError(
            f"No valid transition available from state {self.state.value}. "
            f"State invariants may be violated. "
            f"valid_d={valid_d}, valid_c={valid_c}, det={det}, cross={cross}, "
            f"s_dist={s_dist}, sr_dist={sr_dist}, rc_dist={rc_dist}, c_dist={c_dist}, "
            f"unc_dist={unc_dist}"
        )
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def update_data(self, confidence: Optional[float] = None, 
                   ttc: Optional[float] = None, 
                   is_crossing: Optional[bool] = None):
        """
        Update buffers with new sensor data.
        
        Args:
            confidence: Detection confidence [0, 1]
            ttc: Time to collision (seconds)
            is_crossing: Whether pedestrian is crossing
        """
        if confidence is not None:
            self.B_C.appendleft(confidence)
        
        if ttc is not None:
            self.B_TTC.appendleft(ttc)
        
        if is_crossing is not None:
            self.B_cross.appendleft(1 if is_crossing else 0)
    
    def step(self) -> Action:
        """
        Execute one step of the automaton.
        
        Computes dt as the difference between now and the last step call.
        Updates staleness timers and checks for state transitions.
        
        Returns:
            Action to be taken (or Action.NONE if no state change)
            
        Raises:
            NoTransitionAvailableError: If no valid transition exists
        """
        # Get current time
        current_time = time.time()
        
        # Compute dt (seconds)
        if self.last_step_call is None:
            dt = 0.0  # First call, no time has passed
        else:
            dt = current_time - self.last_step_call
        
        # Update last step call time
        self.last_step_call = current_time
        
        # Update staleness timers (convert dt from seconds to ms)
        dt_ms = dt * 1000
        
        # Update detection staleness
        if self._detected():
            self.s_d = 0
        else:
            self.s_d = min(self.s_d + dt_ms, TH_D_STALE)
        
        # Update crossing staleness
        if self._crossing():
            self.s_c = 0
        else:
            self.s_c = min(self.s_c + dt_ms, TH_C_STALE)
        
        # Check for state transitions
        new_state, action = self._check_transitions()
        
        # Update state
        self.state = new_state
        
        return action
    
    def get_status(self) -> dict:
        """
        Get complete status of the automaton.
        
        Returns:
            Dictionary containing state, buffers, and timers
        """
        return {
            'state': self.state.value,
            'buffers': {
                'B_C': list(self.B_C),
                'B_TTC': list(self.B_TTC),
                'B_cross': list(self.B_cross)
            },
            'staleness': {
                's_d': self.s_d,
                's_c': self.s_c
            },
            'validity': {
                'valid_d': self._valid_d(),
                'valid_c': self._valid_c(),
                'detected': self._detected(),
                'crossing': self._crossing()
            },
            'distance': {
                's_distance': self._s_distance(),
                'sr_distance': self._s_r_distance(),
                'rc_distance': self._r_c_distance(),
                'c_distance': self._c_distance(),
                'uncertain': self._uncertain_distance()
            }
        }