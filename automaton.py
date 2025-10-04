from enum import Enum
from typing import Optional
from collections import deque
import time

# ============================================================================
# GLOBAL THRESHOLDS
# ============================================================================

# Buffer size
N = 10

# Detection thresholds
TH_C = 0.5  # Threshold for certain detection (confidence)
TH_D_STALE = 500  # Threshold for detection stale data (ms)
TH_C_STALE = 500  # Threshold for crossing stale data (ms)

# Time-to-collision thresholds (seconds)
TH_TTC_S = 5.0  # Safe TTC threshold
TH_TTC_R = 3.0  # Risky TTC threshold
TH_TTC_C = 1.5  # Critical TTC threshold

# Staleness upper bound
MAX_STALE = 2000  # Maximum staleness (ms)

DETECTION_CONSENSUS = CROSSING_CONSENSUS = 0.7
S_DISTANCE_CONSENSUS = 0.8
SR_DISTANCE_CONSENSUS = 0.6
RC_DISTANCE_CONSENSUS = 0.4
C_DISTANCE_CONSENSUS = 0.2



# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class State(Enum):
    """States of the pedestrian protection automaton"""
    NORMAL = "Normal"
    SAFE_WARNING = "SafeWarning"
    THROTTLING = "Throttling"
    CRITICAL_SLOWDOWN = "CriticalSlowdown"
    SOFT_BRAKING = "SoftBraking"
    EMERGENCY_BRAKING = "EmergencyBraking"


class Action(Enum):
    """Actions that can be emitted by the automaton"""
    BRAKE = "brake"
    STOP = "stop"
    THROTTLE_ACCELERATION = "throttle_acceleration"
    ALERTING_DRIVER = "alerting_driver"
    REMOVE_ALERT = "remove_alert"
    STOP_THROTTLING = "stop_throttling"
    STOP_BRAKING = "stop_braking"
    BRAKE_TO_THROTTLE = "brake_to_throttle"
    NONE = "_"  # No action


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
        self.s_d = 0  # Detection staleness
        self.s_c = 0  # Crossing staleness
        
        # Last step call timestamp (seconds)
        self.last_step_call: Optional[float] = None
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _detected(self) -> bool:
        """Check if pedestrian is detected based on confidence buffer"""
        if len(self.B_C) == 0:
            return False
        count = sum(1 for c in self.B_C if c > TH_C)
        return count >= DETECTION_CONSENSUS * N
    
    def _crossing(self) -> bool:
        """Check if pedestrian is crossing based on crossing buffer"""
        if len(self.B_cross) == 0:
            return False
        count = sum(self.B_cross)
        return count >= CROSSING_CONSENSUS * N
    
    def _valid_d(self) -> bool:
        """Check if detection data is valid (fresh or recently detected)"""
        return self._detected() or self.s_d < TH_D_STALE
    
    def _valid_c(self) -> bool:
        """Check if crossing data is valid (fresh or recently crossing)"""
        return self._crossing() or self.s_c < TH_C_STALE
    
    def _s_distance(self) -> bool:
        """Check if distance is safe"""
        if len(self.B_TTC) == 0:
            return True
        count = sum(1 for ttc in self.B_TTC if ttc > TH_TTC_S)
        return count >= S_DISTANCE_CONSENSUS * N
    
    def _s_r_distance(self) -> bool:
        """Check if distance is safe-to-risky"""
        if len(self.B_TTC) == 0:
            return False
        count = sum(1 for ttc in self.B_TTC if TH_TTC_R <= ttc < TH_TTC_S)
        return count >= SR_DISTANCE_CONSENSUS * N
    
    def _r_c_distance(self) -> bool:
        """Check if distance is risky-to-critical"""
        if len(self.B_TTC) == 0:
            return False
        count = sum(1 for ttc in self.B_TTC if TH_TTC_C <= ttc < TH_TTC_R)
        return count >= RC_DISTANCE_CONSENSUS * N
    
    def _c_distance(self) -> bool:
        """Check if distance is critical"""
        if len(self.B_TTC) == 0:
            return False
        count = sum(1 for ttc in self.B_TTC if ttc < TH_TTC_C)
        return count >= C_DISTANCE_CONSENSUS * N
    
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
        s_dist = self._s_distance()
        sr_dist = self._s_r_distance()
        rc_dist = self._r_c_distance()
        c_dist = self._c_distance()
        
        # Transition logic based on current state
        if self.state == State.NORMAL:
            # e1: Stay in Normal
            if not valid_d or s_dist:
                return State.NORMAL, Action.NONE
            # e2: To SafeWarning
            if valid_d and not valid_c and sr_dist:
                return State.SAFE_WARNING, Action.ALERTING_DRIVER
            # e3: To Throttling
            if valid_d and ((not valid_c and rc_dist) or (valid_c and sr_dist)):
                return State.THROTTLING, Action.THROTTLE_ACCELERATION
            # e4: To CriticalSlowdown
            if valid_d and not valid_c and c_dist:
                return State.CRITICAL_SLOWDOWN, Action.BRAKE
            # e5: To SoftBraking
            if valid_d and valid_c and rc_dist:
                return State.SOFT_BRAKING, Action.BRAKE
            # e6: To EmergencyBraking
            if valid_d and valid_c and c_dist:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.SAFE_WARNING:
            # e7: To Normal
            if not valid_d or s_dist:
                return State.NORMAL, Action.NONE
            # e8: Stay in SafeWarning
            if valid_d and not valid_c and sr_dist:
                return State.SAFE_WARNING, Action.NONE
            # e9: To Throttling
            if valid_d and ((not valid_c and rc_dist) or (valid_c and sr_dist)):
                return State.THROTTLING, Action.THROTTLE_ACCELERATION
            # e10: To CriticalSlowdown
            if valid_d and not valid_c and c_dist:
                return State.CRITICAL_SLOWDOWN, Action.BRAKE
            # e11: To SoftBraking
            if valid_d and valid_c and rc_dist:
                return State.SOFT_BRAKING, Action.BRAKE
            # e12: To EmergencyBraking
            if valid_d and valid_c and c_dist:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.THROTTLING:
            # e13: To Normal
            if not valid_d or s_dist:
                return State.NORMAL, Action.STOP_THROTTLING
            # e14: To SafeWarning
            if valid_d and not valid_c and sr_dist:
                return State.SAFE_WARNING, Action.STOP_THROTTLING
            # e15: Stay in Throttling
            if valid_d and ((not valid_c and rc_dist) or (valid_c and sr_dist)):
                return State.THROTTLING, Action.NONE
            # e16: To CriticalSlowdown
            if valid_d and not valid_c and c_dist:
                return State.CRITICAL_SLOWDOWN, Action.BRAKE
            # e17: To SoftBraking
            if valid_d and valid_c and rc_dist:
                return State.SOFT_BRAKING, Action.BRAKE
            # e18: To EmergencyBraking
            if valid_d and valid_c and c_dist:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.CRITICAL_SLOWDOWN:
            # e19: To Normal
            if not valid_d or s_dist:
                return State.NORMAL, Action.STOP_BRAKING
            # e20: To SafeWarning
            if valid_d and not valid_c and sr_dist:
                return State.SAFE_WARNING, Action.STOP_BRAKING
            # e21: To Throttling
            if (not valid_c and rc_dist) or (valid_c and sr_dist):
                return State.THROTTLING, Action.BRAKE_TO_THROTTLE
            # e22: Stay in CriticalSlowdown
            if valid_d and not valid_c and c_dist:
                return State.CRITICAL_SLOWDOWN, Action.NONE
            # e23: To SoftBraking
            if valid_d and valid_c and rc_dist:
                return State.SOFT_BRAKING, Action.BRAKE
            # e24: To EmergencyBraking
            if valid_d and valid_c and c_dist:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.SOFT_BRAKING:
            # e25: To Normal
            if not valid_d or s_dist:
                return State.NORMAL, Action.STOP_BRAKING
            # e26: To SafeWarning
            if valid_d and not valid_c and sr_dist:
                return State.SAFE_WARNING, Action.STOP_BRAKING
            # e27: Stay in SoftBraking
            if valid_d and valid_c and rc_dist:
                return State.SOFT_BRAKING, Action.NONE
            # e28: To EmergencyBraking
            if valid_d and valid_c and c_dist:
                return State.EMERGENCY_BRAKING, Action.STOP
        
        elif self.state == State.EMERGENCY_BRAKING:
            # e29: To Normal
            if not valid_d or not valid_c or s_dist:
                return State.NORMAL, Action.NONE
        
        # No transition available - this should not happen with proper invariants
        raise NoTransitionAvailableError(
            f"No valid transition available from state {self.state.value}. "
            f"State invariants may be violated. "
            f"valid_d={valid_d}, valid_c={valid_c}, "
            f"s_dist={s_dist}, sr_dist={sr_dist}, rc_dist={rc_dist}, c_dist={c_dist}"
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
            self.s_d = min(self.s_d + dt_ms, MAX_STALE)
        
        # Update crossing staleness
        if self._crossing():
            self.s_c = 0
        else:
            self.s_c = min(self.s_c + dt_ms, MAX_STALE)
        
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
            }
        }