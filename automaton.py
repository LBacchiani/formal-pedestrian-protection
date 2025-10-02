
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import math


class State(Enum):
    """Automaton states for pedestrian protection"""
    NORMAL = auto()
    SAFE_WARNING = auto()
    RISKY_SLOWDOWN = auto()
    CRITICAL_SLOWDOWN = auto()
    SOFT_BRAKING = auto()
    EMERGENCY_BRAKING = auto()


class Action(Enum):
    """Actions corresponding to each state"""
    NONE = auto()
    WARNING = auto()
    THROTTLE_REDUCE = auto()
    GENTLE_BRAKE = auto()
    STRONG_BRAKE = auto()
    EMERGENCY_BRAKE = auto()


@dataclass
class StateVector:
    """Sensor readings at a given time step"""
    # Camera data
    C_cam: float  # Camera confidence score
    TTC_c: float  # Time-to-collision from camera
    T_cam_detect: float  # Time camera has been detecting
    T_cam_stale: float  # Time since last camera update
    
    # Fallback sensor (lidar/radar) data
    S_fb: bool  # Fallback sensor detection flag
    TTC_fb: float  # Time-to-collision from fallback
    T_fb_detect: float  # Time fallback has been detecting
    T_fb_stale: float  # Time since last fallback update


@dataclass
class Thresholds:
    """System thresholds and parameters"""
    TH_C_c: float = 0.7  # Camera confidence threshold
    TH_TTC_s: float = 2.5  # Safe TTC threshold
    TH_TTC_r: float = 1.5  # Risky TTC threshold
    TH_TTC_c: float = 0.8  # Critical TTC threshold
    TH_T_detect: float = 0.15  # Temporal consistency threshold
    TH_T_stale: float = 0.2  # Staleness threshold (max acceptable age)
    
    # Epsilon values for hysteresis
    E_s: float = 0.2  # Safe epsilon
    E_r: float = 0.2  # Risky epsilon
    E_c: float = 0.1  # Critical epsilon
    
    def __post_init__(self):
        """Validate threshold ordering to prevent overlaps"""
        assert self.TH_TTC_c + self.E_c < self.TH_TTC_r - self.E_r, \
            "Critical and risky ranges overlap"
        assert self.TH_TTC_r + self.E_r < self.TH_TTC_s - self.E_s, \
            "Risky and safe ranges overlap"
        assert self.TH_T_stale > 0, "Staleness threshold must be positive"
        assert self.TH_T_detect < self.TH_T_stale, \
            "Detection time should be less than staleness threshold"


class PedestrianProtectionAutomaton:
    """
    Finite State Automaton for pedestrian protection with dual sensor fusion
    and staleness detection.
    
    Key invariants:
    - Sensors must be fresh (not stale) to be considered available
    - Invasive braking requires both sensors available with temporal consistency
    - State transitions are deterministic based on sensor inputs
    """
    
    def __init__(self, thresholds: Optional[Thresholds] = None):
        self.th = thresholds or Thresholds()
        self.state = State.NORMAL
        self.previous_state = State.NORMAL
        
    def _avail_cam(self, data: StateVector) -> bool:
        """
        Camera availability: confidence above threshold, temporally stable, and fresh.
        
        This is a critical safety predicate - stale data is treated as unavailable.
        """
        return (data.C_cam >= self.th.TH_C_c and 
                data.T_cam_detect > self.th.TH_T_detect and 
                data.T_cam_stale < self.th.TH_T_stale)
    
    def _avail_fb(self, data: StateVector) -> bool:
        """
        Fallback sensor availability: detecting, temporally stable, and fresh.
        
        This is a critical safety predicate - stale data is treated as unavailable.
        """
        return (data.S_fb and 
                data.T_fb_detect > self.th.TH_T_detect and 
                data.T_fb_stale < self.th.TH_T_stale)
    
    def _no_detection(self, data: StateVector) -> bool:
        """No available detection from either sensor"""
        return not self._avail_cam(data) and not self._avail_fb(data)
    
    def _normal_cam_only(self, data: StateVector) -> bool:
        """Camera only available, safe distance"""
        return (self._avail_cam(data) and 
                not self._avail_fb(data) and 
                data.TTC_c > self.th.TH_TTC_s + self.th.E_s)
    
    def _normal_fb_only(self, data: StateVector) -> bool:
        """Fallback only available, safe distance"""
        return (not self._avail_cam(data) and 
                self._avail_fb(data) and 
                data.TTC_fb > self.th.TH_TTC_s + self.th.E_s)
    
    def _normal_both(self, data: StateVector) -> bool:
        """Both sensors available, safe distance"""
        return (self._avail_cam(data) and 
                self._avail_fb(data) and 
                data.TTC_c > self.th.TH_TTC_s and 
                data.TTC_fb > self.th.TH_TTC_s + self.th.E_s)
    
    def _safe_warning_cam_only(self, data: StateVector) -> bool:
        """Camera only available, warning range"""
        return (self._avail_cam(data) and 
                not self._avail_fb(data) and 
                self.th.TH_TTC_r - self.th.E_r <= data.TTC_c <= self.th.TH_TTC_s + self.th.E_s)
    
    def _safe_warning_fb_only(self, data: StateVector) -> bool:
        """Fallback only available, warning range"""
        return (not self._avail_cam(data) and 
                self._avail_fb(data) and 
                self.th.TH_TTC_r - self.th.E_r <= data.TTC_fb <= self.th.TH_TTC_s + self.th.E_s)
    
    def _safe_warning_both(self, data: StateVector) -> bool:
        """Both sensors available, warning range"""
        min_ttc = min(data.TTC_c, data.TTC_fb)
        return (self._avail_cam(data) and 
                self._avail_fb(data) and 
                self.th.TH_TTC_r - self.th.E_r <= min_ttc <= self.th.TH_TTC_s + self.th.E_s)
    
    def _risky_slowdown_cam_only(self, data: StateVector) -> bool:
        """Camera only available, risky range"""
        return (self._avail_cam(data) and 
                not self._avail_fb(data) and 
                self.th.TH_TTC_c - self.th.E_c <= data.TTC_c < self.th.TH_TTC_r + self.th.E_r)
    
    def _risky_slowdown_fb_only(self, data: StateVector) -> bool:
        """Fallback only available, risky range"""
        return (not self._avail_cam(data) and 
                self._avail_fb(data) and 
                self.th.TH_TTC_c - self.th.E_c <= data.TTC_fb < self.th.TH_TTC_r + self.th.E_r)
    
    def _critical_slowdown_cam_only(self, data: StateVector) -> bool:
        """Camera only available, critical range"""
        return (self._avail_cam(data) and 
                not self._avail_fb(data) and 
                data.TTC_c < self.th.TH_TTC_c + self.th.E_c)
    
    def _critical_slowdown_fb_only(self, data: StateVector) -> bool:
        """Fallback only available, critical range"""
        return (not self._avail_cam(data) and 
                self._avail_fb(data) and 
                data.TTC_fb < self.th.TH_TTC_c + self.th.E_c)
    
    def _soft_braking(self, data: StateVector) -> bool:
        """Both sensors available and agree, risky range"""
        min_ttc = min(data.TTC_c, data.TTC_fb)
        return (self._avail_cam(data) and 
                self._avail_fb(data) and 
                self.th.TH_TTC_c - self.th.E_c <= min_ttc < self.th.TH_TTC_r + self.th.E_r)
    
    def _emergency_braking(self, data: StateVector) -> bool:
        """Both sensors available and agree, critical range"""
        min_ttc = min(data.TTC_c, data.TTC_fb)
        return (self._avail_cam(data) and 
                self._avail_fb(data) and 
                min_ttc < self.th.TH_TTC_c + self.th.E_c)
    
    def _inv_normal(self, data: StateVector) -> bool:
        """Invariant for Normal state"""
        return (self._no_detection(data) or 
                self._normal_cam_only(data) or 
                self._normal_fb_only(data) or 
                self._normal_both(data))
    
    def _inv_safe_warning(self, data: StateVector) -> bool:
        """Invariant for SafeWarning state"""
        return (self._safe_warning_cam_only(data) or 
                self._safe_warning_fb_only(data) or 
                self._safe_warning_both(data))
    
    def _inv_risky_slowdown(self, data: StateVector) -> bool:
        """Invariant for RiskySlowdown state"""
        return (self._risky_slowdown_cam_only(data) or 
                self._risky_slowdown_fb_only(data))
    
    def _inv_critical_slowdown(self, data: StateVector) -> bool:
        """Invariant for CriticalSlowdown state"""
        return (self._critical_slowdown_cam_only(data) or 
                self._critical_slowdown_fb_only(data))
    
    def _inv_soft_braking(self, data: StateVector) -> bool:
        """Invariant for SoftBraking state"""
        return self._soft_braking(data)
    
    def _inv_emergency_braking(self, data: StateVector) -> bool:
        """Invariant for EmergencyBraking state"""
        return self._emergency_braking(data)
    
    def compute_next_state(self, data: StateVector) -> State:
        """
        Compute next state based on current state and sensor data.
        
        This method is deterministic and implements the complete transition table.
        """
        # Check invariants in priority order (most critical first)
        if self._inv_emergency_braking(data):
            return State.EMERGENCY_BRAKING
        elif self._inv_soft_braking(data):
            return State.SOFT_BRAKING
        elif self._inv_critical_slowdown(data):
            return State.CRITICAL_SLOWDOWN
        elif self._inv_risky_slowdown(data):
            return State.RISKY_SLOWDOWN
        elif self._inv_safe_warning(data):
            return State.SAFE_WARNING
        elif self._inv_normal(data):
            return State.NORMAL
        else:
            # Should never reach here if invariants are complete
            # Default to most conservative safe state
            return State.CRITICAL_SLOWDOWN
    
    def step(self, data: StateVector) -> tuple[State, Action]:
        """
        Execute one step of the automaton.
        
        Returns: (new_state, action)
        """
        self.previous_state = self.state
        self.state = self.compute_next_state(data)
        action = self.get_action(self.state)
        return self.state, action
    
    @staticmethod
    def get_action(state: State) -> Action:
        """Map state to action"""
        action_map = {
            State.NORMAL: Action.NONE,
            State.SAFE_WARNING: Action.WARNING,
            State.RISKY_SLOWDOWN: Action.THROTTLE_REDUCE,
            State.CRITICAL_SLOWDOWN: Action.GENTLE_BRAKE,
            State.SOFT_BRAKING: Action.STRONG_BRAKE,
            State.EMERGENCY_BRAKING: Action.EMERGENCY_BRAKE,
        }
        return action_map[state]


# ============================================================================
# CROSSHAIR VERIFICATION PROPERTIES
# ============================================================================

def make_valid_sensor_data(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> StateVector:
    """Helper to create valid sensor data with preconditions for CrossHair."""
    return StateVector(
        C_cam=C_cam,
        TTC_c=TTC_c,
        T_cam_detect=T_cam_detect,
        T_cam_stale=T_cam_stale,
        S_fb=S_fb,
        TTC_fb=TTC_fb,
        T_fb_detect=T_fb_detect,
        T_fb_stale=T_fb_stale
    )


# ============================================================================
# ORIGINAL PROPERTIES (Updated for Staleness)
# ============================================================================

def check_P1_no_unintended_braking(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P1: No Unintended Braking
    
    Invasive braking only occurs when both sensors are available
    (fresh, confident, temporally stable).
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    state, _ = automaton.step(data)
    
    if state in [State.EMERGENCY_BRAKING, State.SOFT_BRAKING]:
        return automaton._avail_cam(data) and automaton._avail_fb(data)
    return True


def check_P3_emergency_braking_necessity(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P3: Emergency Braking Necessity
    
    When both sensors are available and confirm imminent collision,
    emergency braking must activate.
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    th = automaton.th
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    
    # Check precondition for emergency with availability
    min_ttc = min(TTC_c, TTC_fb)
    emergency_conditions = (
        automaton._avail_cam(data) and 
        automaton._avail_fb(data) and 
        min_ttc < th.TH_TTC_c + th.E_c
    )
    
    if emergency_conditions:
        state, _ = automaton.step(data)
        return state == State.EMERGENCY_BRAKING
    return True


def check_P4_no_spurious_emergency_from_single_sensor(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P4: No Spurious Emergency from Single Sensor
    
    Emergency braking never occurs with only one sensor available.
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    state, _ = automaton.step(data)
    
    if state == State.EMERGENCY_BRAKING:
        # Both sensors must be available
        return automaton._avail_cam(data) and automaton._avail_fb(data)
    return True


def check_P8_single_active_state(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P8: Single Active State
    
    At least one state invariant is true at any time (completeness).
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    
    # At least one invariant must be satisfied (completeness)
    inv_count = sum([
        automaton._inv_normal(data),
        automaton._inv_safe_warning(data),
        automaton._inv_risky_slowdown(data),
        automaton._inv_critical_slowdown(data),
        automaton._inv_soft_braking(data),
        automaton._inv_emergency_braking(data),
    ])
    
    return inv_count >= 1


def check_P10_sensor_agreement_for_braking(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P10: Sensor Agreement for Crossing Detection
    
    Soft and emergency braking only when both sensors are available.
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    state, _ = automaton.step(data)
    
    if state in [State.SOFT_BRAKING, State.EMERGENCY_BRAKING]:
        return automaton._avail_cam(data) and automaton._avail_fb(data)
    return True


# ============================================================================
# NEW STALENESS-SPECIFIC PROPERTIES
# ============================================================================

def check_P_stale1_no_braking_with_stale_data(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P_STALE1: No Braking With Stale Data
    
    Invasive braking never occurs when either sensor has stale data.
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    th = automaton.th
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    state, _ = automaton.step(data)
    
    if state in [State.SOFT_BRAKING, State.EMERGENCY_BRAKING]:
        # Both sensors must have fresh data
        return (data.T_cam_stale < th.TH_T_stale and 
                data.T_fb_stale < th.TH_T_stale)
    return True


def check_P_stale2_stale_sensor_treated_as_unavailable(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P_STALE2: Stale Sensor Treated as Unavailable
    
    If a sensor has stale data, it's treated as unavailable even if
    it meets other criteria (confidence, detection time).
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    th = automaton.th
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    
    # If camera data is stale, camera should not be available
    if data.T_cam_stale >= th.TH_T_stale:
        if not automaton._avail_cam(data):
            return True  # Correctly treated as unavailable
        else:
            return False  # Bug: stale sensor treated as available
    
    # If fallback data is stale, fallback should not be available
    if data.T_fb_stale >= th.TH_T_stale:
        if not automaton._avail_fb(data):
            return True  # Correctly treated as unavailable
        else:
            return False  # Bug: stale sensor treated as available
    
    return True


def check_P_stale3_fresh_data_required_for_availability(
    C_cam: float,
    TTC_c: float,
    T_cam_detect: float,
    T_cam_stale: float,
    S_fb: bool,
    TTC_fb: float,
    T_fb_detect: float,
    T_fb_stale: float
) -> bool:
    """
    P_STALE3: Fresh Data Required for Availability
    
    A sensor can only be available if its data is fresh.
    This is the converse of P_STALE2.
    
    pre: C_cam >= 0.0 and C_cam <= 1.0
    pre: TTC_c >= 0.0 and TTC_c <= 10.0
    pre: T_cam_detect >= 0.0 and T_cam_detect <= 5.0
    pre: T_cam_stale >= 0.0 and T_cam_stale <= 1.0
    pre: TTC_fb >= 0.0 and TTC_fb <= 10.0
    pre: T_fb_detect >= 0.0 and T_fb_detect <= 5.0
    pre: T_fb_stale >= 0.0 and T_fb_stale <= 1.0
    post: _
    """
    automaton = PedestrianProtectionAutomaton()
    th = automaton.th
    data = make_valid_sensor_data(C_cam, TTC_c, T_cam_detect, T_cam_stale,
                                   S_fb, TTC_fb, T_fb_detect, T_fb_stale)
    
    # If camera is available, its data must be fresh
    if automaton._avail_cam(data):
        if data.T_cam_stale >= th.TH_T_stale:
            return False  # Bug: available despite stale data
    
    # If fallback is available, its data must be fresh
    if automaton._avail_fb(data):
        if data.T_fb_stale >= th.TH_T_stale:
            return False  # Bug: available despite stale data
    
    return True


# def check_P_stale4_graceful_degradation_on_staleness(
#     C_cam: float,
#     TTC_c: float,
#     T_cam_detect: float,
#     T_cam_stale: float,
#     S_fb: bool,
#     TTC_fb: float,
#     T_fb_detect: float,
#     T_fb_stale: float
# ) -> bool:
#     """
#     P_STALE4: Graceful Degradation on Staleness
    
#     If one sensor becomes stale during braking, the system should
#     not immediately de-escalate to a less critical state.
#     (Conservative safety behavior)
    
#     pre: C_cam >= 0.0 and C_cam <= 1.0
#     pre: TTC_c >= 0.0 and TTC_c <= 10.0
#     pre: T_cam_detect >= 0.0