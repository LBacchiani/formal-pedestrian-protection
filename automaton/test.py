import math
import matplotlib.pyplot as plt
from automaton import *
import time

# ==============================================================
# Utility functions
# ==============================================================

def consensus_frames(ratio: float) -> int:
    return math.ceil(ratio * N)

def consensus_threshold(ratio: float) -> int:
    k = consensus_frames(ratio)
    return math.ceil(CONSENSUS * k)

def run_sequence(auto, ttc_seq, conf=1.0, cross=True):
    states = []
    buffers = []
    for ttc in ttc_seq:
        auto.update_data(confidence=conf, ttc=ttc, is_crossing=cross)
        auto.step()
        states.append(auto.state)
        buffers.append({
            'B_TTC': list(auto.B_TTC),
            'B_C': list(auto.B_C),
            'B_cross': list(auto.B_cross)
        })
        time.sleep(CAMERA_FREQ / 1000.0)
    return states, buffers

def report(test_name, condition, ttc_seq=None, states=None, buffers=None):
    if condition:
        print(f"[PASS] {test_name}")
    else:
        print(f"[FAIL] {test_name}")
        if ttc_seq is not None and states is not None:
            plot_failure(test_name, ttc_seq, states)
        if buffers is not None:
            print("\nBuffers snapshot at last step:")
            last_buf = buffers[-1]
            for k, v in last_buf.items():
                print(f"{k}: {v}")

def plot_failure(name, ttc_seq, states):
    state_map = {
        State.NORMAL: 0,
        State.SAFE_WARNING: 1,
        State.THROTTLING: 2,
        State.SOFT_BRAKING: 3,
        State.EMERGENCY_BRAKING: 4
    }
    state_colors = {
        State.NORMAL: 'green',
        State.SAFE_WARNING: 'yellow',
        State.THROTTLING: 'orange',
        State.SOFT_BRAKING: 'red',
        State.EMERGENCY_BRAKING: 'darkred'
    }
    steps = list(range(len(ttc_seq)))
    state_values = [state_map[s] for s in states]
    colors = [state_colors[s] for s in states]

    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(steps, ttc_seq, 'b-o', label="TTC")
    plt.axhline(y=TH_TTC_S, color='g', linestyle='--', label=f"Safe {TH_TTC_S}")
    plt.axhline(y=TH_TTC_R, color='orange', linestyle='--', label=f"Risky {TH_TTC_R}")
    plt.axhline(y=TH_TTC_C, color='r', linestyle='--', label=f"Critical {TH_TTC_C}")
    plt.ylabel("TTC [ms]")
    plt.title(f"{name} - TTC")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.scatter(steps, state_values, c=colors, s=100, edgecolors='black')
    plt.yticks(list(state_map.values()), [s.value for s in state_map.keys()])
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.title(f"{name} - Automaton States")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================
# Tests
# ==============================================================

def test_initial_state():
    auto = PedestrianProtectionAutomaton()
    report("Initial state is NORMAL", auto.state == State.NORMAL)

def test_safe_consensus_normal():
    auto = PedestrianProtectionAutomaton()
    ttc_seq = [TH_TTC_S * 1.5] * N
    states, buffers = run_sequence(auto, ttc_seq)
    report("Safe TTC maintains NORMAL", auto.state == State.NORMAL, ttc_seq, states, buffers)

def test_throttling_consensus():
    auto = PedestrianProtectionAutomaton()
    k = consensus_frames(SR_DISTANCE_CONSENSUS)
    threshold = consensus_threshold(SR_DISTANCE_CONSENSUS)

    ttc_seq1 = [TH_TTC_S * 0.95] * (threshold - 1)
    states1, buffers1 = run_sequence(auto, ttc_seq1)
    cond1 = auto.state == State.NORMAL

    ttc_seq2 = [TH_TTC_S * 0.95] * k
    states2, buffers2 = run_sequence(auto, ttc_seq2)
    cond2 = auto.state == State.THROTTLING  # Correct for risky TTC

    report("Throttling triggered only after consensus",
           cond1 and cond2, ttc_seq1 + ttc_seq2, states1 + states2, buffers1 + buffers2)

def test_emergency_consensus():
    auto = PedestrianProtectionAutomaton()
    k = consensus_frames(C_DISTANCE_CONSENSUS)
    threshold = consensus_threshold(C_DISTANCE_CONSENSUS)

    ttc_seq1 = [TH_TTC_C * 0.9] * (threshold - 1)
    states1, buffers1 = run_sequence(auto, ttc_seq1)
    cond1 = auto.state != State.EMERGENCY_BRAKING
    ttc_seq2 = [TH_TTC_C * 0.9] * k
    states2, buffers2 = run_sequence(auto, ttc_seq2)
    cond2 = auto.state == State.EMERGENCY_BRAKING

    report("Emergency braking triggered only after consensus",
           cond1 and cond2, ttc_seq1 + ttc_seq2, states1 + states2, buffers1 + buffers2)

def test_noise_resilience():
    auto = PedestrianProtectionAutomaton()
    k = consensus_frames(C_DISTANCE_CONSENSUS)
    ttc_seq = []
    for i in range(N * 2):
        if i % (k + 1) == 0:
            ttc_seq.append(TH_TTC_C * 0.8)  # Noise spike
        else:
            ttc_seq.append(TH_TTC_S * 1.5)
    states, buffers = run_sequence(auto, ttc_seq)
    cond = all(s in (State.NORMAL, State.SAFE_WARNING) for s in states)
    report("Noise spikes do not trigger braking", cond, ttc_seq, states, buffers)

def test_deescalation_after_consensus():
    auto = PedestrianProtectionAutomaton()
    k = consensus_frames(C_DISTANCE_CONSENSUS)

    ttc_seq1 = [TH_TTC_C * 0.9] * k
    states1, buffers1 = run_sequence(auto, ttc_seq1)
    cond1 = auto.state == State.EMERGENCY_BRAKING

    ttc_seq2 = [TH_TTC_S * 1.5] * RT_HALF_FRAMES
    states2, buffers2 = run_sequence(auto, ttc_seq2, conf=1.0, cross=False)
    cond2 = auto.state == State.NORMAL

    report("Automaton de-escalates to NORMAL after safe consensus",
           cond1 and cond2,
           ttc_seq1 + ttc_seq2,
           states1 + states2,
           buffers1 + buffers2)

# ==============================================================
# Run all tests
# ==============================================================

if __name__ == "__main__":
    print("Running PedestrianProtectionAutomaton consensus-aware tests...\n")

    test_initial_state()
    test_safe_consensus_normal()
    test_throttling_consensus()
    test_emergency_consensus()
    test_noise_resilience()
    test_deescalation_after_consensus()

    print("\nAll tests completed.")
