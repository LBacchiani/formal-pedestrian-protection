import time
import matplotlib.pyplot as plt
from automaton import PedestrianProtectionAutomaton, State, Action

# ==============================================================
# Utility
# ==============================================================

def simulate_scenario(name, ttc_seq, conf_seq, cross_seq, expected_states, delay_ms=100):
    """
    Simulate a scenario and compare actual vs expected state transitions.
    Returns history for plotting.
    """
    print(f"\n=== SCENARIO: {name} ===")
    automaton = PedestrianProtectionAutomaton()
    history = {"step": [], "ttc": [], "state": [], "action": []}

    for i, (ttc, conf, cross) in enumerate(zip(ttc_seq, conf_seq, cross_seq)):
        automaton.update_data(confidence=conf, ttc=ttc, is_crossing=cross)
        action = automaton.step()
        state = automaton.state

        history["step"].append(i)
        history["ttc"].append(ttc)
        history["state"].append(state)
        history["action"].append(action)

        print(f"Step {i:02d} | TTC={ttc:5.0f} ms | Conf={conf:.2f} | Cross={cross} "
              f"| State={state.value:<18} | Action={action.value:<16}")

        time.sleep(delay_ms / 1000.0)

    # Check expected end-state
    final_state = automaton.state
    expected_final = expected_states[-1]
    print(f"Expected final: {expected_final.value}, Got: {final_state.value}")

    assert final_state == expected_final, (
        f"[FAIL] {name}: Expected final {expected_final.value}, got {final_state.value}"
    )

    print(f"[OK] {name} completed successfully.")
    return history


def plot_scenario(name, history):
    """Plot TTC and state evolution."""
    plt.figure(figsize=(10, 4))
    plt.title(f"Scenario: {name}")
    plt.plot(history["step"], history["ttc"], label="TTC (ms)")
    plt.xlabel("Step")
    plt.ylabel("TTC [ms]")

    # Add state annotations
    states = [s.value for s in history["state"]]
    for i, state in enumerate(states):
        plt.text(i, history["ttc"][i] + 100, state, fontsize=7, rotation=45, ha="center")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==============================================================
# Scenarios
# ==============================================================

# 1. FAST APPROACH (crossing)
ttc_fast = [5000, 4000, 3000, 2000, 1500, 1000, 700, 500, 400, 300]
conf_high = [0.9] * len(ttc_fast)
cross_yes = [True] * len(ttc_fast)

expected_states_fast = [
    State.NORMAL,
    State.SAFE_WARNING,
    State.THROTTLING,
    State.SOFT_BRAKING,
    State.EMERGENCY_BRAKING
]

hist1 = simulate_scenario("FAST APPROACH (CROSSING)", ttc_fast, conf_high, cross_yes, expected_states_fast)
plot_scenario("FAST APPROACH (CROSSING)", hist1)


# 2. BRAKING AFTER DETECTION (crossing but TTC increases)
ttc_braking = [800, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000]
conf_high = [0.9] * len(ttc_braking)
cross_yes = [True] * len(ttc_braking)

expected_states_brake = [
    State.EMERGENCY_BRAKING,
    State.SOFT_BRAKING,
    State.THROTTLING,
    State.SAFE_WARNING,
    State.NORMAL
]

hist2 = simulate_scenario("BRAKING AFTER DETECTION", ttc_braking, conf_high, cross_yes, expected_states_brake)
plot_scenario("BRAKING AFTER DETECTION", hist2)


# 3. NO PEDESTRIAN (safe cruise)
ttc_safe = [6000, 6500, 7000, 7200, 7500, 7600, 8000, 8500, 9000, 9500]
conf_low = [0.0] * len(ttc_safe)
cross_no = [False] * len(ttc_safe)

expected_states_safe = [State.NORMAL]
hist3 = simulate_scenario("NO PEDESTRIAN (SAFE)", ttc_safe, conf_low, cross_no, expected_states_safe)
plot_scenario("NO PEDESTRIAN (SAFE)", hist3)
