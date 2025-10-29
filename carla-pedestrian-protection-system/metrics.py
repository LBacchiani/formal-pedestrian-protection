import json
import pandas as pd

METRICS_PATH = "metrics_log.jsonl"
MERGE_WINDOW = 2.0  # seconds to merge consecutive braking events
NEAR_MISS_THRESHOLD = 2.0  # threshold for near-miss events

# --- Load records ---
records = []
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass

df = pd.json_normalize(records)

# --- Scenario classification ---
df["scenario_type"] = df["scenario.weather.sun_altitude_angle"].apply(
    lambda x: "DAY" if x >= 0 else "NIGHT"
)
df["ped_mode"] = df["scenario.pedestrian_mode"].fillna("UNKNOWN")

# --- Count only collisions with pedestrians ---
def count_pedestrian_collisions(events):
    if not isinstance(events, list):
        return 0
    return sum(1 for e in events if e.get("other_actor") == "walker.pedestrian.0042")

df["ped_collisions"] = df["collisions.events"].apply(count_pedestrian_collisions)

# --- Collision rate per scenario ---
collision_summary = (
    df.groupby(["scenario_type", "ped_mode"])
      .agg(
          runs=("run_id", "count"),
          ped_collisions=("ped_collisions", "sum"),
          collision_rate=("ped_collisions", lambda x: (x > 0).mean())
      )
      .reset_index()
)

# --- Extract and merge braking records ---
brake_records = []
for run in records:
    dmin_recs = sorted(run.get("d_min_records", []), key=lambda r: r.get("timestamp", 0))
    if not dmin_recs:
        continue

    merged_recs = []
    prev = None

    for rec in dmin_recs:
        if prev is None:
            prev = rec
            continue

        dt = rec["timestamp"] - prev["timestamp"]
        if dt < MERGE_WINDOW:
            # Merge consecutive brake events within 2 seconds
            prev["d_min"] = min(prev.get("d_min", float("inf")), rec.get("d_min", float("inf")))
            prev["stopping_distance"] = max(
                prev.get("stopping_distance", 0) or 0,
                rec.get("stopping_distance", 0) or 0
            )
            prev["speed_before_brake"] = prev.get("speed_before_brake") or rec.get("speed_before_brake")
            # If any of the merged events is emergency, mark as emergency
            if rec.get("action") == "emergency_brake":
                prev["action"] = "emergency_brake"
        else:
            merged_recs.append(prev)
            prev = rec
    if prev:
        merged_recs.append(prev)

    for ev in merged_recs:
        brake_records.append({
            "run_id": run["run_id"],
            "scenario_type": "DAY" if run["scenario"]["weather"]["sun_altitude_angle"] >= 0 else "NIGHT",
            "ped_mode": run["scenario"]["pedestrian_mode"],
            "action": ev.get("action", "brake"),
            "d_min": ev.get("d_min"),
            "stopping_distance": ev.get("stopping_distance"),
            "speed_before_brake": ev.get("speed_before_brake")
        })

brake_df = pd.DataFrame(brake_records)

# --- Aggregate braking statistics ---
agg_df = (
    brake_df.groupby(["scenario_type", "ped_mode"])
    .agg(
        mean_dmin=("d_min", "mean"),
        min_dmin=("d_min", "min"),
        max_dmin=("d_min", "max"),
        mean_stopdist=("stopping_distance", "mean"),
        mean_speed_before_brake=("speed_before_brake", "mean")
    )
    .reset_index()
)

# --- Near-miss events ---
brake_df["near_miss"] = brake_df["d_min"].apply(
    lambda x: 1 if x is not None and x < NEAR_MISS_THRESHOLD else 0
)
near_miss = brake_df.groupby(["scenario_type", "ped_mode"])["near_miss"].sum().reset_index()

# --- Brake and emergency counts from log ---
brake_counts = (
    brake_df.groupby(["scenario_type", "ped_mode"])
    .agg(
        total_brakes=("action", lambda x: (x == "brake").sum()),
        emergency_brakes=("action", lambda x: (x == "emergency_brake").sum())
    )
    .reset_index()
)

# --- Combine all summaries ---
summary = (
    collision_summary
    .merge(agg_df, on=["scenario_type", "ped_mode"], how="outer")
    .merge(near_miss, on=["scenario_type", "ped_mode"], how="outer")
    .merge(brake_counts, on=["scenario_type", "ped_mode"], how="outer")
)

# --- Derived metrics ---
summary["collision_per_brake_rate"] = summary.apply(
    lambda row: (
        row["ped_collisions"] / (row["total_brakes"] + row["emergency_brakes"])
        if (row["total_brakes"] + row["emergency_brakes"]) > 0 else 0
    ),
    axis=1
)
summary["saved_lives"] = (
    summary["total_brakes"] + summary["emergency_brakes"] - summary["ped_collisions"]
)

summary = summary.fillna(0)

# --- Print summary ---
print("\n=== METRICS BY SCENARIO (merged braking events) ===\n")
print(summary.to_string(index=False, float_format="%.2f"))

# === SUMMARY COLUMN DESCRIPTIONS ===
# scenario_type: Indicates whether the scenario takes place during the day (DAY) or at night (NIGHT).
# ped_mode: Defines the pedestrian’s behavior mode in the scenario.
# runs: Number of simulation runs executed for each combination of scenario type and pedestrian mode.
# ped_collisions: Total number of collisions involving pedestrians (excluding other objects).
# collision_rate: Percentage of runs that resulted in at least one collision with a pedestrian.
# mean_dmin: Average minimum distance between the vehicle and the pedestrian during braking events.
# min_dmin: Absolute minimum distance reached.
# max_dmin: Largest minimum distance recorded.
# mean_stopdist: Average stopping distance of the vehicle after braking.
# mean_speed_before_brake: Average vehicle speed immediately before braking.
# near_miss: Number of events where the minimum distance was below the safety threshold but no collision occurred.
# total_brakes: Number of normal braking events ("brake").
# emergency_brakes: Number of emergency braking events ("emergency_brake").
# collision_per_brake_rate: Fraction of all braking events (normal + emergency) that resulted in a collision.
# saved_lives: Total number of pedestrians effectively avoided = total_brakes + emergency_brakes - ped_collisions.
