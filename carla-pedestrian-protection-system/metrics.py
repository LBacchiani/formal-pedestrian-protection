import json
import pandas as pd

METRICS_PATH = "metrics_log.jsonl"
MERGE_WINDOW = 2.0  # FIXXX seconds to merge consecutive braking events

records = []
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass

df = pd.json_normalize(records)

df["scenario_type"] = df["scenario.weather.sun_altitude_angle"].apply(
    lambda x: "DAY" if x >= 0 else "NIGHT"
)
df["ped_mode"] = df["scenario.pedestrian_mode"].fillna("UNKNOWN")

# Count only collisions with pedestrians
def count_pedestrian_collisions(events):
    if not isinstance(events, list):
        return 0
    return sum(1 for e in events if e.get("other_actor") == "walker.pedestrian.0042")

df["ped_collisions"] = df["collisions.events"].apply(count_pedestrian_collisions)
df["has_ped_collision"] = df["ped_collisions"].apply(lambda x: 1 if x > 0 else 0)

# Collision rate per scenario
collision_summary = (
    df.groupby(["scenario_type", "ped_mode"])
      .agg(
          runs=("run_id", "count"),
          ped_collisions=("ped_collisions", "sum"),
          collision_rate=("has_ped_collision", "mean")
      )
      .reset_index()
)

# Extract all braking records
brake_records = []
for run in records:
    dmin_recs = run.get("d_min_records", [])
    if not dmin_recs:
        continue

    dmin_recs = sorted(dmin_recs, key=lambda r: r.get("timestamp", 0))
    merged_recs = []

    # Merge consecutive braking events within MERGE_WINDOW seconds
    prev = None
    for rec in dmin_recs:
        if prev is None:
            prev = rec
            continue

        dt = rec["timestamp"] - prev["timestamp"]
        if dt < MERGE_WINDOW:
            # Merge: keep the min distance and max stopping distance
            prev["d_min"] = min(prev.get("d_min", float("inf")), rec.get("d_min", float("inf")))
            prev["stopping_distance"] = max(
                prev.get("stopping_distance", 0) or 0,
                rec.get("stopping_distance", 0) or 0
            )
            prev["speed_before_brake"] = (
                prev.get("speed_before_brake") or rec.get("speed_before_brake")
            )
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
            "d_min": ev.get("d_min"),
            "stopping_distance": ev.get("stopping_distance"),
            "speed_before_brake": ev.get("speed_before_brake")
        })

brake_df = pd.DataFrame(brake_records)

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

# Near-miss events (d_min < threshold, no collision)
NEAR_MISS_THRESHOLD = 2.0
brake_df["near_miss"] = brake_df["d_min"].apply(
    lambda x: 1 if x is not None and x < NEAR_MISS_THRESHOLD else 0
)
near_miss = (
    brake_df.groupby(["scenario_type", "ped_mode"])["near_miss"].sum().reset_index()
)

summary = (
    collision_summary
    .merge(agg_df, on=["scenario_type", "ped_mode"], how="outer")
    .merge(near_miss, on=["scenario_type", "ped_mode"], how="outer")
)

summary = summary.fillna(0)

print("\n=== METRICS BY SCENARIO (merged braking events) ===\n")
print(summary.to_string(index=False, float_format="%.2f"))

# Output example:
#scenario_type ped_mode  runs  ped_collisions  collision_rate  mean_dmin  min_dmin  max_dmin  mean_stopdist  mean_speed_before_brake  near_miss
#          DAY      DET     3             160            1.00       5.13      0.21     11.32           4.56                     8.65          3
