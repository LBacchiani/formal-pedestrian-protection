import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_DIR = "./logs"  # cartella con i file .jsonl (uno per scenario)
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Carica tutti i log jsonl ===
def load_all_logs(folder):
    records = []
    for filename in os.listdir(folder):
        if filename.endswith(".jsonl"):
            scenario_name = filename.replace(".jsonl", "")
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        data["scenario_name"] = scenario_name
                        records.append(data)
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(records)

df = load_all_logs(INPUT_DIR)
print(f"[INFO] Loaded {len(df)} runs from {df['scenario_name'].nunique()} scenarios.")

# === Normalizza e prepara i campi principali ===
def extract_metrics(row):
    scen = row.get("scenario", {})
    speed_kmh = scen.get("speed_kmh", None)
    is_day = scen.get("is_day", True)
    stop_events = row.get("stop_events", [])

    # Stop distance: media degli eventi, se presenti
    stop_distance = None
    if stop_events:
        stop_distance = pd.DataFrame(stop_events)["stopping_distance"].mean()

    # Reaction times
    rt_ttc = []
    rt_sim = []
    for a in ("mild_brake", "brake", "emergency_brake"):
        rt_ttc += row["reaction_times_ttc_based"].get(a, [])
        rt_sim += row["reaction_times_simulation"].get(a, [])

    return pd.Series({
        "scenario_name": row.get("scenario_name"),
        "speed_kmh": speed_kmh,
        "is_day": is_day,
        "residual_speed_kmh": row.get("residual_speed_kmh"),
        "impact_force_N": row.get("impact_force_N"),
        "collision_count": row.get("collisions", {}).get("count", 0),
        "with_pedestrian": row.get("collisions", {}).get("with_pedestrian", 0),
        "reaction_time_ttc_mean": pd.Series(rt_ttc).mean() if rt_ttc else None,
        "reaction_time_sim_mean": pd.Series(rt_sim).mean() if rt_sim else None,
        "stop_distance_mean": stop_distance,
    })

metrics_df = df.apply(extract_metrics, axis=1)
metrics_df.dropna(subset=["speed_kmh"], inplace=True)
metrics_df["day_night"] = metrics_df["is_day"].map({True: "Day", False: "Night"})

# === Calcola collision rate ===
# → fraction of runs where a pedestrian was hit
collision_rate = (
    metrics_df.groupby(["scenario_name", "speed_kmh", "day_night"])["with_pedestrian"]
    .mean()
    .reset_index()
    .rename(columns={"with_pedestrian": "collision_rate"})
)

# === Aggrega le altre metriche per gruppo ===
agg_metrics = (
    metrics_df.groupby(["scenario_name", "speed_kmh", "day_night"])
    .agg({
        "residual_speed_kmh": "mean",
        "reaction_time_ttc_mean": "mean",
        "reaction_time_sim_mean": "mean",
        "stop_distance_mean": "mean",
    })
    .reset_index()
    .merge(collision_rate, on=["scenario_name", "speed_kmh", "day_night"])
)

# === Salva CSV aggregato ===
output_csv = os.path.join(OUTPUT_DIR, "aggregated_metrics.csv")
agg_metrics.to_csv(output_csv, index=False)
print(f"[OK] Aggregated metrics saved to {output_csv}")

# === Plot automatici per scenario ===
sns.set(style="whitegrid")

for scen, group in agg_metrics.groupby("scenario_name"):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=group,
        x="speed_kmh", y="collision_rate", hue="day_night",
        palette="coolwarm", alpha=0.8
    )
    plt.title(f"Collision Rate – {scen}")
    plt.ylabel("Collision Rate")
    plt.xlabel("Vehicle speed [km/h]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_collision_rate.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=group,
        x="speed_kmh", y="reaction_time_sim_mean",
        hue="day_night", marker="o"
    )
    plt.title(f"Reaction Time (Simulation-based) – {scen}")
    plt.ylabel("Reaction time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_sim.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=group,
        x="speed_kmh", y="residual_speed_kmh",
        hue="day_night", marker="o"
    )
    plt.title(f"Residual Speed  {scen}")
    plt.ylabel("Residual speed [km/h]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_residual_speed.png"))
    plt.close()

print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
