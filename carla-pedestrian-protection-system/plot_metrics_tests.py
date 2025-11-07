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

# === Estrazione metrica per ogni run ===
def extract_metrics(row):
    scen = row.get("scenario", {}) or {}
    speed_kmh = scen.get("speed_kmh")
    is_day = scen.get("is_day", True)
    stop_events = row.get("stop_events", [])

    stop_distance = None
    d_min_value = None
    emergency_brake_count = 0
    total_stops = 0

    if isinstance(stop_events, list) and stop_events:
        try:
            stop_df = pd.DataFrame(stop_events)
            total_stops = len(stop_df)
            if "action" in stop_df.columns:
                emergency_brake_count = (stop_df["action"] == "emergency_brake").sum()
            if "stopping_distance" in stop_df.columns:
                stop_distance = stop_df["stopping_distance"].mean()
            if "d_min" in stop_df.columns:
                d_min_value = stop_df["d_min"].mean()
        except Exception:
            pass

    # Reaction times
    rt_ttc, rt_sim = [], []
    for a in ("mild_brake", "brake", "emergency_brake"):
        rt_ttc += row.get("reaction_times_ttc_based", {}).get(a, []) or []
        rt_sim += row.get("reaction_times_simulation", {}).get(a, []) or []

    collisions = row.get("collisions", {}) or {}

    return pd.Series({
        "code": row.get("code"),
        "scenario_name": row.get("scenario_name"),
        "speed_kmh": speed_kmh,
        "is_day": is_day,
        "residual_speed_kmh": row.get("residual_speed_kmh"),
        "impact_force_N": row.get("impact_force_N"),
        "collision_count": collisions.get("count", 0),
        "with_pedestrian": collisions.get("with_pedestrian", 0),
        "reaction_time_ttc_mean": pd.Series(rt_ttc).mean() if rt_ttc else None,
        "reaction_time_sim_mean": pd.Series(rt_sim).mean() if rt_sim else None,
        "stop_distance_mean": stop_distance,
        "d_min": d_min_value,
        "emergency_brake_count": emergency_brake_count,
        "total_stops": total_stops
    })

def compute_scenario_score(row):
    import pandas as pd

    scenario_code = row.get("code")
    v_test = row.get("speed_kmh", 0)
    v_impact = row.get("residual_speed_kmh", v_test)
    collided = (row.get("with_pedestrian", 0) > 0) or (row.get("collision_count", 0) > 0)

    if pd.isna(v_impact):
        v_impact = v_test

    # Evita valori negativi
    v_impact = max(0.0, v_impact)

    # Soglie massime per scenario (velocità d'impatto in km/h a cui il punteggio = 0)
    # Valori basati su Euro NCAP 2024 v4.5.1 (AEB Pedestrian)
    max_vimpact = {
        "CPFA": 40,   # Car-to-Pedestrian Farside Adult 50 %
        "CPNA": 35,   # Nearside Adult
        "CPNCO": 30,  # Nearside Child Obstructed
        "CPLA": 25,   # Longitudinal Adult
        "CPTA": 20,   # Turning Adult
        "CPRA": 8,    # Reverse Adult
        "CPRC": 8,    # Reverse Child
    }

    vmax = max_vimpact.get(scenario_code, 35.0)

    # --- Calcolo punteggio secondo NCAP (interpolazione lineare) ---
    if not collided:
        return 100.0

    # Se collisione e impatto ≥ soglia → 0 punti
    if v_impact >= vmax:
        return 0.0

    # Se collisione con impatto inferiore → interpolazione lineare
    # punteggio = (1 - v_impact/vmax) * 100
    score = max(0.0, min(100.0, (1.0 - (v_impact / vmax)) * 100.0))
    return score


metrics_df = df.apply(extract_metrics, axis=1)
metrics_df.dropna(subset=["speed_kmh"], inplace=True)
metrics_df["day_night"] = metrics_df["is_day"].map({True: "Day", False: "Night"})
metrics_df["points_ncap"] = metrics_df.apply(compute_scenario_score, axis=1)

print(metrics_df.groupby(["code", "speed_kmh", "day_night"])["points_ncap"].describe())
# === Aggregazione ===
agg_metrics = (
    metrics_df.groupby(["scenario_name", "code", "speed_kmh", "day_night"])
    .agg({
        "residual_speed_kmh": "mean",
        "reaction_time_ttc_mean": "mean",
        "reaction_time_sim_mean": "mean",
        "stop_distance_mean": "mean",
        "d_min": "mean",
        "collision_count": "sum",
        "with_pedestrian": "mean",
        "emergency_brake_count": "sum",
        "total_stops": "sum",
        "points_ncap": "mean",
    })
    .reset_index()
)

# === Calcolo rate ===
agg_metrics["collision_rate"] = agg_metrics["with_pedestrian"]
agg_metrics["emergency_brake_rate"] = (
    agg_metrics["emergency_brake_count"] / agg_metrics["total_stops"]
).replace([float("inf"), pd.NA], 0)

# === Salva CSV con nomi personalizzati ===
output_csv = os.path.join(OUTPUT_DIR, "aggregated_metrics.csv")
agg_metrics_renamed = agg_metrics.rename(columns={
    "reaction_time_ttc_mean": "Automaton reaction time",
    "reaction_time_sim_mean": "4-second TTC reaction time",
    "stop_distance_mean": "Automaton braking distance [m]",
    "d_min": "Minimum distance from pedestrian [m]",
    "points_ncap": "NCAP points"
})


agg_metrics_renamed.to_csv(output_csv, index=False)
print(f"[OK] Aggregated metrics saved to {output_csv} (with renamed columns)")


# === Plot automatici ===
sns.set(style="whitegrid")

def safe_plot(plot_func, data, x, y, **kwargs):
    """Esegue un plot solo se esistono dati validi."""
    if y not in data.columns:
        return
    sub = data.dropna(subset=[y])
    if sub.empty:
        return
    plt.figure(figsize=(10, 6))
    plot_func(data=sub, x=x, y=y, **kwargs)
    plt.tight_layout()

for scen, group in agg_metrics.groupby("scenario_name"):
    # Collision rate
    safe_plot(
        sns.barplot,
        data=group,
        x="speed_kmh",
        y="collision_rate",
        hue="day_night",
        palette="coolwarm",
        alpha=0.8,
    )
    plt.title(f"Collision Rate – {scen}")
    plt.ylabel("Collision Rate")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_collision_rate.png"))
    plt.close()

    # Reaction time (simulation)
    safe_plot(
        sns.lineplot,
        data=group,
        x="speed_kmh",
        y="reaction_time_sim_mean",
        hue="day_night",
        marker="o",
    )
    plt.title(f"Reaction Time (Simulation-based) – {scen}")
    plt.ylabel("Reaction time [s]")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_sim.png"))
    plt.close()

    # Residual speed
    safe_plot(
        sns.lineplot,
        data=group,
        x="speed_kmh",
        y="residual_speed_kmh",
        hue="day_night",
        marker="o",
    )
    plt.title(f"Residual Speed – {scen}")
    plt.ylabel("Residual speed [km/h]")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_residual_speed.png"))
    plt.close()

    # d_min mean
    safe_plot(
        sns.lineplot,
        data=group,
        x="speed_kmh",
        y="d_min",
        hue="day_night",
        marker="o",
    )
    plt.title(f"Mean d_min – {scen}")
    plt.ylabel("d_min [m]")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_dmin_mean.png"))
    plt.close()

    # Emergency brake rate
    safe_plot(
        sns.barplot,
        data=group,
        x="speed_kmh",
        y="emergency_brake_rate",
        hue="day_night",
        palette="Reds",
        alpha=0.8,
    )
    plt.title(f"Emergency Brake Rate – {scen}")
    plt.ylabel("Emergency brake rate")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_emergency_brake_rate.png"))
    plt.close()

    safe_plot(
        sns.barplot,
        data=group,
        x="speed_kmh",
        y="punteggio_ncap",
        hue="day_night",
        palette="viridis",
    )
    plt.title(f"Punteggio NCAP – {scen}")
    plt.ylabel("Punteggio (%)")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_punteggio_ncap.png"))
    plt.close()



print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
