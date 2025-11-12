import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "./logs"
OUTPUT_DIR = "./results1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_logs(folder):
    records = []
    for filename in os.listdir(folder):
        if filename.endswith("1.jsonl"):
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

all_rt = {"mild_brake": [], "brake": [], "emergency_brake": []}

for _, row in df.iterrows():
    rt_dict = row.get("reaction_times_ttc_based", {}) or {}
    for k in all_rt.keys():
        all_rt[k] += rt_dict.get(k, []) or []

global_means = {k: (np.mean(v) if v else None) for k, v in all_rt.items()}

# Converti in DataFrame per salvare insieme al resto
global_df = pd.DataFrame([global_means])
global_df["type"] = "Global Mean (reaction_times_ttc_based)"
print("[INFO] Global reaction time means:", global_means)

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
        "collision_count": collisions.get("with_pedestrian", 0),
        "with_pedestrian": collisions.get("with_pedestrian", 0),
        "reaction_time_ttc_mean": pd.Series(rt_ttc).mean() if rt_ttc else None,
        "reaction_time_sim_mean": pd.Series(rt_sim).mean() if rt_sim else None,
        "stop_distance_mean": stop_distance,
        "d_min": d_min_value,
        "emergency_brake_count": emergency_brake_count,
        "total_stops": total_stops
    })

def compute_scenario_score(row):
    import numpy as np
    import pandas as pd

    v_test = row.get("speed_kmh", 0.0)
    v_impact = row.get("residual_speed_kmh", v_test)
    collided = (row.get("with_pedestrian", 0) > 0) or (row.get("collision_count", 0) > 0)

    if pd.isna(v_impact):
        v_impact = v_test

    v_impact = max(0.0, v_impact)
    delta_v = max(0.0, v_test - v_impact) 

    if v_test < 40:
        return 100.0 if not collided else 0.0


    x = np.array([0, 5, 10, 15, 20])
    y = np.array([0, 0.25, 0.5, 0.75, 1.0])
    score_norm = np.interp(delta_v, x, y)

    if not collided:
        return 100.0
    return round(score_norm * 100.0, 2)


metrics_df = df.apply(extract_metrics, axis=1)
metrics_df.dropna(subset=["speed_kmh"], inplace=True)
metrics_df["day_night"] = metrics_df["is_day"].map({True: "Day", False: "Night"})
metrics_df["points_ncap"] = metrics_df.apply(compute_scenario_score, axis=1)

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

agg_metrics["collision_rate"] = agg_metrics["with_pedestrian"]
agg_metrics["emergency_brake_rate"] = (
    agg_metrics["emergency_brake_count"] / agg_metrics["total_stops"]
).replace([float("inf"), pd.NA], 0)

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
    # === Boxplot della d_min ===
    sub = metrics_df[metrics_df["scenario_name"] == scen]

    if not sub.empty and "d_min" in sub.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub,
            x="speed_kmh",
            y="d_min",
            hue="day_night",
            palette="coolwarm",
            showfliers = False,
            linewidth=1.2,    # spessore linee box
            width=0.6
        )
        plt.title(f"Distribution of d_min – {scen}", fontsize=14, weight="bold")
        plt.ylabel("Minimum distance from pedestrian [m]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_dmin_boxplot.png"), dpi=200)
        plt.close()


    # === BOX 1: Automaton reaction time (reaction_times_ttc_based) ===
    sub_ttc = metrics_df[metrics_df["scenario_name"] == scen]
    if not sub_ttc.empty and "reaction_time_ttc_mean" in sub_ttc.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub_ttc,
            x="speed_kmh",
            y="reaction_time_ttc_mean",
            hue="day_night",
            palette="crest",
            showfliers=False,
            linewidth=1.2,
            width=0.6
        )
        plt.title(f"Automaton Reaction Time – {scen}", fontsize=14, weight="bold")
        plt.ylabel("Automaton reaction time [s]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_ttc_based.png"), dpi=200)
        plt.close()


    # === BOX 2: Simulation reaction time (reaction_times_simulation) ===
    sub_sim = metrics_df[metrics_df["scenario_name"] == scen]
    if not sub_sim.empty and "reaction_time_sim_mean" in sub_sim.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub_sim,
            x="speed_kmh",
            y="reaction_time_sim_mean",
            hue="day_night",
            palette="mako",
            showfliers=False,
            linewidth=1.2,
            width=0.6
        )
        plt.title(f"Simulation Reaction Time – {scen}", fontsize=14, weight="bold")
        plt.ylabel("Simulation reaction time [s]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_simulation.png"), dpi=200)
        plt.close()


print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
