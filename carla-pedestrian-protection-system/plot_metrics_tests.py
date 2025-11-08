import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "./logs"
OUTPUT_DIR = "./results2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=group,
        x="speed_kmh",
        y="collision_rate",
        hue="day_night",
        palette="coolwarm",
        alpha=0.8,
        edgecolor="black"
    )
    plt.title(f"Collision Rate – {scen}", fontsize=14, weight="bold")
    plt.ylabel("Collision Rate", fontsize=12)
    plt.xlabel("Vehicle speed [km/h]", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_collision_rate.png"), dpi=200)
    plt.close()

    sub = group[group["collision_rate"] > 0]
    if not sub.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=sub,
            x="speed_kmh",
            y="residual_speed_kmh",
            hue="day_night",
            palette="coolwarm",
            alpha=0.9,
            edgecolor="black"
        )
        plt.axhline(20, color="gray", linestyle="--", linewidth=1)
        plt.text(sub["speed_kmh"].max(), 21, "NCAP 20 km/h", color="gray", ha="right", fontsize=10)

        plt.title(f"Residual Impact Speed – {scen}", fontsize=14, weight="bold")
        plt.ylabel("Residual impact speed [km/h]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.ylim(0, max(sub["residual_speed_kmh"].max() * 1.2, 10))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_impact_speed.png"), dpi=200)
        plt.close()





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
        y="points_ncap",
        hue="day_night",
        palette="viridis",
    )
    plt.title(f"NCAP Score")
    plt.ylabel("Punteggio (%)")
    plt.xlabel("Vehicle speed [km/h]")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_punteggio_ncap.png"))
    plt.close()

print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
