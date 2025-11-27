import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "./logs"
OUTPUT_DIR = "./results3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_logs(folder):
    records = []
    for filename in os.listdir(folder):
        if filename.endswith("3.jsonl"):
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

df["cross_side"] = df["code"].apply(
    lambda x: "FarSide" if "FarSide" in str(x) else ("NearSide" if "NearSide" in str(x) else None)
)

all_rt = {"mild_brake": [], "brake": [], "emergency_brake": []}

for _, row in df.iterrows():
    rt_dict = row.get("reaction_times_ttc_based", {}) or {}
    for k in all_rt.keys():
        all_rt[k] += rt_dict.get(k, []) or []

global_means = {k: (np.mean(v) if v else None) for k, v in all_rt.items()}

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
            else:
                d_min_value = None

        except Exception:
            pass

    collisions = row.get("collisions", {}) or {}
    if collisions.get("with_pedestrian", 0) > 0:
        d_min_value = 0.0

    # Reaction times
    rt_ttc, rt_sim = [], []
    for a in ("mild_brake", "brake", "emergency_brake"):
        rt_ttc += row.get("reaction_times_ttc_based", {}).get(a, []) or []
        rt_sim += row.get("reaction_times_simulation", {}).get(a, []) or []

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
        "total_stops": total_stops,
        "cross_side": "FarSide" if "FarSide" in str(row.get("code")) else ("NearSide" if "NearSide" in str(row.get("code")) else None),

    })


def compute_scenario_score(row):
    import numpy as np
    import pandas as pd

    scen_name = row.get("scenario_name", "").lower()
    v_test = row.get("speed_kmh", 0.0)
    v_impact = row.get("residual_speed_kmh", v_test)
    collided = (row.get("with_pedestrian", 0) > 0) or (row.get("collision_count", 0) > 0)

    if pd.isna(v_impact):
        v_impact = v_test

    v_impact = max(0.0, v_impact)

    if "cpta" in scen_name:
        if not collided:
            return 100.0

        if v_test <= 10:
            score = max(0.0, 1.0 - (v_impact / 10.0)) * 60.0
            return round(score, 2)

        if v_test <= 20:
            if v_impact <= 5:
                score = 100.0 - (v_impact * 6.0)   
            else:
                score = max(0.0, 70.0 - (v_impact - 5) * 10)

            return round(score, 2)
        return 0.0

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
metrics_df = metrics_df[(metrics_df["d_min"].isna()) | (metrics_df["d_min"] < 15)]
metrics_df["day_night"] = metrics_df["is_day"].map({True: "Day", False: "Night"})
metrics_df["points_ncap"] = metrics_df.apply(compute_scenario_score, axis=1)


def compute_residual_speed_latex(series):
    series = pd.to_numeric(series.dropna(), errors="coerce")
    n = len(series)

    if n == 0:
        return None 

    if n == 1:
        m = series.iloc[0]
        return f"{m:.2f} ± 0.00"

    mean = series.mean()
    std = series.std(ddof=1)
    ci = 1.96 * std / np.sqrt(n)

    return f"{mean:.2f} ± {ci:.2f}"


agg_metrics = (
    metrics_df.groupby(["scenario_name", "code", "speed_kmh", "day_night"])
    .agg({
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


residual_group = (
    metrics_df.groupby(["scenario_name", "code", "speed_kmh", "day_night"])["residual_speed_kmh"]
    .apply(compute_residual_speed_latex)
    .reset_index(name="residual_speed_latex")
)

agg_metrics = agg_metrics.merge(
    residual_group,
    on=["scenario_name", "code", "speed_kmh", "day_night"],
    how="left"
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

PALETTE_DAY_NIGHT = {"Day": "#99d3fd", "Night": "#ff7f0e"}

for scen, group in agg_metrics.groupby("scenario_name"):
    sub = metrics_df[metrics_df["scenario_name"] == scen]

    if not sub.empty and "d_min" in sub.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub,
            x="speed_kmh",
            y="d_min",
            hue="cross_side",
            hue_order=["FarSide", "NearSide"],
            palette={"FarSide": "#49a7eb", "NearSide": "#efad73"},
            showfliers=False,
            linewidth=1.2,
            width=0.6
        )
        plt.legend(title="Crossing side", fontsize=11, title_fontsize=12)

        plt.ylabel("Minimum distance from pedestrian [m]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_dmin_boxplot.png"), dpi=200)
        plt.close()

    if not sub.empty and "reaction_time_ttc_mean" in sub.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub,
            x="speed_kmh",
            y="reaction_time_ttc_mean",
            hue="cross_side",
            hue_order=["FarSide", "NearSide"],
            palette={"FarSide": "#49a7eb", "NearSide": "#efad73"},
            showfliers=False,
            linewidth=1.2,
            width=0.6
        )

        plt.legend(title="Crossing side", fontsize=11, title_fontsize=12)

        plt.ylabel("Reaction time [s]", fontsize=12)
        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_ttc.png"), dpi=200)
        plt.close()

    if not sub.empty and "reaction_time_sim_mean" in sub.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub,
            x="speed_kmh",
            y="reaction_time_sim_mean",
            hue="cross_side",
            hue_order=["FarSide", "NearSide"],
            palette={"FarSide": "#49a7eb", "NearSide": "#efad73"},
            showfliers=False,
            linewidth=1.2,
            width=0.6
        )

        plt.ylabel("Reaction time [s]", fontsize=12)
        plt.legend(title="Crossing side", fontsize=11, title_fontsize=12)

        plt.xlabel("Vehicle speed [km/h]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{scen}_reaction_time_sim.png"), dpi=200)
        plt.close()


print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
