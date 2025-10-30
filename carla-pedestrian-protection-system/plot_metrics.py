import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS_PATH = "metrics_log.jsonl"

MIN_DIST = 3.0

def load_brake_data():
    """Load all braking records (including mild brakes) from the metrics log."""
    records = []
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    # Flatten d_min_records and mild_brake_records if present
    brake_records = []
    for r in records:
        scenario_type = "DAY" if r["scenario"]["weather"]["sun_altitude_angle"] >= 0 else "NIGHT"
        ped_mode = r["scenario"]["pedestrian_mode"]

        for ev in r.get("d_min_records", []):
            brake_records.append({
                "timestamp": ev.get("timestamp"),
                "run_id": r["run_id"],
                "action": ev.get("action", "brake"),
                "d_min": ev.get("d_min"),
                "stopping_distance": ev.get("stopping_distance"),
                "speed_before_brake": ev.get("speed_before_brake"),
                "scenario_type": scenario_type,
                "ped_mode": ped_mode
            })

        for ev in r.get("mild_brake", []):
            brake_records.append({
                "timestamp": ev.get("timestamp"),
                "run_id": r["run_id"],
                "action": "mild_brake",
                "d_min": None,
                "stopping_distance": ev.get("distance_travelled"),
                "speed_before_brake": ev.get("speed_before_mild_brake"),
                "scenario_type": scenario_type,
                "ped_mode": ped_mode
            })

    df = pd.DataFrame(brake_records)
    df = df.sort_values("timestamp")
    return df


# === Conteggio azioni di frenata ===
def plot_brake_counts(df):
    """Bar chart for number of mild_brake, brake, and emergency_brake events."""
    counts = df["action"].value_counts().reset_index()
    counts.columns = ["action", "count"]

    # Imposta l’ordine desiderato
    order = ["mild_brake", "brake", "emergency_brake"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=counts,
        x="action",
        y="count",
        order=order,
        palette="coolwarm"
    )
    ax.set_title("Number of Brake Events by Type")
    ax.set_xlabel("Action Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()



# ===  Distanze nel tempo ===
def plot_box_by_run(df):
    """
    Boxplot of braking distances (merged brake + emergency_brake)
    grouped by simulation run.
    """
    # Filtra solo brake + emergency_brake
    df_box = df[df["action"].isin(["brake", "emergency_brake"])].copy()
    df_box = df_box[df_box["d_min"].notnull()]
    if df_box.empty:
        print("No braking distance data found for brake/emergency_brake.")
        return

    # Ordina le run in base al tempo medio per avere X più leggibile
    run_order = (
        df_box.groupby("run_id")["timestamp"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    sns.set_style("whitegrid")
    plt.figure(figsize=(max(8, len(run_order) * 0.6), 5))
    ax = sns.boxplot(
        data=df_box,
        x="run_id",
        y="d_min",
        order=run_order,
        color="#4C72B0",   # blu coerente
        width=0.6
    )

    # Calcola la media per ciascuna run e aggiungi overlay
    mean_dmin = (
        df_box.groupby("run_id")["d_min"]
        .mean()
        .reindex(run_order)
        .values
    )
    plt.scatter(
        range(len(run_order)),
        mean_dmin,
        color="black",
        s=60,
        marker="o",
        label="Mean per run"
    )

    # Linea di sicurezza a 1 metro
    plt.axhline(MIN_DIST, color="red", linestyle="--", label=f"{MIN_DIST}m threshold")

    plt.title("Braking Distance Distribution per Run (Brake + Emergency)")
    plt.xlabel("Simulation Run ID")
    plt.ylabel("Minimum Distance (m)")

    # Abbrevia run_id
    ax.set_xticklabels([rid[:6] for rid in run_order], rotation=45, ha="right")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()




# ===  Stopping distance vs speed ===
def plot_stopdist_vs_speed(df):
    """Scatter plot showing how stopping distance relates to initial speed."""
    df_filtered = df[df["stopping_distance"].notnull() & df["speed_before_brake"].notnull()]
    if df_filtered.empty:
        print("No stopping distance data found.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_filtered,
        x="speed_before_brake",
        y="stopping_distance",
        hue="action",
        style="scenario_type",
        palette="viridis",
        s=60
    )
    plt.title("Stopping Distance vs. Speed Before Brake")
    plt.xlabel("Speed before brake (m/s)")
    plt.ylabel("Stopping distance (m)")
    plt.tight_layout()
    plt.show()


# === (Extra) Boxplot per scenario ===
def plot_box_by_scenario(df):
    """Boxplot of d_min grouped by scenario type and action."""
    df_filtered = df[df["d_min"].notnull()]
    if df_filtered.empty:
        print("No minimum distance data found.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_filtered, x="scenario_type", y="d_min", hue="action", palette="Set2")
    plt.axhline(MIN_DIST, color="red", linestyle="--", label=f"{MIN_DIST}m threshold")
    plt.title("Minimum Distance Distribution by Scenario and Brake Type")
    plt.xlabel("Scenario Type")
    plt.ylabel("Minimum Distance (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_brake_data()
    print(f"Loaded {len(df)} total braking records.")
    plot_brake_counts(df)
    plot_box_by_run(df)
    plot_stopdist_vs_speed(df)
    plot_box_by_scenario(df)
