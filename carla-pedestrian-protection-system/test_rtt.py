import json
import os
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "./logs/mqtt_times.jsonl"

def safe_extend(target_list, value):
    if isinstance(value, list):
        target_list.extend(value)

def stats(name, data):
    arr = np.array(data)
    if len(arr) == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "count": len(arr),
    }

def main():
    reception_all = []
    automaton_all = []
    return_all = []

    if not os.path.exists(INPUT_FILE):
        print(f"Errore: file non trovato: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                continue

            times = data.get("times", {})
            safe_extend(reception_all, times.get("reception_time", []))
            safe_extend(automaton_all, times.get("automaton_time", []))
            safe_extend(return_all, times.get("return_time", []))

    s_rec = stats("reception_time", reception_all)
    s_aut = stats("automaton_time", automaton_all)
    s_ret = stats("return_time", return_all)

    mean_rec = s_rec["mean"]
    mean_aut = s_aut["mean"]
    mean_ret = s_ret["mean"]
    total_time = mean_rec + mean_aut + mean_ret

    print("\n=== STATISTICAL RESULTS OF MQTT TIMES ===\n")

    def print_stats(label, s):
        print(f"{label}:")
        print(f"    media: {s['mean']*1000:.3f} ms")
        print(f"    min:   {s['min']*1000:.3f} ms")
        print(f"    max:   {s['max']*1000:.3f} ms")
        print(f"    std:   {s['std']*1000:.3f} ms")
        print(f"    n:     {s['count']}")
        print()

    print_stats("Reception_time  (sensors → automaton)", s_rec)
    print_stats("Automaton_time (processing)", s_aut)
    print_stats("Return_time     (automaton → sensors)", s_ret)

    print("-----------------------------------------")
    print(f"Average total time (round trip): {total_time*1000:.3f} ms")
    print("-----------------------------------------\n")

if __name__ == "__main__":
    main()
