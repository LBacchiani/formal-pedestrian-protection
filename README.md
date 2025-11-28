# Formal-pedestrian-protection
# Hybrid Automaton Digital Twin & CARLA Pedestrian Protection System

This repository is organized into two major components:

1. **Digital Twin + Hybrid Automaton (DT/HA)** — located in the `automaton/` directory  
2. **CARLA Pedestrian Protection System** — located in `carla-pedestrian-protection-system/`

These two parts work together to evaluate pedestrian-related Euro NCAP AEB scenarios using a hybrid digital-twin architecture and a rule-based automaton for braking decisions.

---

## 1. Digital Twin & Hybrid Automaton & Formal definition

This module contains:

- **A formal definition** of the Hybrid Automaton governing vehicle longitudinal braking logic.
- **A runnable Digital Twin environment**, deployable via Docker, that:
  - Listens to CARLA-generated sensor data via MQTT
  - Performs state-transition evaluation (mild brake → brake → emergency brake)
  - Publishes braking commands back to the simulation
  - Logs events & formal guarantees (reaction times, transitions, stop conditions)

### Running the Digital Twin via Docker

From inside the `automaton/` folder:

```bash
docker build -t automaton-dt .
docker run -it automaton-dt
```

The container includes:
- Python 3.9
- `z3-solver` for formal reasoning
- Mosquitto MQTT broker
- The Hybrid Automaton runtime (`vehicle_dt.py`, `automa.py`)

The Digital Twin must be running **before** launching any CARLA test scenario.

---

## 🚗 2. CARLA Pedestrian Protection System (`carla-pedestrian-protection-system/`)

This folder contains the full scenario runner pipeline for evaluating AEB pedestrian scenarios.

### Requirements
- **CARLA Simulator ≥ 0.9.15**
- **Python 3.8**
- **NVIDIA GPU** (mandatory for YOLOv8-based pedestrian detection)
- **CUDA-compatible drivers**
- **Linux operative system**

### Creating the Anaconda Environment (Python 3.8)

```bash
conda create -n carla-aeb python=3.8
conda activate carla-aeb
pip install -r requirements.txt

```
---
## Test Scenarios

The repository includes multiple scenario scripts (`test1.py`, `test2.py`, etc.), each corresponding to a Euro NCAP pedestrian test case:

### Test Scenario Matrix

| Script        | Scenario Type | Direction      | Expected Speeds   |
|--------------|----------------|----------------|--------------------|
| `test1.py`   | CPFA           | Perpendicular  | **30 / 50 km/h**  |
| `test2.py`   | CPNA           | Perpendicular  | **30 / 50 km/h**  |
| `test3.py`   | CPNCO          | Perpendicular  | **30 / 50 km/h**  |
| `test4_o.py` | CPTA           | Opposite       | **10 / 20 km/h**  |
| `test4_s.py` | CPTA           | Same           | **10 km/h**       |


### Running a Scenario

Example:

```bash
python test1.py --speed 50
```

All scripts accept:

```
--speed <value>
```

and automatically save:
- JSONL logs (metrics, timings, detection events)
- Scenario results into `resultsX/` and `logs/`

---

## Batch Execution

You can execute multiple full-scenario runs automatically using:

```bash
./loop_run.sh
./loop_all_run.sh
```

These scripts cycle through all tests at all configured speeds and create separate result directories.

---

## Output Structure

- `logs/` — simulation Metrics  
- `results1/`, `results2/`, `results3/`, `results4/` — aggregated metrics for each test batch  
- `plot_metrics_tests.py` — plotting utilities for post-processing  

---


## Repository Structure (Simplified)

```
.
├── automaton/
│   ├── Dockerfile
│   ├── automa.py
│   ├── vehicle_dt.py
│   ├── mosquitto.conf
│   └── formal_hybrid_automaton_definition/
│
└── carla-pedestrian-protection-system/
    ├── test1.py ... test4_s.py
    ├── loop_all_run.sh
    ├── loop_run.sh
    ├── results*/ 
    ├── logs/
    ├── agents/
    ├── SyncSimulation.py
    └── yolov8n.pt
```

---

