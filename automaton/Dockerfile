FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    mosquitto mosquitto-clients \
    libfreetype6 \
    libpng16-16 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY automaton.py vehicle_dt.py props.py z3_automaton.py test.py /app/

RUN pip install --no-cache-dir \
    z3-solver \
    matplotlib \
    paho-mqtt

EXPOSE 1883

CMD mosquitto & \
    python3 vehicle_dt.py
