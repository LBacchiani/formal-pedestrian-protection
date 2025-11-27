#!/bin/bash

SCRIPT="test1.py"
ITER=20
SPEEDS=(30 50)

echo "🔁 Avvio loop_run.sh (PID principale: $$)"
echo "   → Puoi interrompere tutto in qualsiasi momento con: kill -INT $$"
echo

for SPEED in "${SPEEDS[@]}"; do

    if [[ $SPEED -eq 50 ]]; then
        WAIT_TIME=16
    elif [[ $SPEED -eq 40 ]]; then
        WAIT_TIME=17
    elif [[ $SPEED -eq 30 ]]; then
        WAIT_TIME=18
    else
        WAIT_TIME=19
    fi

    echo "Avvio batch da $ITER run a ${SPEED} km/h (timeout: ${WAIT_TIME}s)"
    echo "=============================================="

    for ((i=1; i<=ITER; i++)); do
        echo "▶️  Esecuzione $i di $ITER (velocità: ${SPEED} km/h, timeout ${WAIT_TIME}s)..."
        echo "🔁 Avvio loop_run.sh (PID principale: $$)"

        timeout -s INT $WAIT_TIME python3 "$SCRIPT" --speed "$SPEED"
        STATUS=$?

        if [[ $STATUS -eq 124 ]]; then
            echo "⏹️  Timeout raggiunto: script interrotto dopo ${WAIT_TIME}s"
        else
            echo "✅ Script terminato con codice $STATUS"
        fi

        echo "----------------------------------"
    done

    echo "🏁 Completate tutte le $ITER esecuzioni per ${SPEED} km/h"
    echo
done

echo "✅ Tutti i test (25, 40, 50 km/h) completati."
