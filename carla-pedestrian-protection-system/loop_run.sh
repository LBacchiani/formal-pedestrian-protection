#!/bin/bash

SCRIPT="test2.py"
ITER=100
WAIT_TIME=15
SPEEDS=(25 40 50)

for SPEED in "${SPEEDS[@]}"; do
    echo "🚗 Avvio batch da $ITER run a ${SPEED} km/h"
    echo "=============================================="

    for ((i=1; i<=ITER; i++)); do
        echo "▶️  Esecuzione $i di $ITER (velocità: ${SPEED} km/h, timeout ${WAIT_TIME}s)..."

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
