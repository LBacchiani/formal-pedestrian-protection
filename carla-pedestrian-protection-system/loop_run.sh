#!/bin/bash

SCRIPT="test2.py"
ITER=100
WAIT_TIME=15

for ((i=1; i<=ITER; i++)); do
    echo "▶️  Esecuzione $i di $ITER (foreground, timeout ${WAIT_TIME}s)..."
    
    timeout -s INT $WAIT_TIME python3 "$SCRIPT"
    STATUS=$?
    
    if [[ $STATUS -eq 124 ]]; then
        echo "⏹️  Timeout raggiunto: script interrotto dopo ${WAIT_TIME}s"
    else
        echo "✅ Script terminato con codice $STATUS"
    fi
    
    echo "----------------------------------"
done

echo "🏁 Tutte le $ITER esecuzioni completate."
