#!/bin/bash

SCRIPTS=("test4_s.py")

ITER=50

echo "Avvio loop multiplo (PID principale: $$)"
echo "   → Puoi interrompere tutto in qualsiasi momento con: kill -INT $$"
echo

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "=============================="
    echo "Avvio test per script: $SCRIPT"
    echo "=============================="

    if [[ "$SCRIPT" == "test4_o.py" ]]; then
        SPEEDS=(10 20)
    elif [[ "$SCRIPT" == "test4_s.py" ]]; then
        SPEEDS=(10)
    else
        SPEEDS=(30 50)
    fi

    for SPEED in "${SPEEDS[@]}"; do
        if [[ $SPEED -eq 50 ]]; then
            WAIT_TIME=15
        elif [[ $SPEED -eq 40 ]]; then
            WAIT_TIME=17
        elif [[ $SPEED -eq 30 ]]; then
            WAIT_TIME=18
        elif [[ $SPEED -eq 20 ]]; then
            WAIT_TIME=19
        else
            WAIT_TIME=20
        fi

        echo "🚗 Avvio batch da $ITER run a ${SPEED} km/h per $SCRIPT (timeout: ${WAIT_TIME}s)"
        echo "----------------------------------------------"

        for ((i=1; i<=ITER; i++)); do
            echo "Esecuzione $i di $ITER (velocità: ${SPEED} km/h, timeout ${WAIT_TIME}s)... (PID principale: $$)"
            timeout -s INT $WAIT_TIME python3 "$SCRIPT" --speed "$SPEED"
            STATUS=$?

            if [[ $STATUS -eq 124 ]]; then
                echo "Timeout raggiunto dopo ${WAIT_TIME}s"
            else
                echo "Script terminato con codice $STATUS"
            fi
            echo "----------------------------------"
        done

        echo "🏁 Completate tutte le $ITER esecuzioni per ${SPEED} km/h in $SCRIPT"
        echo
    done
done

echo "Tutti i test completati con successo."
