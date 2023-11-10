#!/bin/bash

if [ $# = "0" ]; then
    echo "Para ayuda:"
    echo "./main --help"
elif [ $1 = "--help" ]; then
    echo "Para ingresar numero de threads:"
    echo "./main.sh [n_threads]"
else
    N_THREAD=$1
    echo -e "Numero de threads $N_THREAD\n"

    rm busy_waiting.out condi_variables.out semaphores.out

    # compilacion
    g++ busy_waiting.cpp -o busy_waiting.out -lpthread -fopenmp
    g++ condi_variables.cpp -o condi_variables.out -lpthread -fopenmp
    g++ semaphores.cpp -o semaphores.out -lpthread -fopenmp

    echo "Busy waiting"
    ./busy_waiting.out $N_THREAD
    echo "Condition variables"
    ./condi_variables.out $N_THREAD
    echo "Semaphores"
    ./semaphores.out $N_THREAD
fi