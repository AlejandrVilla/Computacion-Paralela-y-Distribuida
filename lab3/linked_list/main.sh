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

    rm one_mutex_list.out one_per_node.out rw_lock.out

    # compilacion
    g++ LL_one_mutex_list.cpp -o one_mutex_list.out -lpthread -fopenmp
    g++ LL_one_mutex_per_node.cpp -o one_per_node.out -lpthread -fopenmp
    g++ LL_rw_lock.cpp -o rw_lock.out -lpthread -fopenmp

    echo "Read-Write locks"
    ./rw_lock.out $N_THREAD
    echo "One mutex for entire list"
    ./one_mutex_list.out $N_THREAD
    echo "One mutex per node"
    ./one_per_node.out $N_THREAD
fi