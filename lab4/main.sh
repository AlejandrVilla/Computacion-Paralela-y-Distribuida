#!/bin/bash

if [ -z $1 ]; then
    echo "Para ayuda:"
    echo "./main.sh -h "
elif [ "-h" = "$1" ]; then
    echo "Para compilar:"
    echo "g++ [nombre].cpp -o [nombre].out -fopenmp"
    echo "Para ejecutar:"
    echo -e "./[nombre].out [numero threads]\n"
    echo "Para compilar y ejecutar:"
    echo "./main.sh [numero de threads"]
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    rm odd-even-sort.out
    g++ odd-even-sort.cpp -o odd-even-sort.out -fopenmp
    ./odd-even-sort.out $1
fi