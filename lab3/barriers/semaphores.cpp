#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <omp.h>
#include <iostream>

int counter, thread_count;
sem_t count_sem;
sem_t barrier_sem;

pthread_t* thread_pool;

void* thread_work(void*);

int main(int argc, char* argv[])
{
    // puntero al semaforo, compartir con otro hilo, valor inicial
    sem_init(&count_sem, 0, 1);
    sem_init(&barrier_sem, 0, 0);

    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];

    double startTime, endTime;

    startTime = omp_get_wtime();
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, thread_work, NULL);
    
    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime();
    std::cout<<"Tiempo: "<<endTime-startTime<<'\n';   

    sem_destroy(&count_sem);
    sem_destroy(&barrier_sem);
    delete[] thread_pool;
}

// sem_wait -> si el valor del semaforo es 0 el hilo espera, si no es 0, decrementa en 1 y el hilo avanza
// sem_post -> incrementa el valor del semaforo en 1 y el hilo avanza

void* thread_work(void*)
{
    // Barrier
    sem_wait(&count_sem);
    if(counter == thread_count-1)
    {
        counter = 0;
        sem_post(&count_sem);
        for(int j=0; j<thread_count-1; ++j)
            sem_post(&barrier_sem);
    }
    else
    {
        counter++;
        sem_post(&count_sem);
        sem_wait(&barrier_sem);
    }

    return NULL;
}