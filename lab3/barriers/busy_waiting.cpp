#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <iostream>

int counter;
int thread_count;
pthread_mutex_t barrier_mutex;
pthread_t* thread_pool;

void* thread_work(void*);

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];
    pthread_mutex_init(&barrier_mutex, NULL);
    double startTime, endTime;
    counter=0;

    startTime = omp_get_wtime();
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, thread_work, NULL);
    
    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime();
    std::cout<<"Tiempo: "<<endTime-startTime<<'\n';   

    pthread_mutex_destroy(&barrier_mutex);
    delete[] thread_pool;
}

void* thread_work(void*)
{
    // Barrier
    pthread_mutex_lock(&barrier_mutex);     // bloquea mutex
    counter++;
    pthread_mutex_unlock(&barrier_mutex);   // desbloquea mutex
    while(counter < thread_count);

    return NULL;
}