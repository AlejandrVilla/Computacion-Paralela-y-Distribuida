#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <omp.h>
#include <iostream>

int counter, thread_count;
pthread_mutex_t mutex;
pthread_cond_t cond_var;

pthread_t* thread_pool;

void* thread_work(void*);

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_var, NULL);
    counter=0;
    double startTime, endTime;

    startTime = omp_get_wtime();
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, thread_work, NULL);
    
    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime();
    std::cout<<"Tiempo: "<<endTime-startTime<<'\n';   

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_var);
    delete[] thread_pool;
}

void* thread_work(void*)
{

    // Barrier
    pthread_mutex_lock(&mutex); // bloquea mutex
    counter++;
    if(counter == thread_count)
    {
        counter=0;
        pthread_cond_broadcast(&cond_var);      // despierta todos los hilos
    }
    else
    {
        while(pthread_cond_wait(&cond_var, &mutex) != 0);   // duerme un hilo y desbloque el mutex
    }
    pthread_mutex_unlock(&mutex);   // desbloque mutex

    return NULL;
}