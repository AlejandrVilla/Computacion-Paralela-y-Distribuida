#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <iostream>

/*
 * Structure describing a read-write lock.
 */
typedef struct rwlock_tag {
    pthread_mutex_t     mutex;
    pthread_cond_t      read;           /* wait for read */
    pthread_cond_t      write;          /* wait for write */
    int                 valid;          /* set when valid */
    int                 r_active;       /* readers active */
    int                 w_active;       /* writer active */
    int                 r_wait;         /* readers waiting */
    int                 w_wait;         /* writers waiting */
} rwlock_t;

#define RWLOCK_VALID    0xfacade

/*
 * Support static initialization of barriers
 */
#define RWL_INITIALIZER \
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, \
    PTHREAD_COND_INITIALIZER, RWLOCK_VALID, 0, 0, 0, 0}

/*
 * Define read-write lock functions
 */
extern int rwl_init (rwlock_t *rwlock);
extern int rwl_destroy (rwlock_t *rwlock);
extern int rwl_readlock (rwlock_t *rwlock);
extern int rwl_readtrylock (rwlock_t *rwlock);
extern int rwl_readunlock (rwlock_t *rwlock);
extern int rwl_writelock (rwlock_t *rwlock);
extern int rwl_writetrylock (rwlock_t *rwlock);
extern int rwl_writeunlock (rwlock_t *rwlock);

#ifdef DEBUG
# define DPRINTF(arg) printf arg
#else
# define DPRINTF(arg)
#endif

#define err_abort(code,text) do { \
    fprintf (stderr, "%s at \"%s\":%d: %s\n", \
        text, __FILE__, __LINE__, strerror (code)); \
    abort (); \
    } while (0)
#define errno_abort(text) do { \
    fprintf (stderr, "%s at \"%s\":%d: %s\n", \
        text, __FILE__, __LINE__, strerror (errno)); \
    abort (); \
    } while (0)

/*
 * Initialize a read-write lock
 */
int rwl_init (rwlock_t *rwl)
{
    int status;

    rwl->r_active = 0;
    rwl->r_wait = rwl->w_wait = 0;
    rwl->w_active = 0;
    status = pthread_mutex_init (&rwl->mutex, NULL);
    if (status != 0)
        return status;
    status = pthread_cond_init (&rwl->read, NULL);
    if (status != 0) {
        /* if unable to create read CV, destroy mutex */
        pthread_mutex_destroy (&rwl->mutex);
        return status;
    }
    status = pthread_cond_init (&rwl->write, NULL);
    if (status != 0) {
        /* if unable to create write CV, destroy read CV and mutex */
        pthread_cond_destroy (&rwl->read);
        pthread_mutex_destroy (&rwl->mutex);
        return status;
    }
    rwl->valid = RWLOCK_VALID;
    return 0;
}

/*
 * Destroy a read-write lock
 */
int rwl_destroy (rwlock_t *rwl)
{
    int status, status1, status2;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;

    /*
     * Check whether any threads own the lock; report "BUSY" if
     * so.
     */
    if (rwl->r_active > 0 || rwl->w_active) {
        pthread_mutex_unlock (&rwl->mutex);
        return EBUSY;
    }

    /*
     * Check whether any threads are known to be waiting; report
     * EBUSY if so.
     */
    if (rwl->r_wait != 0 || rwl->w_wait != 0) {
        pthread_mutex_unlock (&rwl->mutex);
        return EBUSY;
    }

    rwl->valid = 0;
    status = pthread_mutex_unlock (&rwl->mutex);
    if (status != 0)
        return status;
    status = pthread_mutex_destroy (&rwl->mutex);
    status1 = pthread_cond_destroy (&rwl->read);
    status2 = pthread_cond_destroy (&rwl->write);
    return (status == 0 ? status : (status1 == 0 ? status1 : status2));
}

/*
 * Handle cleanup when the read lock condition variable
 * wait is cancelled.
 *
 * Simply record that the thread is no longer waiting,
 * and unlock the mutex.
 */
static void rwl_readcleanup (void *arg)
{
    rwlock_t    *rwl = (rwlock_t *)arg;

    rwl->r_wait--;
    pthread_mutex_unlock (&rwl->mutex);
}

/*
 * Lock a read-write lock for read access.
 */
int rwl_readlock (rwlock_t *rwl)
{
    int status;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    if (rwl->w_active) {
        rwl->r_wait++;
        pthread_cleanup_push (rwl_readcleanup, (void*)rwl);
        while (rwl->w_active) {
            status = pthread_cond_wait (&rwl->read, &rwl->mutex);
            if (status != 0)
                break;
        }
        pthread_cleanup_pop (0);
        rwl->r_wait--;
    }
    if (status == 0)
        rwl->r_active++;
    pthread_mutex_unlock (&rwl->mutex);
    return status;
}

/*
 * Attempt to lock a read-write lock for read access (don't
 * block if unavailable).
 */
int rwl_readtrylock (rwlock_t *rwl)
{
    int status, status2;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    if (rwl->w_active)
        status = EBUSY;
    else
        rwl->r_active++;
    status2 = pthread_mutex_unlock (&rwl->mutex);
    return (status2 != 0 ? status2 : status);
}

/*
 * Unlock a read-write lock from read access.
 */
int rwl_readunlock (rwlock_t *rwl)
{
    int status, status2;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    rwl->r_active--;
    if (rwl->r_active == 0 && rwl->w_wait > 0)
        status = pthread_cond_signal (&rwl->write);
    status2 = pthread_mutex_unlock (&rwl->mutex);
    return (status2 == 0 ? status : status2);
}

/*
 * Handle cleanup when the write lock condition variable
 * wait is cancelled.
 *
 * Simply record that the thread is no longer waiting,
 * and unlock the mutex.
 */
static void rwl_writecleanup (void *arg)
{
    rwlock_t *rwl = (rwlock_t *)arg;

    rwl->w_wait--;
    pthread_mutex_unlock (&rwl->mutex);
}

/*
 * Lock a read-write lock for write access.
 */
int rwl_writelock (rwlock_t *rwl)
{
    int status;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    if (rwl->w_active || rwl->r_active > 0) {
        rwl->w_wait++;
        pthread_cleanup_push (rwl_writecleanup, (void*)rwl);
        while (rwl->w_active || rwl->r_active > 0) {
            status = pthread_cond_wait (&rwl->write, &rwl->mutex);
            if (status != 0)
                break;
        }
        pthread_cleanup_pop (0);
        rwl->w_wait--;
    }
    if (status == 0)
        rwl->w_active = 1;
    pthread_mutex_unlock (&rwl->mutex);
    return status;
}

/*
 * Attempt to lock a read-write lock for write access. Don't
 * block if unavailable.
 */
int rwl_writetrylock (rwlock_t *rwl)
{
    int status, status2;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    if (rwl->w_active || rwl->r_active > 0)
        status = EBUSY;
    else
        rwl->w_active = 1;
    status2 = pthread_mutex_unlock (&rwl->mutex);
    return (status != 0 ? status : status2);
}

/*
 * Unlock a read-write lock from write access.
 */
int rwl_writeunlock (rwlock_t *rwl)
{
    int status;

    if (rwl->valid != RWLOCK_VALID)
        return EINVAL;
    status = pthread_mutex_lock (&rwl->mutex);
    if (status != 0)
        return status;
    rwl->w_active = 0;
    if (rwl->r_wait > 0) {
        status = pthread_cond_broadcast (&rwl->read);
        if (status != 0) {
            pthread_mutex_unlock (&rwl->mutex);
            return status;
        }
    } else if (rwl->w_wait > 0) {
        status = pthread_cond_signal (&rwl->write);
        if (status != 0) {
            pthread_mutex_unlock (&rwl->mutex);
            return status;
        }
    }
    status = pthread_mutex_unlock (&rwl->mutex);
    return status;
}

struct list_node_s{
    int data;
    struct list_node_s* next;
};

// main

void* test1(void*);
void* test2(void*);
int member(int value);
int insert(int value);
int Delete(int value);
void print();
void Delete_list();

struct list_node_s* head_p = NULL;
pthread_t* thread_pool;
rwlock_t rwl;
int thread_count;

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];
    rwl_init(&rwl);

    for(int i=0 ; i<1000 ; ++i)
        insert(i);

    double startTime, endTime;
    srand(time(NULL));
    startTime = omp_get_wtime(); 
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, test1, NULL);

    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime(); 
    std::cout<<"Tiempo test1: "<<(endTime - startTime)<<"s\n";

    srand(time(NULL));
    startTime = omp_get_wtime(); 
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, test2, NULL);

    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime(); 
    std::cout<<"Tiempo test2: "<<(endTime - startTime)<<"s\n";
    
    rwl_destroy(&rwl);
    Delete_list();
}

void* test1(void*)
{
    int n=99900;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        rwl_readlock(&rwl);
        member(value);
        rwl_readunlock(&rwl);
    }

    n=50;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        rwl_writelock(&rwl);
        insert(value);
        rwl_writeunlock(&rwl);
    }

    n=50;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        rwl_writelock(&rwl);
        Delete(value);
        rwl_writeunlock(&rwl);
    }

    return NULL;
}

void* test2(void*)
{
    int n=80000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        rwl_readlock(&rwl);
        member(value);
        rwl_readunlock(&rwl);
    }

    n=10000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        rwl_writelock(&rwl);
        insert(value);
        rwl_writeunlock(&rwl);
    }

    n=10000;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        rwl_writelock(&rwl);
        Delete(value);
        rwl_writeunlock(&rwl);
    }

    return NULL;
}

int member(int value)
{
    struct list_node_s* curr_p = head_p;

    while(curr_p != NULL && curr_p->data < value)
        curr_p = curr_p->next;
    
    if(curr_p == NULL || curr_p->data > value)
    {
        return 0;
    }
    return 1;
}

int insert(int value)
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p == NULL || curr_p->data > value)
    {
        temp_p = new struct list_node_s;
        temp_p->data = value;
        temp_p->next = curr_p;
        if(pred_p == NULL)
            head_p = temp_p;
        else
            pred_p->next = temp_p;
        return 1;
    }
    return 0;
}

int Delete(int value)
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p != NULL && curr_p->data == value)
    {
        if(pred_p == NULL)
        {
            head_p = curr_p->next;
            delete curr_p;
        }
        else
        {
            pred_p->next = curr_p->next;
            delete curr_p;
        }
        return 1;
    }
    return 0;
}

void print()
{
    struct list_node_s* curr_p = head_p;

    while(curr_p != NULL)
    {
        std::cout<<curr_p->data<<"->";
        curr_p=curr_p->next;
    }
    std::cout<<'\n';
}

void Delete_list()
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;

    while(curr_p != NULL)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
        delete pred_p;
    }
    head_p = NULL;
}