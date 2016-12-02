/*--------------------------------------------------------------------------------------------------
 * Author:      Dan Cassidy
 * Date:        2016-10-31
 * Assignment:  Homework 3
 * Source File: main.c
 * Language:    C
 * Course:      CSCI-B 424 Parallel Programming
 * Purpose:     Loops over a specified number of MPI ranks, incrementing the message as it goes.
--------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <mpi.h>

/* Set the number of loops the program will make. */
const int NUM_LOOPS = 10;

int main(int argc, char * argv[])
{
    /* Declare variables. */
    int num_ranks = 0, my_rank = 0, message = 1;

    /* Initialize MPI. */
    MPI_Init(&argc, &argv);

    /* Get the number of ranks and the rank number of the process. */
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Have rank 0 start the initial loop. */
    if (my_rank == 0)
    {
        printf("Looping %d times across %d ranks.\n", NUM_LOOPS, num_ranks);
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0: Sent initial message of '%d'.\n", message);
    }
    
    /* Loop NUM_LOOPS times. */
    while ((message - 1) / num_ranks < NUM_LOOPS - 1)
    {
        /* Receive the message. */
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* Increment the message. */
        message++;

        /* Send the next message. */
        MPI_Send(&message, 1, MPI_INT, message % num_ranks, 0, MPI_COMM_WORLD);
    }

    /* Have rank 0 collect the last message sent and complete the final loop. */
    if (my_rank == 0)
    {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        printf("Rank 0: Received final message of '%d'.\n", message);
    }

    /* Finalize MPI. */
    MPI_Finalize();

    return 0;
}