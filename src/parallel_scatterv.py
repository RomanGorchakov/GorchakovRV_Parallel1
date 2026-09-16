#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import time

def read_params(filename="data/in.dat"):
    # Чтение параметров
    f1 = open('data/in.dat', 'r')
    N = int(f1.readline().strip())
    M = int(f1.readline().strip())
    f1.close()
    
    return np.array(N, dtype=np.int32), np.array(M, dtype=np.int32)

def auxiliary_arrays_determination(M, numprocs):
    """Расчет rcounts и displs для разбиения данных"""
    rcounts = np.zeros(numprocs, dtype=np.int32)
    displs = np.zeros(numprocs, dtype=np.int32)

    if numprocs > 1:
        ave, res = divmod(M, numprocs - 1)
        # Процесс 0 не участвует в вычислениях
        rcounts[0] = 0
        displs[0] = 0
        for k in range(1, numprocs):
            if k <= res:
                rcounts[k] = ave + 1
            else:
                rcounts[k] = ave
            displs[k] = displs[k-1] + rcounts[k-1]
    else:
        # Если запущен всего 1 процесс, он делает всю работу
        rcounts[0] = M
        displs[0] = 0

    return rcounts, displs

def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
    # Чтение параметров на процессе 0
    N = np.array(0, dtype=np.int32)
    M = np.array(0, dtype=np.int32)

    if rank == 0:
        N, M = read_params("data/in.dat")

    # Рассылка параметров через Bcast
    comm.Bcast([N, 1, MPI.INT], root=0)

    # Определение вспомогательных массивов для Scatterv на процессе 0
    if rank == 0:
        rcounts, displs = auxiliary_arrays_determination(int(M), numprocs)
        sendbuf_M = [rcounts, np.ones(numprocs, dtype=np.int32), np.arange(numprocs, dtype=np.int32), MPI.INT]
    else:
        rcounts = None
        displs = None
        sendbuf_M = None

    # Распределение количества строк для каждого процесса
    M_part = np.array(0, dtype=np.int32)
    comm.Scatterv(sendbuf_M, [M_part, 1, MPI.INT], root=0)

    # Выделение памяти под локальную часть матрицы A
    A_part = np.empty((int(M_part), int(N)), dtype=np.float64)
    
    # Распределение матрицы через Scatterv
    if rank == 0:
        # Чтение матрицы A на процессе 0
        f2 = open('data/AData.dat', 'r')
        A = np.empty((int(M), int(N)), dtype=np.float64)
        for j in range(int(M)):
            for i in range(int(N)):
                A[j, i] = np.float64(f2.readline())
        f2.close()

        sendbuf_A = [A, rcounts * int(N), displs * int(N), MPI.DOUBLE]
    else:
        sendbuf_A = None

    comm.Scatterv(sendbuf_A, [A_part, int(M_part * N), MPI.DOUBLE], root=0)
    
    # Распределение вектора через Bcast
    x = np.empty(int(N), dtype=np.float64)
    if rank == 0:
        f3 = open('data/xData.dat', 'r')
        for i in range(int(N)):
            x[i] = np.float64(f3.readline())
        f3.close()

    comm.Bcast([x, int(N), MPI.DOUBLE], root=0)

    # Замер времени начала вычислений
    comm.Barrier()
    start_time = time.time()
    
    # Локальное вычисление
    b_part = np.dot(A_part, x)

    comm.Barrier()
    
    # Сбор результатов через Gatherv
    if rank == 0:
        b = np.empty(int(M), dtype=np.float64)
        recvbuf_b = [b, rcounts, displs, MPI.DOUBLE]
    else:
        b = None
        recvbuf_b = None

    comm.Gatherv([b_part, int(M_part), MPI.DOUBLE], recvbuf_b, root=0)
    
    # Запись результата
    if rank == 0:
        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")

        f4 = open('Results_parallel_scatterv.dat', 'w')
        for j in range(M):
            print(b[j], file=f4)
        f4.close()
    
    pass

if __name__ == "__main__":
    main()