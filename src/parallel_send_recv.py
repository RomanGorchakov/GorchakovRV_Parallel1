#!/usr/bin/env python3
"""
Параллельная версия умножения матрицы на вектор с использованием Send/Recv.
Основана на лекционном материале (стр. 7-12).
"""
from mpi4py import MPI
import numpy as np
import time

def read_params(filename="data/in.dat"):
    # Чтение параметров
    f1 = open('data/in.dat', 'r')
    N = int(f1.readline().strip())
    M = int(f1.readline().strip())
    f1.close()
    
    return N, M

def auxiliary_arrays_determination(M, numprocs):
    """Расчет rcounts и displs для разбиения данных"""
    num_workers = numprocs - 1 if numprocs > 1 else 1
    base = M // num_workers
    rem = M % num_workers

    rcounts = np.array([base + (1 if i < rem else 0) for i in range(num_workers)], dtype=np.int32)
    displs = np.array([sum(rcounts[:i]) for i in range(num_workers)], dtype=np.int32)

    return rcounts, displs

def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
    start_time = None
    
    # Чтение параметров на процессе 0
    if rank == 0:
        start_time = time.time()
        N_val, M_val = read_params("data/in.dat")
        params = np.array([N_val, M_val], dtype=np.int32)
    else:
        params = np.empty(2, dtype=np.int32)
    
    # Рассылка параметров через Bcast
    comm.Bcast([params, 2, MPI.INT], root=0)
    N, M = params[0], params[1]
    
    # Определение схемы распределения нагрузки
    num_workers = numprocs - 1 if numprocs > 1 else 1
    rcounts, displs = auxiliary_arrays_determination(M, numprocs)
    
    # Распределение матрицы через Send/Recv
    if rank == 0:
        f2 = open("data/AData.dat", "r")

        if numprocs > 1:
            # Чтение и отправление подматрицы для каждого рабочего процесса
            for k in range(1, numprocs):
                local_m = rcounts[k - 1]
                A_part = np.empty((local_m, N), dtype=np.float64)
                for j in range(local_m):
                    for i in range(N):
                        A_part[j, i] = float(f2.readline())
                comm.Send([A_part, MPI.DOUBLE], dest=k, tag=0)
            f2.close()
            A_part = None
        else:
            A_part = np.empty((M, N), dtype=np.float64)
            for j in range(M):
                for i in range(N):
                    A_part[j, i] = float(f2.readline())
            f2.close()

    else:
        # Рабочие процессы получают свою часть матрицы A
        local_m = rcounts[rank - 1]
        A_part = np.empty((local_m, N), dtype=np.float64)
        comm.Recv([A_part, MPI.DOUBLE], source=0, tag=0)
    
    # Распределение вектора через Bcast
    x = np.empty(N, dtype=np.float64)
    
    if rank == 0:
        f3 = open('data/xData.dat', 'r')
        for i in range(N):
            x[i] = float(f3.readline())
        f3.close()

    comm.Bcast([x, MPI.DOUBLE], root=0)
    
    # Локальное вычисление
    if numprocs > 1:
        if rank != 0:
            b_part = np.dot(A_part, x)
    else:
        b_part = np.dot(A_part, x)
        
    # Сбор результатов через Send/Recv
    if rank == 0:
        b = np.empty(M, dtype=np.float64)

        if numprocs > 1:
            for k in range(1, numprocs):
                local_m = rcounts[k - 1]
                offset = displs[k - 1]
                comm.Recv([b[offset : offset + local_m], MPI.DOUBLE], source=k, tag=0)
        else:
            b = b_part

    else:
        comm.Send([b_part, MPI.DOUBLE], dest=0, tag=0)
        
    # Запись результата
    if rank == 0:
        relapsed_time = time.time() - start_time
        print(f"Вычисления завершены за {relapsed_time} секунд.")

        f4 = open('Results_parallel_send_recv.dat', 'w')
        for j in range(M):
            print(b[j], file=f4)
        f4.close()

if __name__ == "__main__":
    main()