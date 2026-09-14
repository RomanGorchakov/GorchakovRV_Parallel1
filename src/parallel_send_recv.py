#!/usr/bin/env python3
"""
Параллельная версия умножения матрицы на вектор с использованием Send/Recv.
Основана на лекционном материале (стр. 7-12).
"""
from mpi4py import MPI
import numpy as np
import time

def read_params(filename="data/in.dat"):
    # TODO: Реализуйте чтение параметров
    pass

def auxiliary_arrays_determination(M, numprocs):
    """Расчет rcounts и displs для разбиения данных"""
    # TODO: Реализуйте расчет массивов для произвольного M
    pass

def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
    # TODO: Реализуйте:
    # 1. Чтение параметров на процессе 0
    # 2. Рассылку параметров через Bcast
    # 3. Распределение матрицы через Send/Recv
    # 4. Распределение вектора через Bcast
    # 5. Локальное вычисление
    # 6. Сбор результатов через Send/Recv
    # 7. Запись результата
    
    pass

if __name__ == "__main__":
    main()