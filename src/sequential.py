#!/usr/bin/env python3
"""
Последовательная версия умножения матрицы на вектор.
Основана на лекционном материале (стр. 5-6).
"""
import numpy as np
import time

def read_params(filename="data/in.dat"):
    """Чтение размеров матрицы из файла"""
    # TODO: Реализуйте чтение N и M из файла
    f1 = open('data/in.dat', 'r')
    N = int(f1.readline())
    M = int(f1.readline())
    f1.close()
    
    return N, M

def read_matrix(filename="data/AData.dat", M=None, N=None):
    """Чтение матрицы из файла"""
    # TODO: Реализуйте чтение матрицы A
    A = np.empty((M, N))
    x = np.empty(N)
    b = np.empty(M)
    
    f2 = open('data/AData.dat', 'r')
    for j in range(M):
        for i in range(N):
            A[j, i] = float(f2.readline())
    f2.close()
    return A

def read_vector(filename="data/xData.dat"):
    """Чтение вектора из файла"""
    # TODO: Реализуйте чтение вектора x
    f3 = open('data/xData.dat', 'r')
    x = np.array([float(line) for line in f3 if line.strip()])
    f3.close()
    return x

def sequential_mat_vec_mult(A, x):
    """Последовательное умножение матрицы на вектор"""
    # TODO: Реализуйте умножение матрицы A на вектор x
    M, N = A.shape
    b = np.empty(M)
    for j in range(M):
        b[j] = 0
        for i in range(N):
            b[j] += A[j, i] * x[i]
    return b

def main():
    """Основная функция"""
    # TODO: Реализуйте основную логику программы
    N, M = read_params(filename="data/in.dat")
    A = read_matrix(filename="data/AData.dat", M=M, N=N)
    x = read_vector(filename="data/xData.dat")
    
    start_time = time.time()
    b = sequential_mat_vec_mult(A, x)
    elapsed_time = time.time() - start_time
    
    print(f"Вычисления завершены за {elapsed_time:.6f} секунд.")
    
    f4 = open('Results_sequential.dat', 'w')
    for j in range(M):
        print(b[j], file=f4)
    f4.close()

if __name__ == "__main__":
    main()