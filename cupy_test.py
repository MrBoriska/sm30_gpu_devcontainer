import numpy as np
import cupy as cp
import time
import tqdm
# Создание случайной матрицы
t0 = time.time()
matrix = np.random.rand(100, 100)
# CPU версия
#result_cpu = np.dot(matrix, matrix)
#print(f"cpu: {time.time()-t0}")
# GPU версия (с библиотекой CuPy)
matrix_gpu = cp.array(matrix)
print(f"copy gpu: {time.time()-t0}")
t0 = time.time()
result_gpu = matrix_gpu.copy()
for x in tqdm.tqdm(range(1000000)):
    cp.dot(matrix_gpu, matrix_gpu.T, result_gpu)

print(f"gpu: {time.time()-t0}")
#del result_gpu