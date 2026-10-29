import math
import time
import numpy as np
from numba import cuda


def multiplicar_secuencial(A, B):
   n = A.shape[0]
   C = np.zeros((n, n), dtype=np.float32)
   for i in range(n):
       for j in range(n):
           suma = np.float32(0.0)
           for k in range(n):
               suma += A[i, k] * B[k, j]
           C[i, j] = suma
   return C


@cuda.jit
def multiplicar_paralelo(A, B, C):
   i, j = cuda.grid(2)
   n = A.shape[0]
   if i < n and j < n:
       suma = np.float32(0.0)
       for k in range(n):
           suma += A[i, k] * B[k, j]
       C[i, j] = suma


def main():
   cuda.detect()
   n = 256
   rng = np.random.default_rng(1)
   A = rng.random((n, n), dtype=np.float32)
   B = rng.random((n, n), dtype=np.float32)
   t0 = time.perf_counter()
   C_cpu = multiplicar_secuencial(A, B)
   t_cpu = time.perf_counter() - t0
   dA = cuda.to_device(A)
   dB = cuda.to_device(B)
   dC = cuda.device_array((n, n), dtype=np.float32)
   hilos = (16, 16)
   bloques = (math.ceil(n / 16), math.ceil(n / 16))

# La primera llamada compila el kernel.
   multiplicar_paralelo[bloques, hilos](dA, dB, dC)
   cuda.synchronize()
   t0 = time.perf_counter()
   multiplicar_paralelo[bloques, hilos](dA, dB, dC)
   cuda.synchronize()
   t_gpu = time.perf_counter() - t0
   C_gpu = dC.copy_to_host()


   print(f"secuencial: {t_cpu:.3f} s")
   print(f"paralelo:   {t_gpu:.4f} s")
   print(f"coincide:   {np.allclose(C_cpu, C_gpu, rtol=1e-3, atol=1e-3)}")
