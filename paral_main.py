import math
import time
import numpy as np
from numba import cuda


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
   n = 1000
   rng = np.random.default_rng(1)
   A = rng.random((n, n), dtype=np.float32)
   B = rng.random((n, n), dtype=np.float32)

   print(f"A: {A.shape}, {A.dtype}, {A.nbytes} bytes")
   print(f"B: {B.shape}, {B.dtype}, {B.nbytes} bytes")
   print(A[:3, :3])

   # La primera transferencia también inicializa el contexto CUDA, no se mide.
   cuda.to_device(np.zeros(1, dtype=np.float32))
   cuda.synchronize()

   t0 = time.perf_counter()
   dA = cuda.to_device(A)
   dB = cuda.to_device(B)
   cuda.synchronize()
   t_h2d = time.perf_counter() - t0

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

   t0 = time.perf_counter()
   C_gpu = dC.copy_to_host()
   t_d2h = time.perf_counter() - t0

   mb = (A.nbytes + B.nbytes) / (1024 * 1024)
   print(f"bloques: {bloques}, hilos por bloque: {hilos}")
   print(f"host -> GPU:  {t_h2d * 1000:.3f} ms ({mb:.2f} MB, {mb / t_h2d / 1024:.2f} GB/s)")
   print(f"kernel:       {t_gpu * 1000:.3f} ms")
   print(f"GPU -> host:  {t_d2h * 1000:.3f} ms")
   print(f"coincide:     {np.allclose(C_gpu, A @ B, rtol=1e-3, atol=1e-2)}")


if __name__ == "__main__":
   main()
