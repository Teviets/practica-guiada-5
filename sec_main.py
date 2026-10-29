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


def nucleos_cuda():
   # Núcleos CUDA por multiprocesador según la arquitectura (8.6 = RTX 30xx, 7.5 = T4 de Colab).
   nucleos_por_sm = {(6, 1): 128, (7, 0): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128, (8, 9): 128}
   gpu = cuda.get_current_device()
   return gpu.MULTIPROCESSOR_COUNT * nucleos_por_sm.get(tuple(gpu.compute_capability), 128)


def imprimir_rendimiento(t_cpu, t_gpu, p):
   speedup_sec = t_cpu / t_cpu
   speedup_par = t_cpu / t_gpu
   print(f"\n{'version':<12}{'tiempo (s)':>12}{'p':>8}{'speedup':>12}{'eficiencia':>12}")
   print(f"{'secuencial':<12}{t_cpu:>12.4f}{1:>8}{speedup_sec:>11.2f}x{speedup_sec / 1:>12.2%}")
   print(f"{'paralelo':<12}{t_gpu:>12.4f}{p:>8}{speedup_par:>11.2f}x{speedup_par / p:>12.2%}")


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

# La primera llamada compila el kernel y la primera copia inicializa CUDA.
   multiplicar_paralelo[bloques, hilos](dA, dB, dC)
   cuda.synchronize()

   t0 = time.perf_counter()
   dA = cuda.to_device(A)
   dB = cuda.to_device(B)
   cuda.synchronize()
   t_h2d = time.perf_counter() - t0

   t0 = time.perf_counter()
   multiplicar_paralelo[bloques, hilos](dA, dB, dC)
   cuda.synchronize()
   t_kernel = time.perf_counter() - t0

   t0 = time.perf_counter()
   C_gpu = dC.copy_to_host()
   t_d2h = time.perf_counter() - t0

   t_gpu = t_h2d + t_kernel + t_d2h
   p = nucleos_cuda()


   # NumPy resume la matriz con "..." cuando es muy grande.
   np.set_printoptions(precision=3, linewidth=120)
   print(f"C secuencial {C_cpu.shape}:\n{C_cpu}")
   print(f"C paralelo {C_gpu.shape}:\n{C_gpu}")

   print(f"secuencial: {t_cpu:.3f} s")
   print(f"paralelo:   {t_gpu:.4f} s (host->GPU {t_h2d * 1000:.3f} ms + kernel {t_kernel * 1000:.3f} ms + GPU->host {t_d2h * 1000:.3f} ms)")
   print(f"coincide:   {np.allclose(C_cpu, C_gpu, rtol=1e-3, atol=1e-3)}")
   imprimir_rendimiento(t_cpu, t_gpu, p)


if __name__ == "__main__":
   main()
