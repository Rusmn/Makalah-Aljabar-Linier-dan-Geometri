import numpy as np
import matplotlib.pyplot as plt
import time

def generate_data(n_points=1000):
    t = np.linspace(0, 10, n_points)
    
    base_prey = np.linspace(10, 27, n_points)
    noise_prey = np.random.normal(0, 0.5, n_points) 
    x = base_prey + noise_prey
    
    base_predator = 5 - 0.15 * t
    noise_predator = np.random.normal(0, 0.3, n_points) 
    y = base_predator + noise_predator
    
    return t, x, y

# Visualisasi data
def plot_data(t, x, y, title="Data Populasi Predator-Prey"):
   plt.figure(figsize=(12,6))
   plt.plot(t, x, 'b-', label='Prey', alpha=0.7)
   plt.plot(t, y, 'r-', label='Predator', alpha=0.7)
   plt.xlabel('Waktu')
   plt.ylabel('Populasi')
   plt.title(title)
   plt.legend()
   plt.grid(True)
   plt.show()

# Implementasi Least Squares
def build_least_squares(t, x, y):
   dx = np.gradient(x, t)
   dy = np.gradient(y, t)
   
   n = len(t)
   A = np.zeros((2*n, 4))
   b = np.zeros(2*n)
   
   for i in range(n):
       # Persamaan untuk dx/dt
       A[2*i] = [x[i], -x[i]*y[i], 0, 0]
       b[2*i] = dx[i]
       
       # Persamaan untuk dy/dt
       A[2*i+1] = [0, 0, -y[i], x[i]*y[i]]
       b[2*i+1] = dy[i]
   
   # Bentuk persamaan normal
   ATA = np.dot(A.T, A)
   ATb = np.dot(A.T, b)
   
   return ATA, ATb

# Eliminasi Gauss dengan Partial Pivoting
def gauss_elimination(A, b):
   n = A.shape[0]
   # Buat augmented matrix [A|b]
   Ab = np.column_stack([A, b])
   x = np.zeros(n)
   
   # Forward elimination
   for i in range(n):
       max_idx = i + np.argmax(np.abs(Ab[i:, i]))
       if max_idx != i:
           Ab[i], Ab[max_idx] = Ab[max_idx].copy(), Ab[i].copy()
           
       pivot = Ab[i,i]
       if np.abs(pivot) < 1e-10: 
           continue
           
       for j in range(i+1, n):
           factor = Ab[j,i] / pivot
           Ab[j] -= factor * Ab[i]
   
   # Back substitution
   for i in range(n-1, -1, -1):
       if np.abs(Ab[i,i]) < 1e-10: 
           x[i] = 0
           continue
       x[i] = Ab[i,-1]
       for j in range(i+1, n):
           x[i] -= Ab[i,j] * x[j]
       x[i] /= Ab[i,i]
   
   return x

# Dekomposisi LU dengan Pivoting
def lu_decomposition(A):
   n = A.shape[0]
   L = np.eye(n)
   U = A.copy()
   P = np.eye(n)
   
   for i in range(n-1):
       pivot_idx = i + np.argmax(np.abs(U[i:, i]))
       if pivot_idx != i:
           U[[i,pivot_idx]] = U[[pivot_idx,i]]
           P[[i,pivot_idx]] = P[[pivot_idx,i]]
           if i > 0:
               L[[i,pivot_idx], :i] = L[[pivot_idx,i], :i]
       
       for j in range(i+1, n):
           if np.abs(U[i,i]) < 1e-10: 
               continue
           factor = U[j,i] / U[i,i]
           L[j,i] = factor
           U[j,i:] -= factor * U[i,i:]
           
   return P, L, U

def solve_lu(P, L, U, b):
   # Solve Ly = Pb
   Pb = np.dot(P, b)
   n = L.shape[0]
   y = np.zeros(n)
   
   for i in range(n):
       y[i] = Pb[i]
       for j in range(i):
           y[i] -= L[i,j] * y[j]
   
   # Solve Ux = y
   x = np.zeros(n)
   for i in range(n-1, -1, -1):
       if np.abs(U[i,i]) < 1e-10:
           x[i] = 0
           continue
       x[i] = y[i]
       for j in range(i+1, n):
           x[i] -= U[i,j] * x[j]
       x[i] /= U[i,i]
       
   return x

def calculate_error(A, x, b):
   """Hitung relative error"""
   residual = np.dot(A, x) - b
   return np.linalg.norm(residual) / np.linalg.norm(b)

def main():
    # Generate data dengan berbagai ukuran untuk analisis
    sizes = [100, 500, 1000, 2000, 5000]
    
    print("\nAnalisis Performa dengan Berbagai Ukuran Data:")
    print("\nUkuran Data | Waktu Gauss (s) | Waktu LU Decomp (s) | Waktu LU Solve (s) | Error Gauss  | Error LU")
    print("-" * 95)
    
    for n in sizes:
        t, x, y = generate_data(n)
        A, b = build_least_squares(t, x, y)
        
        # Eliminasi Gauss
        start_time = time.perf_counter()
        x_gauss = gauss_elimination(A.copy(), b.copy())
        gauss_time = time.perf_counter() - start_time
        error_gauss = calculate_error(A, x_gauss, b)
        
        # Dekomposisi LU - pisahkan waktu dekomposisi dan penyelesaian
        start_decomp = time.perf_counter()
        P, L, U = lu_decomposition(A.copy())
        decomp_time = time.perf_counter() - start_decomp
        
        start_solve = time.perf_counter()
        x_lu = solve_lu(P, L, U, b.copy())
        solve_time = time.perf_counter() - start_solve
        
        error_lu = calculate_error(A, x_lu, b)
        
        print(f"{n:^10d} | {gauss_time:^14.6f} | {decomp_time:^17.6f} | {solve_time:^16.6f} | {error_gauss:^11.2e} | {error_lu:^8.2e}")

        # Demonstrasi keunggulan LU untuk multiple RHS
        if n == sizes[-1]:  # Hanya untuk ukuran terbesar
            print(f"\nDemonstrasi Multiple RHS untuk n = {n}:")
            num_rhs = 5  # Jumlah RHS yang berbeda
            
            # Waktu untuk Gauss dengan multiple RHS
            start_time = time.perf_counter()
            for _ in range(num_rhs):
                b_new = np.random.randn(len(b))
                x_gauss = gauss_elimination(A.copy(), b_new)
            gauss_multi_time = time.perf_counter() - start_time
            
            # Waktu untuk LU dengan multiple RHS (reuse dekomposisi)
            start_time = time.perf_counter()
            P, L, U = lu_decomposition(A.copy()) 
            for _ in range(num_rhs):
                b_new = np.random.randn(len(b))
                x_lu = solve_lu(P, L, U, b_new)
            lu_multi_time = time.perf_counter() - start_time
            
            print(f"\nWaktu untuk {num_rhs} RHS berbeda:")
            print(f"Gauss: {gauss_multi_time:.6f} s")
            print(f"LU   : {lu_multi_time:.6f} s")

    # Plot data
    plot_data(t, x, y, f"Data Asli (n={sizes[-1]})")

if __name__ == "__main__":
    main()