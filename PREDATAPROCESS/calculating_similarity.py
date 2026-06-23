"Gaussiankernel similarity"
import numpy as np
import pandas as pd
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
            # print("gau",gauss_kernel_sim)

    return gauss_kernel_sim

df = pd.read_excel('../data/miRNA_drug_matrix.xlsx', index_col=0)
A=df.to_numpy()
A_T = A.T

# 计算高斯核相似性矩阵
mm = calculate_GaussianKernel_sim(A)
dd = calculate_GaussianKernel_sim(A_T)

print("miRNA 相似性矩阵大小:", mm.shape)
print("药物 相似性矩阵大小:", dd.shape)

# --- 修复 Bug：拼接成完整的异构图邻接矩阵 ---
# 把 mm(miRNA-miRNA), A(miRNA-Drug), A.T(Drug-miRNA), dd(Drug-Drug) 拼成一个大矩阵
top = np.hstack((mm, A))
bottom = np.hstack((A_T, dd))
MD_adjaceny = np.vstack((top, bottom))

print("拼接后完整的异构网络大小:", MD_adjaceny.shape)

# 保存完整的矩阵（这才是真正的 MD_adjaceny ！）
np.savetxt("../data/MD_adjaceny.txt", MD_adjaceny, fmt='%.4f')
