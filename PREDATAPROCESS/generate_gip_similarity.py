import numpy as np
import pandas as pd


def calculate_gip_similarity(interaction_matrix):
    n_mirna = interaction_matrix.shape[0]
    norm_sq = np.sum(np.square(interaction_matrix), axis=1)
    gamma = 1.0 / (np.sum(norm_sq) / n_mirna)
    gip_matrix = np.zeros((n_mirna, n_mirna))
    for i in range(n_mirna):
        for j in range(n_mirna):
            diff_sq = np.sum(np.square(interaction_matrix[i, :] - interaction_matrix[j, :]))
            gip_matrix[i, j] = np.exp(-gamma * diff_sq)
    return gip_matrix


if __name__ == "__main__":
    print("正在读取数据...")
    # 这里换成了你服务器的绝对路径！绝对不可能找不到！
    df = pd.read_excel('/data/coding/DLMVF_model-master/data/miRNA_drug_matrix.xlsx', index_col=0)
    association_matrix = df.values

    print("正在计算 GIP 相似性矩阵，请耐心等待 1-2 分钟...")
    mirna_sim_matrix = calculate_gip_similarity(association_matrix)

    print("计算完成！正在保存文件...")
    # 保存路径也是绝对路径！
    np.savetxt("/data/coding/DLMVF_model-master/data/miRNA_GIP_similarity.csv", mirna_sim_matrix, delimiter=",")
    print("🎉 恭喜！1578 x 1578 的 miRNA 专属相似性矩阵生成完毕！")
