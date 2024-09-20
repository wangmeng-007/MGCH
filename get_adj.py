import numpy as np
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    dataset = 'mirflickr'
    DATA_DIR = '/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/' + dataset + '/'
    adj_file = '/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/' + dataset + '/adj.mat'

    label_train = loadmat(DATA_DIR + "lab.mat")['lab']   # 加载标签数据

    num_class = label_train.shape[1]   # 获取类别数
    adj = np.zeros((num_class, num_class), dtype=int)   # 创建全零邻接矩阵
    num = np.zeros((num_class), dtype=int)   # 创建全零数组用于存储每个类别的数量

    for row in label_train:   # 对于每一行（每个样本）的标签数据
        for i in range(num_class):    # 对于每个类别
            if row[i] == 0:
                continue   # 如果标签值为0，则跳过当前类别
            else:
                num[i] += 1   # 类别数量加1

            for j in range(i, num_class):   # 对于当前类别及其之后的每个类别
                if row[j] == 1:
                    adj[i][j] += 1    # 邻接矩阵对应位置数值加1
                    adj[j][i] += 1    # 邻接矩阵对应对称位置数值加1

    file = {'adj': adj, 'nums': num}   # 创建字典，包含邻接矩阵和类别数量数组
    savemat(adj_file, file)   # 将字典数据保存到邻接矩阵文件
