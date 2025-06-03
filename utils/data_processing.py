from Bio import SeqIO
import numpy as np

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in sequence])


def load_hbv_data(data_dir):
    """加载 HBV 相关数据"""
    # 这里需要根据实际的 HBV 数据格式进行调整
    # 示例代码，假设数据以 CSV 格式存储
    import pandas as pd
    data = pd.read_csv(os.path.join(data_dir, 'hbv_data.csv'))
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return X, y