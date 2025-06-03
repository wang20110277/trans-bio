import os
import numpy as np
from Bio import SeqIO

def read_fasta(file_path):
    """
    读取FASTA文件中的序列
    
    Args:
        file_path (str): FASTA文件路径
        
    Returns:
        list: 序列列表
        
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当解析文件出错时
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
        
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
        return sequences
    except Exception as e:
        raise ValueError(f"解析FASTA文件时出错: {e}")

def one_hot_encode(sequence):
    """
    对DNA序列进行One-hot编码
    
    Args:
        sequence (str): DNA序列
        
    Returns:
        numpy.ndarray: 编码后的序列
    """
    mapping = {'A': [1, 0, 0, 0], 
               'T': [0, 1, 0, 0], 
               'C': [0, 0, 1, 0], 
               'G': [0, 0, 0, 1]}
    return np.array([mapping.get(nucleotide.upper(), [0, 0, 0, 0]) for nucleotide in sequence])

def calculate_gc_content(sequences):
    """
    计算序列的GC含量
    
    Args:
        sequences (list): DNA序列列表
        
    Returns:
        list: GC含量列表
    """
    gc_contents = []
    for seq in sequences:
        if len(seq) == 0:
            gc_contents.append(0)
            continue
            
        seq_upper = seq.upper()
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        gc_content = gc_count / len(seq)
        gc_contents.append(gc_content)
    return gc_contents

def load_hbv_data(data_dir):
    """加载 HBV 相关数据"""
    # 这里需要根据实际的 HBV 数据格式进行调整
    # 示例代码，假设数据以 CSV 格式存储
    import pandas as pd
    
    csv_file_path = os.path.join(data_dir, 'hbv_data.csv')
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"数据文件 {csv_file_path} 不存在")
        
    try:
        data = pd.read_csv(csv_file_path)
        if 'target' not in data.columns:
            raise ValueError("数据文件中缺少target列")
            
        X = data.drop('target', axis=1).values
        y = data['target'].values
        return X, y
    except Exception as e:
        raise ValueError(f"加载HBV数据时出错: {e}")