from Bio import SeqIO

# 读取 .fna 文件
for record in SeqIO.parse("GCF_000004075.3_Cucumber_9930_V3_genomic.fna", "fasta"):
    print(f"序列 ID: {record.id}")
    print(f"序列长度: {len(record.seq)}")
    print(f"前 50 个碱基: {record.seq[:50]}")