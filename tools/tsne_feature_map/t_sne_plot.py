import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# 配置参数
input_paths = [
    "data/grass/tsne/-1.csv",
    "data/grass/tsne/0.csv",
    "data/grass/tsne/1.csv",
    "data/grass/tsne/2.csv",
    "data/grass/tsne/2.csv"  # 注意：这里路径重复，请确认是否应为 3.csv？
] 

output_pdf = "data/grass/tsne/all_plots.pdf"  # 最终输出路径
figsize = (8, 6)  # 统一图片尺寸
dpi = 300         # 打印级分辨率
cmap = "viridis"  # 统一配色方案

def validate_csv(path):
    """检查CSV文件是否有效"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_csv(path)
    required_columns = {'x', 'y', 'label'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"文件 {os.path.basename(path)} 缺少必要列: {missing}")
    return df

def plot_single_page(df, pdf, page_num):
    """绘制单个PDF页面"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # 绘制散点图
    plt.scatter(
        x=df['x'], 
        y=df['y'], 
        c=df['label'], 
        cmap=cmap,
        alpha=0.7,
        s=10,          # 适当增大点尺寸
        edgecolor='w',  # 白色边缘增强对比
        linewidth=0.3
    )
    
    # 添加页码和文件名
    filename = os.path.basename(input_paths[page_num])
    plt.title(f"t-SNE Visualization - {filename}\nPage {page_num+1}", fontsize=10, pad=12)
    plt.axis('off')
    
    # 保存到PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# 主处理流程
with PdfPages(output_pdf) as pdf:
    for i, path in enumerate(input_paths):
        try:
            df = validate_csv(path)
            plot_single_page(df, pdf, i)
            print(f"成功处理第 {i+1} 页: {path}")
        except Exception as e:
            print(f"处理文件 {path} 时出错: {str(e)}")
            continue  # 跳过错误文件

print(f"PDF已生成至: {os.path.abspath(output_pdf)}")