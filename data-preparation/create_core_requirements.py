#!/usr/bin/env python3
"""
创建数据科学核心包列表
从完整的包导出中筛选出数据科学和地理空间分析的核心包
"""

import json

def create_core_data_science_requirements():
    """创建核心数据科学包的requirements.txt"""
    
    # 读取完整的包导出
    with open('python_packages_export__Library_Developer_CommandLineTools_usr_bin_python3.json', 'r') as f:
        export_data = json.load(f)
    
    all_packages = export_data['packages']
    
    # 定义核心数据科学和地理空间分析包
    core_packages = {
        # 数据处理
        'pandas', 'numpy', 'scipy',
        # 可视化
        'matplotlib', 'seaborn', 'plotly', 'plotnine',
        # 机器学习
        'scikit-learn', 'sklearn', 'xgboost',
        # 深度学习
        'torch', 'torch-geometric', 'torchaudio', 'torchvision',
        'transformers', 'tokenizers', 'huggingface-hub', 'safetensors',
        # 地理空间分析
        'geopandas', 'shapely', 'fiona', 'pyproj', 'rasterio',
        'folium', 'contextily', 'geopy', 'googlemaps',
        # 空间分析专用
        'libpysal', 'pyogrio',
        # 数据格式
        'pyarrow', 'fastparquet',
        # Jupyter和notebook
        'jupyter', 'jupyterlab', 'notebook', 'ipykernel', 'ipython',
        # 其他重要工具
        'requests', 'tqdm', 'nltk', 'networkx'
    }
    
    # 筛选核心包
    selected_packages = []
    for package_line in all_packages:
        package_name = package_line.split('==')[0].lower()
        if any(core in package_name for core in core_packages):
            selected_packages.append(package_line)
    
    # 保存到requirements.txt
    with open('data_science_core_requirements.txt', 'w') as f:
        f.write('# 数据科学核心包 - 从Python 3.9.6环境导出\n')
        f.write('# 生成时间: ' + export_data['export_date'] + '\n')
        f.write('# 原始包总数: ' + str(len(all_packages)) + '\n')
        f.write('# 筛选包数量: ' + str(len(selected_packages)) + '\n\n')
        
        for package in sorted(selected_packages):
            f.write(package + '\n')
    
    # 同时创建一个更宽松的版本（不固定版本号）
    with open('data_science_core_requirements_latest.txt', 'w') as f:
        f.write('# 数据科学核心包 - 最新版本\n')
        f.write('# 注意：这个文件使用最新版本，可能会有兼容性问题\n\n')
        
        for package in sorted(selected_packages):
            package_name = package.split('==')[0]
            f.write(package_name + '\n')
    
    print(f"✅ 核心包筛选完成:")
    print(f"  • 原始包总数: {len(all_packages)}")
    print(f"  • 筛选包数量: {len(selected_packages)}")
    print(f"  • 固定版本文件: data_science_core_requirements.txt")
    print(f"  • 最新版本文件: data_science_core_requirements_latest.txt")
    
    # 显示筛选出的包
    print(f"\n📦 筛选出的核心包:")
    for package in sorted(selected_packages):
        print(f"  {package}")

if __name__ == "__main__":
    create_core_data_science_requirements()
