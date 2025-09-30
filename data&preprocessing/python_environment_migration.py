#!/usr/bin/env python3
"""
Python环境迁移工具
帮助将包从旧Python环境迁移到新环境，并清理旧版本

使用方法:
1. 先运行导出命令导出包列表
2. 然后在新环境中安装这些包
3. 最后清理旧环境
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(cmd, shell=True):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_python_info(python_path):
    """获取Python版本和路径信息"""
    success, stdout, stderr = run_command(f"{python_path} --version")
    if success:
        version = stdout.strip()
        success2, stdout2, _ = run_command(f"{python_path} -c 'import sys; print(sys.executable)'")
        executable = stdout2.strip() if success2 else python_path
        return version, executable
    return None, python_path

def export_packages(python_path, output_file):
    """导出Python环境中的包列表"""
    print(f"🔍 正在导出 {python_path} 的包列表...")
    
    # 获取Python信息
    version, executable = get_python_info(python_path)
    print(f"Python版本: {version}")
    print(f"Python路径: {executable}")
    
    # 导出pip包列表
    success, stdout, stderr = run_command(f"{python_path} -m pip list --format=freeze")
    
    if success:
        packages = []
        for line in stdout.strip().split('\n'):
            if line and '==' in line:
                packages.append(line)
        
        # 保存到文件
        export_data = {
            'python_version': version,
            'python_executable': executable,
            'export_date': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'packages': packages
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 成功导出 {len(packages)} 个包到 {output_file}")
        return True
    else:
        print(f"❌ 导出失败: {stderr}")
        return False

def install_packages_from_export(target_python, export_file):
    """从导出文件安装包到目标Python环境"""
    if not os.path.exists(export_file):
        print(f"❌ 导出文件不存在: {export_file}")
        return False
    
    with open(export_file, 'r', encoding='utf-8') as f:
        export_data = json.load(f)
    
    packages = export_data['packages']
    print(f"🚀 正在安装 {len(packages)} 个包到 {target_python}...")
    
    # 获取目标Python信息
    version, executable = get_python_info(target_python)
    print(f"目标Python版本: {version}")
    print(f"目标Python路径: {executable}")
    
    # 创建requirements.txt
    requirements_file = "temp_requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(packages))
    
    # 安装包
    success, stdout, stderr = run_command(f"{target_python} -m pip install -r {requirements_file}")
    
    # 清理临时文件
    os.remove(requirements_file)
    
    if success:
        print("✅ 包安装完成!")
        return True
    else:
        print(f"❌ 安装失败: {stderr}")
        return False

def list_python_environments():
    """列出所有可用的Python环境"""
    print("🐍 可用的Python环境:")
    
    # 1. 系统Python
    pythons = ['/usr/bin/python3', '/usr/local/bin/python3', 'python3']
    
    for py in pythons:
        version, executable = get_python_info(py)
        if version:
            print(f"  • {executable} ({version})")
    
    # 2. Conda环境
    success, stdout, stderr = run_command("conda info --envs")
    if success:
        print("\n🔬 Conda环境:")
        for line in stdout.split('\n'):
            if line.strip() and not line.startswith('#') and not line.startswith('base'):
                parts = line.split()
                if len(parts) >= 2:
                    env_name = parts[0]
                    env_path = parts[-1]
                    python_path = f"{env_path}/bin/python"
                    if os.path.exists(python_path):
                        version, _ = get_python_info(python_path)
                        print(f"  • {env_name}: {python_path} ({version})")

def main():
    """主函数"""
    print("🔧 Python环境迁移工具")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python python_environment_migration.py list                    # 列出所有环境")
        print("  python python_environment_migration.py export <python_path>    # 导出包列表")
        print("  python python_environment_migration.py install <python_path> <export_file>  # 安装包")
        print()
        print("示例:")
        print("  python python_environment_migration.py export /usr/bin/python3")
        print("  python python_environment_migration.py install /Users/goffy/miniconda3/bin/python3 python_packages_export.json")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_python_environments()
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("❌ 请指定Python路径")
            return
        
        python_path = sys.argv[2]
        output_file = f"python_packages_export_{python_path.replace('/', '_').replace(' ', '_')}.json"
        export_packages(python_path, output_file)
    
    elif command == "install":
        if len(sys.argv) < 4:
            print("❌ 请指定目标Python路径和导出文件")
            return
        
        target_python = sys.argv[2]
        export_file = sys.argv[3]
        install_packages_from_export(target_python, export_file)
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()
