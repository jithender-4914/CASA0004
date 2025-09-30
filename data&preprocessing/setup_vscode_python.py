#!/usr/bin/env python3
"""
VS Code Python解释器切换工具
自动配置VS Code以使用推荐的conda环境
"""

import json
import os
from pathlib import Path

def setup_vscode_python():
    """配置VS Code使用正确的Python解释器"""
    
    # VS Code配置路径
    vscode_dir = Path.cwd() / ".vscode"
    settings_file = vscode_dir / "settings.json"
    
    # 推荐的Python路径
    python_path = "/Users/goffy/miniconda3/bin/python"
    
    # 创建.vscode目录
    vscode_dir.mkdir(exist_ok=True)
    
    # 准备设置
    settings = {
        "python.defaultInterpreterPath": python_path,
        "python.terminal.activateEnvironment": True,
        "jupyter.kernels.trusted": [python_path],
        "jupyter.defaultKernel": python_path,
        "python.analysis.autoImportCompletions": True,
        "python.analysis.typeCheckingMode": "basic"
    }
    
    # 如果设置文件已存在，读取并更新
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
            existing_settings.update(settings)
            settings = existing_settings
        except json.JSONDecodeError:
            print("⚠️ 现有设置文件格式错误，将创建新文件")
    
    # 写入设置
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    
    print(f"✅ VS Code配置已更新:")
    print(f"  📁 配置文件: {settings_file}")
    print(f"  🐍 Python路径: {python_path}")
    print(f"  🔧 设置项: {len(settings)} 个")
    
    return True

def create_launch_config():
    """创建VS Code调试配置"""
    
    vscode_dir = Path.cwd() / ".vscode"
    launch_file = vscode_dir / "launch.json"
    
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "python": "/Users/goffy/miniconda3/bin/python"
            },
            {
                "name": "Python: Jupyter Notebook",
                "type": "python",
                "request": "launch",
                "module": "jupyter",
                "args": ["notebook"],
                "console": "integratedTerminal",
                "python": "/Users/goffy/miniconda3/bin/python"
            }
        ]
    }
    
    with open(launch_file, 'w', encoding='utf-8') as f:
        json.dump(launch_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ VS Code调试配置已创建: {launch_file}")

def main():
    """主函数"""
    print("🔧 VS Code Python环境配置工具")
    print("=" * 50)
    
    # 检查是否在项目目录中
    current_dir = Path.cwd()
    print(f"📂 当前目录: {current_dir}")
    
    # 配置VS Code
    setup_vscode_python()
    create_launch_config()
    
    print("\n🎯 配置完成！接下来请:")
    print("  1. 重启VS Code")
    print("  2. 按 Cmd+Shift+P，搜索 'Python: Select Interpreter'")
    print("  3. 选择 '/Users/goffy/miniconda3/bin/python'")
    print("  4. 测试你的notebook是否正常运行")
    
    print(f"\n💡 提示:")
    print(f"  如果notebook仍使用旧kernel，请在notebook中:")
    print(f"  点击右上角的kernel选择器，选择正确的Python环境")

if __name__ == "__main__":
    main()
