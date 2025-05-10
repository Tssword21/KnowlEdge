import json

def check_notebook():
    """检查KnowlEdge.ipynb文件的内容"""
    notebook_path = "KnowlEdge.ipynb"
    
    # 读取notebook文件
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Notebook has {len(notebook['cells'])} cells")
    
    # 检查每个单元格
    for i, cell in enumerate(notebook['cells']):
        print(f"\nCell {i+1} type: {cell['cell_type']}")
        
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            print(f"Source length: {len(source)} characters")
            
            # 检查是否包含数据库相关代码
            if 'get_db_connection' in source:
                print("Contains get_db_connection function")
            
            if 'initialize_database' in source:
                print("Contains initialize_database function")
            
            if 'sqlite3' in source:
                print("Contains sqlite3 references")
            
            # 打印前100个字符
            print(f"First 100 chars: {source[:100]}")
            
            # 如果单元格很长，也打印最后100个字符
            if len(source) > 200:
                print(f"Last 100 chars: {source[-100:]}")

if __name__ == "__main__":
    check_notebook() 