import re

# 读取 .ris 文件
def read_ris_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 更新 T2 字段
def update_bibtex_entry(bib_entry):
    # 查找 T2 字段
    t2_field = re.search(r'T2\s*-*\s*(.*)', bib_entry)
    
    # 如果找到了 T2 字段
    if t2_field:
        t2_value = t2_field.group(1)
        
        # 如果 T2 中包含 "Lecture Notes in Computer Science"
        if "Lecture Notes in Computer Science" in t2_value:
            # 查找 ECCV 的年份信息（例如：2024）
            year_match = re.search(r'(\d{4})', bib_entry)
            if year_match:
                year = year_match.group(1)
                # 替换 T2 为 ECCV 年份
                updated_t2 = f'ECCV {year}'
                bib_entry = bib_entry.replace(t2_value, updated_t2)
    
    return bib_entry

# 保存更新后的 .ris 文件
def save_updated_ris_file(updated_entries, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(updated_entries)

# 处理 .ris 文件
def process_ris_file(input_path, output_path):
    # 读取文件内容
    ris_data = read_ris_file(input_path)
    
    # 分割成多个条目
    entries = ris_data.split("\n\n")  # 假设条目之间是空行分隔的
    
    # 更新每个条目
    updated_entries = ""
    for entry in entries:
        updated_entries += update_bibtex_entry(entry) + "\n\n"
    
    # 保存更新后的内容
    save_updated_ris_file(updated_entries, output_path)
    print(f"Updated RIS data has been saved to {output_path}")

# 示例：读取、处理并保存文件
input_ris_file = "arxiv\\top\\scopus (3).ris"  # 输入文件路径
output_ris_file = "./updated_scopus.ris"  # 输出文件路径

process_ris_file(input_ris_file, output_ris_file)
