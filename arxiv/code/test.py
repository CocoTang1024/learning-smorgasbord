def parse_ris_file(file_path):
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        entry = {}
        for line in f:
            line = line.strip()
            if line == 'ER  -':
                if 'TI' in entry:
                    entries.append(entry)
                entry = {}
            elif line:
                if line[:2].isalpha() and line[2:6] == '  - ':
                    key = line[:2]
                    value = line[6:].strip()
                    if key in entry:
                        entry[key] += ' ' + value
                    else:
                        entry[key] = value
        # Add last entry if not ended properly
        if entry and 'TI' in entry:
            entries.append(entry)
    return entries

def compare_ris_titles(file1, file2):
    entries1 = parse_ris_file(file1)
    entries2 = parse_ris_file(file2)

    titles1 = {entry['TI']: entry.get('T2', '') for entry in entries1}
    titles2 = {entry['TI']: entry.get('T2', '') for entry in entries2}

    unique_to_file1 = set(titles1.keys()) - set(titles2.keys())
    unique_to_file2 = set(titles2.keys()) - set(titles1.keys())

    print(f"Entries unique to {file1}:")
    for title in unique_to_file1:
        print(f"Title: {title}")
        print(f"Journal/Conference: {titles1[title]}")
        print()

    print(f"Entries unique to {file2}:")
    for title in unique_to_file2:
        print(f"Title: {title}")
        print(f"Journal/Conference: {titles2[title]}")
        print()

# 示例用法
file1 = R'D:\Programs\Codes\Skill-Up\learning-smorgasbord\arxiv\data\4373.ris'
file2 = R'D:\Programs\Codes\Skill-Up\learning-smorgasbord\arxiv\data\20250508_scopus_3837_tad_tal 20250508_wos_886_tad_tal_3891-3_3888_deduplication_end_arxiv_1635_5523_deduplication_4374.ris'
compare_ris_titles(file1, file2)
