import re
import uuid
from datetime import datetime

def parse_ris_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    entries = content.strip().split('\nER  -\n')
    return [entry + '\nER  -' for entry in entries if entry.strip()]

def extract_conference_info(n1):
    if not n1:
        return None, None
    match = re.search(r'Conference name: ([^;]+);', n1)
    if not match:
        return None, None
    conf_name = match.group(1).strip()
    year_match = re.search(r'Conference date:.*?(\d{4})', n1)
    year = year_match.group(1) if year_match else None
    return conf_name, year

def get_conference_abbreviation(conf_name):
    conference_map = {
        'European Conference on Computer Vision': 'ECCV',
        'Asian Conference on Computer Vision': 'ACCV',
        'Asian Conference on Pattern Recognition': 'ACPR',
        'International Conference on Artificial Intelligence': 'AIE',
        'International Conference on Computer Analysis of Images and Patterns': 'CAIP',
        'Cross Domain and Multi-modal Knowledge Discovery and Augmentation': 'CD-MAKE',
        'Computer Graphics International': 'CGI',
        'CAAI International Conference on Artificial Intelligence': 'CICAI',
        'Computational Visual Media Conference': 'CVM',
        'German Conference on Pattern Recognition': 'DAGM-GCPR',
        'International Conference on Artificial Neural Networks': 'ICANN',
        'International Conference on Image Analysis and Processing': 'ICIAP',
        'International Conference on Intelligent Computing': 'ICIC',
        'International Conference on Image and Graphics': 'ICIG',
        'International Conference on Intelligent Robotics and Applications': 'ICIRA',
        'International Conference on Neural Information Processing': 'ICONIP',
        'International Conference on Pattern Recognition': 'ICPR',
        'International Symposium on Neural Networks': 'ISNN',
        'International Symposium on Visual Computing': 'ISVC',
        'International Conference on Multimedia Modeling': 'MMM',
        'Pacific-Rim Conference on Multimedia': 'PCM',
        'Chinese Conference on Pattern Recognition and Computer Vision': 'PRCV',
        'Pacific Rim International Conference on Artificial Intelligence': 'PRICAI',
        'Scandinavian Conference on Image Analysis': 'SCIA',
        'International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems': 'IEA/AIE'
    }
    for full_name, abbr in conference_map.items():
        if full_name in conf_name:
            return abbr
    return None

def process_entry(entry):
    lines = entry.strip().split('\n')
    n1_line = next((line for line in lines if line.startswith('N1  -')), None)
    t2_line_index = next((i for i, line in enumerate(lines) if line.startswith('T2  -')), None)
    
    if not n1_line or t2_line_index is None:
        return entry
    
    conf_name, year = extract_conference_info(n1_line)
    if not conf_name or not year:
        return entry
    
    conf_abbr = get_conference_abbreviation(conf_name)
    if not conf_abbr:
        return entry
    
    new_t2 = f'T2  - {conf_abbr} {year}'
    lines[t2_line_index] = new_t2
    return '\n'.join(lines)

def process_ris_file(input_path, output_path):
    entries = parse_ris_file(input_path)
    processed_entries = [process_entry(entry) for entry in entries]
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n\n'.join(processed_entries))

if __name__ == '__main__':
    input_file = 'arxiv\\top\\862_2025_05_07_不限定任何内容_scopus.ris'
    output_file = 'arxiv\\top\\862_2025_05_07_不限定任何内容_scopus.ris_processed.ris'
    process_ris_file(input_file, output_file)
    print(f"Processed RIS file saved to {output_file}")