"""
Failed to update the following entries: Learning Complementary Instance Representation with Parallel Adaptive Graph-Based Network for Action Detection, A Trimodal Dataset: RGB, Thermal, and Depth for Human Segmentation and Temporal Action Detection, Weakly-Supervised Temporal Action Localization with Regional Similarity Consistency, Spatiotemporal Perturbation Based Dynamic Consistency for Semi-supervised Temporal Action Detection, Wonderful clips of playing basketball: A database for localizing wonderful actions, Temporal action localization based on temporal evolution model and multiple instance learning
"""
# 将LNCS的论文分出来
import re
import uuid
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(filename='ris_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_ris_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        entries = content.strip().split('\nER  -\n')
        entries = [entry + '\nER  -' for entry in entries if entry.strip()]
        logging.info(f"Parsed {len(entries)} entries from {file_path}")
        return entries
    except Exception as e:
        logging.error(f"Error parsing file {file_path}: {str(e)}")
        raise

def extract_conference_info(n1):
    if not n1:
        logging.warning("N1 field is empty")
        return None, None
    match = re.search(r'Conference name: ([^;]+);', n1)
    if not match:
        logging.warning(f"No conference name found in N1: {n1}")
        return None, None
    conf_name = match.group(1).strip()
    year_match = re.search(r'Conference date:.*?(\d{4})', n1)
    year = year_match.group(1) if year_match else None
    if not year:
        logging.warning(f"No year found in N1: {n1}")
    return conf_name, year

def get_conference_abbreviation(conf_name):
    conference_map = {
        'European Conference on Computer Vision': 'ECCV',
        'Asian Conference on Computer Vision': 'ACCV',
        'Asian Conference on Pattern Recognition': 'ACPR',
        'Computer Graphics International': 'CGI',
        'German Conference on Pattern Recognition': 'DAGM-GCPR',
        'International Conference on Artificial Neural Networks': 'ICANN',
        'International Conference on Image Analysis and Processing': 'ICIAP',
        'International Conference on Intelligent Computing': 'ICIC',
        'International Conference on Image and Graphics': 'ICIG',
        'International Conference on Intelligent Robotics and Applications': 'ICIRA',
        'International Conference on Neural Information Processing': 'ICONIP',
        'International Conference on Pattern Recognition': 'ICPR',
        'International Symposium on Visual Computing': 'ISVC',
        'International Conference on Multimedia Modeling': 'MMM',
        'Pacific-Rim Conference on Multimedia': 'PCM',
        'Chinese Conference on Pattern Recognition and Computer Vision': 'PRCV',
        'Pacific Rim International Conference on Artificial Intelligence': 'PRICAI',
        'Scandinavian Conference on Image Analysis': 'SCIA'
    }
    for full_name, abbr in conference_map.items():
        if full_name in conf_name:
            return abbr
    return None

def is_lncs_t2(t2_line):
    return t2_line.strip() == 'T2  - Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)'

def get_title(lines):
    title_line = next((line for line in lines if line.startswith('TI  -')), 'TI  - Unknown')
    return title_line.replace('TI  - ', '').strip()

def process_entry(entry):
    lines = entry.strip().split('\n')
    n1_line = next((line for line in lines if line.startswith('N1  -')), None)
    t2_line_index = next((i for i, line in enumerate(lines) if line.startswith('T2  -')), None)
    
    if not n1_line or t2_line_index is None:
        logging.info(f"Entry unchanged (missing N1 or T2): {get_title(lines)}")
        return entry
    
    t2_line = lines[t2_line_index]
    if not is_lncs_t2(t2_line):
        logging.info(f"Entry unchanged (non-LNCS T2): {get_title(lines)}, T2: {t2_line}")
        return entry
    
    conf_name, year = extract_conference_info(n1_line)
    if not conf_name:
        logging.info(f"Entry unchanged (no conference name): {get_title(lines)}")
        return entry
    
    conf_abbr = get_conference_abbreviation(conf_name)
    if not conf_abbr:
        logging.info(f"Entry unchanged (unrecognized conference): {get_title(lines)}, Conference: {conf_name}")
        return entry
    
    if not year:
        logging.warning(f"Failed to update entry (missing year): {get_title(lines)}, Conference: {conf_name}")
        return entry
    
    new_t2 = f'T2  - {conf_abbr} {year}'
    lines[t2_line_index] = new_t2
    logging.info(f"Updated entry: {get_title(lines)}, Conference: {conf_name} to {new_t2}")
    return '\n'.join(lines)

def process_ris_file(input_path, output_path):
    try:
        entries = parse_ris_file(input_path)
        processed_entries = []
        failed_entries = []
        
        for entry in entries:
            processed_entry = process_entry(entry)
            lines = processed_entry.strip().split('\n')
            title = get_title(lines)
            if processed_entry == entry and is_lncs_t2(next((line for line in lines if line.startswith('T2  -')), '')):
                conf_name, year = extract_conference_info(next((line for line in lines if line.startswith('N1  -')), ''))
                if conf_name and not get_conference_abbreviation(conf_name) or not year:
                    failed_entries.append(title)
            processed_entries.append(processed_entry)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n\n'.join(processed_entries))
        
        logging.info(f"Processed {len(entries)} entries, {len(processed_entries)} written to output")
        if failed_entries:
            logging.warning(f"Failed to update {len(failed_entries)} entries: {', '.join(failed_entries)}")
            print(f"Failed to update the following entries: {', '.join(failed_entries)}")
        else:
            print("All entries processed successfully.")
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise

if __name__ == '__main__':
    input_file = 'arxiv\\top\\813_20250507_tad_tal.ris'
    output_file = 'arxiv\\top\\813_20250507_tad_tal_processed.ris'
    process_ris_file(input_file, output_file)
    print(f"Processed RIS file saved to {output_file}. Check ris_processing.log for details.")