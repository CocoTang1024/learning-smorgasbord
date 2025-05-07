# 排查出来多出来的两个
"""
Our code is available in github.com/MengyuanChen21/ECCV2022-DELU
 Compared with ActionFormer (ECCV 2022)
"""
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

def is_eccv_conference(conf_name):
    is_eccv = 'European Conference on Computer Vision' in conf_name if conf_name else False
    if conf_name and not is_eccv:
        logging.info(f"Non-ECCV conference: {conf_name}")
    return is_eccv

def is_lncs_t2(t2_line):
    is_lncs = t2_line.strip() == 'T2  - Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)'
    return is_lncs

def process_entry(entry):
    lines = entry.strip().split('\n')
    n1_line = next((line for line in lines if line.startswith('N1  -')), None)
    t2_line_index = next((i for i, line in enumerate(lines) if line.startswith('T2  -')), None)
    
    if not n1_line or t2_line_index is None:
        logging.info(f"Entry unchanged (missing N1 or T2): {entry[:100]}...")
        return entry
    
    conf_name, year = extract_conference_info(n1_line)
    if not conf_name:
        logging.info(f"Entry unchanged (no conference name): {entry[:100]}...")
        return entry
    
    t2_line = lines[t2_line_index]
    
    # If not ECCV and T2 is LNCS, mark for deletion
    if not is_eccv_conference(conf_name) and is_lncs_t2(t2_line):
        logging.info(f"Deleting non-ECCV LNCS entry: {conf_name}, T2: {t2_line}")
        return None
    
    # Process ECCV entry
    if is_eccv_conference(conf_name) and is_lncs_t2(t2_line):
        if year:
            new_t2 = f'T2  - ECCV {year}'
            lines[t2_line_index] = new_t2
            logging.info(f"Updated ECCV entry: {conf_name} to {new_t2}")
            return '\n'.join(lines)
        else:
            logging.warning(f"ECCV entry not updated due to missing year: {conf_name}")
    
    logging.info(f"Entry unchanged: {conf_name}, T2: {t2_line}")
    return entry

def process_ris_file(input_path, output_path):
    try:
        entries = parse_ris_file(input_path)
        processed_entries = [process_entry(entry) for entry in entries]
        valid_entries = [entry for entry in processed_entries if entry is not None]
        eccv_count = sum(1 for entry in valid_entries if 'T2  - ECCV' in entry)
        logging.info(f"Processed {len(entries)} entries, {len(valid_entries)} remain, {eccv_count} ECCV entries")
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n\n'.join(valid_entries))
        logging.info(f"Output written to {output_path}")
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise

if __name__ == '__main__':
    input_file = 'arxiv\\top\\scopus (3).ris'
    output_file = 'arxiv\\top\\scopus_processed.ris'
    process_ris_file(input_file, output_file)
    print(f"Processed RIS file saved to {output_file}. Check ris_processing.log for details.")