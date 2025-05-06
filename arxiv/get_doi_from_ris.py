import requests
import time
import re

def query_doi(title):
    """
    通过 CrossRef API 根据论文标题查询 DOI，返回 DOI 字符串或 None。
    """
    url = 'https://api.crossref.org/works'
    params = {
        'query.title': title,
        'rows': 1
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get('message', {}).get('items', [])
        if items:
            return items[0].get('DOI')
    except Exception as e:
        print(f"Error querying CrossRef for title `{title}`: {e}")
    return None

def insert_doi_lines(ris_text):
    """
    对一整个 RIS 文本，拆分成多条记录，
    对每条记录提取 TI 字段，查询 DOI，
    并在 ER 之前插入 DO 字段。
    返回所有记录拼接后的新文本。
    """
    records = re.split(r'\nER  -.*', ris_text)
    endings = re.findall(r'(?:\nER  -.*)', ris_text)
    new_records = []

    for rec, ending in zip(records, endings):
        # 跳过空记录
        if not rec.strip():
            continue

        # 提取标题（TI  - 开头的一行）
        m = re.search(r'^TI  - (.+)$', rec, flags=re.MULTILINE)
        title = m.group(1).strip() if m else None

        doi = None
        if title:
            doi = query_doi(title)
            time.sleep(1)  # 避免过快请求限流

        # 如果查到了 DOI，就插入 DO  - 行
        if doi:
            rec = rec + f'\nDO  - {doi}'

        # 把 ER 行放回
        new_records.append(rec + ending)

    return "\n".join(new_records)

def main():
    infile = 'arxiv/arxiv_results_multi.ris'
    outfile = 'arxiv/arxiv_with_doi.ris'

    # 读入原始 RIS
    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read()

    # 插入 DOI
    new_text = insert_doi_lines(text)

    # 写入新文件
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(new_text)

    print(f"已生成带 DOI 的 RIS: {outfile}")

if __name__ == '__main__':
    main()
