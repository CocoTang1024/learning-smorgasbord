import re
import os
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# 清洗文本
def clean_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

# 处理单个 HTML 文件，提取文献信息
def process_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    current_time = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    documents = []
    rows = soup.find_all('tr', class_='TableItems-module__m0Z0b')

    for row in rows:
        preprint_span = row.find('span', string=re.compile('Preprint.*开放获取'))
        if preprint_span and not row.find('h3'):
            continue

        title_div = row.find('div', class_='TableItems-module__sHEzP')
        if title_div and title_div.find('h3'):
            doc = {}
            title_span = title_div.find('h3').find('span')
            doc['title'] = clean_text(title_span.get_text(strip=True)) if title_span else ''

            author_div = row.find('div', class_='author-list')
            authors = []
            if author_div:
                author_buttons = author_div.find_all('button')
                for button in author_buttons:
                    author_name = button.find('span', class_='Typography-module__lVnit')
                    if author_name:
                        authors.append(clean_text(author_name.get_text(strip=True)))
            doc['authors'] = authors

            source_div = row.find('div', class_='DocumentResultsList-module__tqiI3')
            doc['publisher'] = clean_text(source_div.find('span').get_text(strip=True)) if source_div else ''

            year_div = row.find('div', class_='TableItems-module__TpdzW')
            doc['year'] = clean_text(year_div.find('span').get_text(strip=True)) if year_div else ''

            abstract = ''
            current_row = row
            while True:
                next_row = current_row.find_next('tr')
                if not next_row:
                    break
                abstract_div = next_row.find('div', class_='Abstract-module__ukTwj')
                if abstract_div:
                    abstract = clean_text(abstract_div.get_text(strip=True))
                    break
                current_row = next_row
            doc['abstract'] = abstract
            doc['current_time'] = current_time
            documents.append(doc)
    return documents

# 将文献信息写入 RIS 格式
def write_ris(documents, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        for doc in documents:
            f.write("TY  - GEN\n")
            for author in doc['authors']:
                f.write(f"AU  - {author}\n")
            f.write(f"TI  - {doc['title']}\n")
            if doc['abstract']:
                f.write(f"AB  - {doc['abstract']}\n")
            if doc['publisher']:
                f.write(f"PB  - {doc['publisher']}\n")
            if doc['year']:
                f.write(f"PY  - {doc['year']}\n")
            st = doc['title'].split(":")[0] if ":" in doc['title'] else doc['title']
            f.write(f"ST  - {st}\n")
            f.write(f"Y2  - {doc['current_time']}\n")
            f.write("ER  -\n\n")

# 主函数：多线程读取多个文件并合并写入 RIS 文件
def main():
    html_dir = R"D:\Users\tang\Downloads"
    ris_output_path = "arxiv_results_multi.ris"
    
    filenames = [
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：53：23).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：53：02).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：52：37).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：52：11).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：51：47).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：51：20).html",
        "Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：49：05).html",
    ]

    filepaths = [os.path.join(html_dir, name) for name in filenames]

    # 多线程处理文件
    all_documents = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        results = list(executor.map(process_html_file, filepaths))
        for docs in results:
            all_documents.extend(docs)

    # 写入 RIS
    write_ris(all_documents, ris_output_path)
    print(f"已成功处理 {len(filepaths)} 个文件，共导出 {len(all_documents)} 条文献。")

if __name__ == "__main__":
    main()
