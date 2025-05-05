import urllib.request
import feedparser
import time
from datetime import datetime

# 构建查询参数
search_query = 'ti:(temporal OR spatiotemp* OR sequential*) AND ti:(action* OR behavio* OR activit*) AND ti:(detect* OR localiz*)'
start = 0
max_results = 2000  # 可根据需要调整

# 构建 API 请求 URL
base_url = 'http://export.arxiv.org/api/query?'
query = f'search_query={urllib.parse.quote(search_query)}&start={start}&max_results={max_results}'
url = base_url + query

# 发送请求并解析响应
response = urllib.request.urlopen(url).read()
feed = feedparser.parse(response)

# 获取当前时间，格式为 Y2 需要的格式
current_time = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")

# 保存结果到 RIS 文件
with open("arxiv_results1.ris", "w", encoding="utf-8") as f:
    for entry in feed.entries:
        arxiv_id = entry.get("id", "")
        title = entry.get("title", "").strip().replace("\n", " ")
        summary = entry.get("summary", "").strip().replace("\n", " ")
        published = entry.get("published", "")
        updated = entry.get("updated", "")
        arxiv_doi = entry.get("arxiv_doi", "")

        # 处理 authors
        authors_list = [author.name for author in entry.get("authors", [])]

        # 处理 DA 和 PY
        da = updated.split('T')[0].replace('-', '/') + "/"
        py = updated.split('-')[0]

        # 处理 ST（冒号前的部分，如果有）
        st = title.split(":")[0] if ":" in title else title

        f.write("TY  - GEN\n")
        for author in authors_list:
            f.write(f"AU  - {author}\n")
        f.write(f"TI  - {title}\n")
        f.write(f"T2  - \n")
        f.write(f"AB  - {summary}\n")
        f.write(f"DA  - {da}\n")
        f.write(f"PY  - {py}\n")
        f.write(f"DO  - {arxiv_doi}\n")
        f.write(f"DP  - arXiv.org\n")
        f.write(f"PB  - arXiv\n")
        f.write(f"ST  - {st}\n")
        f.write(f"UR  - {arxiv_id}\n")
        f.write(f"Y2  - {current_time}\n")
        f.write("ER  -\n\n")
