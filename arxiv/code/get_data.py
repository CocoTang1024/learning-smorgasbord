import urllib.request
import feedparser

# 构建查询参数
search_query = 'ti:(temporal OR spatiotemp* OR sequential*) AND ti:(action* OR behavio* OR activit*)'
start = 0
max_results = 2000  # 可根据需要调整，最大值为 30000

# 构建 API 请求 URL
base_url = 'http://export.arxiv.org/api/query?'
query = f'search_query={urllib.parse.quote(search_query)}&start={start}&max_results={max_results}'
url = base_url + query

# 发送请求并解析响应
response = urllib.request.urlopen(url).read()
feed = feedparser.parse(response)

# 保存结果到 TXT 文件
with open("arxiv_results___.txt", "w", encoding="utf-8") as f:
    for entry in feed.entries:
        # 提取标准字段
        arxiv_id = entry.get("id", "")
        title = entry.get("title", "").strip().replace("\n", " ")
        summary = entry.get("summary", "").strip().replace("\n", " ")
        published = entry.get("published", "")
        updated = entry.get("updated", "")
        summary = entry.get("summary", "")
        authors = ", ".join(author.name for author in entry.get("authors", []))
        
        # 提取所有链接
        links = entry.get("links", [])
        link_info = []
        for link in links:
            href = link.get("href", "")
            rel = link.get("rel", "")
            link_type = link.get("type", "")
            title_attr = link.get("title", "")
            link_info.append(f"href: {href}, rel: {rel}, type: {link_type}, title: {title_attr}")
        links_str = "\n".join(link_info)
        
        # 提取 arXiv 扩展字段
        arxiv_comment = entry.get("arxiv_comment", "")
        arxiv_primary_category = entry.get("arxiv_primary_category", {}).get("term", "")
        arxiv_doi = entry.get("arxiv_doi", "")
        arxiv_journal_ref = entry.get("arxiv_journal_ref", "")
        
        # 提取所有分类
        categories = ", ".join(tag.get("term", "") for tag in entry.get("tags", []))
        
        # 写入到文件
        f.write(f"ID: {arxiv_id}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Published: {published}\n")
        f.write(f"Updated: {updated}\n")
        f.write(f"Authors: {authors}\n")
        f.write(f"Links:\n{links_str}\n")
        f.write(f"Summary: {summary}\n")
        f.write(f"Comment: {arxiv_comment}\n")
        f.write(f"Primary Category: {arxiv_primary_category}\n")
        f.write(f"DOI: {arxiv_doi}\n")
        f.write(f"Journal Reference: {arxiv_journal_ref}\n")
        f.write(f"Categories: {categories}\n")
        f.write("="*80 + "\n")