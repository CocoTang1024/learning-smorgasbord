import requests
import time

def get_doi_from_title(title):
    url = 'https://api.crossref.org/works'
    params = {
        'query.title': title,
        'rows': 1  # 只取一个最匹配的结果
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json()['message']['items']
        if items:
            return items[0].get('DOI')
    return None

# 示例：ArXiv论文标题列表
titles = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Learning Spatiotemporal Features with 3D Convolutional Networks"
]

for title in titles:
    doi = get_doi_from_title(title)
    print(f"Title: {title}\nDOI: {doi}\n")
    time.sleep(1)  # 避免请求过快被限流
