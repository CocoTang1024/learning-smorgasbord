import urllib.request

# 构建 API 请求 URL
search_query = 'ti:(temporal OR spatiotemp* OR sequential*) AND ti:(action* OR behavio* OR activit*) AND ti:(detect* OR localiz*)'
start = 0
max_results = 2000
base_url = 'http://export.arxiv.org/api/query?'
query = f'search_query={urllib.parse.quote(search_query)}&start={start}&max_results={max_results}'
url = base_url + query

# 发送请求并直接保存 XML
response = urllib.request.urlopen(url).read()

with open("arxiv_raw.xml", "wb") as f:
    f.write(response)

print("已保存为 arxiv_raw.xml")
