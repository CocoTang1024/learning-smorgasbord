import re
from datetime import datetime
from bs4 import BeautifulSoup

# Function to clean text by removing extra whitespace and newlines
def clean_text(text):
    if not text:
        return ""
    # Replace multiple spaces, newlines, and tabs with a single space, then strip
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

# Read HTML content from file
# with open("D:\\Users\\tang\\Downloads\\12323.html", "r", encoding="utf-8") as f:
with open(R"D:\Users\tang\Downloads\Scopus - 文献搜索结果 ｜ 已登录 (2025_5_5 21：49：05).html", "r", encoding="utf-8") as f:
# with open(R"1.html", "r", encoding="utf-8") as f:
# with open(R"D:\Users\tang\Downloads\1.html", "r", encoding="utf-8") as f:
    html = f.read()

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Get current time for Y2 field
current_time = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")

# Initialize list to store document entries
documents = []

# Find all table rows
rows = soup.find_all('tr', class_='TableItems-module__m0Z0b')

# Process rows to extract document information
for row in rows:
    # Skip rows that are "Preprint • 开放获取" without a title
    preprint_span = row.find('span', string=re.compile('Preprint.*开放获取'))
    if preprint_span and not row.find('h3'):
        continue
    
    # Check if row contains a title (indicating a document entry)
    title_div = row.find('div', class_='TableItems-module__sHEzP')
    if title_div and title_div.find('h3'):
        doc = {}
        
        # Extract and clean title
        title_span = title_div.find('h3').find('span')
        doc['title'] = clean_text(title_span.get_text(strip=True)) if title_span else ''
        
        # Extract and clean authors
        author_div = row.find('div', class_='author-list')
        authors = []
        if author_div:
            author_buttons = author_div.find_all('button')
            for button in author_buttons:
                author_name = button.find('span', class_='Typography-module__lVnit')
                if author_name:
                    authors.append(clean_text(author_name.get_text(strip=True)))
        doc['authors'] = authors
        
        # Extract and clean publisher (PB)
        source_div = row.find('div', class_='DocumentResultsList-module__tqiI3')
        doc['publisher'] = clean_text(source_div.find('span').get_text(strip=True)) if source_div else ''
        
        # Extract and clean year (PY)
        year_div = row.find('div', class_='TableItems-module__TpdzW')
        doc['year'] = clean_text(year_div.find('span').get_text(strip=True)) if year_div else ''
        
        # Find the corresponding abstract in the next row(s)
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
        
        documents.append(doc)

# Write to RIS file
with open("arxiv_results_1.ris", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write("TY  - GEN\n")
        
        # Write authors
        for author in doc['authors']:
            f.write(f"AU  - {author}\n")
        
        # Write title
        title = doc['title']
        f.write(f"TI  - {title}\n")
        
        # Write abstract
        if doc['abstract']:
            f.write(f"AB  - {doc['abstract']}\n")
        
        # Write publisher
        if doc['publisher']:
            f.write(f"PB  - {doc['publisher']}\n")
        
        # Write year
        if doc['year']:
            f.write(f"PY  - {doc['year']}\n")
        
        # Write short title (ST)
        st = title.split(":")[0] if ":" in title else title
        f.write(f"ST  - {st}\n")
        
        # Write access date (Y2)
        f.write(f"Y2  - {current_time}\n")
        
        f.write("ER  -\n\n")