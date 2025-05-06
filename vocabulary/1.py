import xml.etree.ElementTree as ET

def extract_words_from_xml(input_file, output_file):
    # 解析XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()

    # 提取<word>标签中的内容，并去重
    words = [item.find('word').text for item in root.findall('item') if item.find('word') is not None]
    unique_words = sorted(set(words))  # 去重并排序

    # 将去重后的单词写入输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in unique_words:
            file.write(word + '\n')

# 输入文件和输出文件的路径
input_file = R"D:\Users\tang\Desktop\六级过.xml"
# input_file = R".\六级过.xml"
output_file = 'words.txt'

# 调用函数进行操作
extract_words_from_xml(input_file, output_file)
