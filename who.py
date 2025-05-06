# coding=utf-8
'''
FilePath     : /Tools/who.py
Author       : CocoTang1024 1972555958@qq.com
Date         : 2024-09-30 16:06:25
Version      : 0.0.1
LastEditTime : 2024-09-30 16:06:30
Email        : robotym@163.com
Description  : 
'''
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk  # 如果是非PNG格式图片，比如JPEG，则需要Pillow库
import re

# 定义7个模式的name_list
modes = {
    "汤一鸣": ["邓贤明", "刘博涵", "连奕恒", "汤一鸣", "蔡雨泽", "董画心", "李瑞雨", "赵海洋", "史寒晓", "王文杰", 
            "金玉洲", "余龙", "贾子木", "贾周", "王凯泽", "李安康", "高荣昊", "李飞", "张鸿铭", "谢玉轩", "张炳杭", 
            "刘永昌", "杨林", "黄梓迪", "李子豪", "张航", "杜权伟"],
    "田奕荷": ["胡盈盈", "职芮铭", "田奕荷", "王怡尧", "田婧", "王彤"],
    "贾子木": ["余龙", "贾子木", "贾周", "王凯泽", "李安康", "高荣昊", "李飞", "张鸿铭", "谢玉轩", "张炳杭", 
            "刘永昌", "杨林", "黄梓迪", "李子豪", "张航", "杜权伟", "张晓恩", "李放", "杨凯文", "王尧飞", 
            "孙世杰", "贾舜宇", "莫惠文", "陈泓臻"],
    "韩亚军": ["郭小慧", "秦可凡", "韩亚军", "赵铭"],
    "王敏": ["王敏", "陈一", "耿璐瑶", "胡丹"],
    "陈秀哲": ["张晓恩", "李放", "杨凯文", "王尧飞", "孙世杰", "贾舜宇", "莫惠文", "陈泓臻", "刘达", "陈秀哲", 
            "张晗博", "徐政豪", "夏宇恒", "王启双", "刘卓凡", "李锦哲", "黄志青", "俞杰", "景闰元", "安琪"],
    "全体同学": ["邓贤明", "刘博涵", "连奕恒", "汤一鸣", "蔡雨泽", "董画心", "李瑞雨", "赵海洋", "史寒晓", "王文杰", 
            "金玉洲", "余龙", "贾子木", "贾周", "王凯泽", "李安康", "高荣昊", "李飞", "张鸿铭", "谢玉轩", "张炳杭", 
            "刘永昌", "杨林", "黄梓迪", "李子豪", "张航", "杜权伟", "胡盈盈", "职芮铭", "田奕荷", "王怡尧", 
            "田婧", "王彤", "郭小慧", "秦可凡", "韩亚军", "赵铭", "王敏", "陈一", "耿璐瑶", "胡丹", "张晓恩", 
            "李放", "杨凯文", "王尧飞", "孙世杰", "贾舜宇", "莫惠文", "陈泓臻", "刘达", "陈秀哲", "张晗博", 
            "徐政豪", "夏宇恒", "王启双", "刘卓凡", "李锦哲", "黄志青", "俞杰", "景闰元", "安琪"]
}

# 提取人名的正则表达式，假设人名是2到4个汉字
def extract_names(text):
    # 使用正则表达式提取2到4个连续汉字的人名
    return re.findall(r'[\u4e00-\u9fa5]{2,4}', text)

# 提取内容并比对人名
def check_names():
    input_text = text_box.get("1.0", tk.END)  # 获取文本框中的内容
    extracted_names = extract_names(input_text)  # 提取文本中的人名
    extracted_names_set = set(extracted_names)  # 去重处理

    # 获取当前选中的模式名和对应的name_list
    selected_mode = mode_var.get()
    current_name_list = modes[selected_mode]

    # 找出 name_list 中有但提取内容中没有的人名
    not_in_text_names = [name for name in current_name_list if name not in extracted_names_set]

    # 清空结果框，并显示结果
    result_box.delete("1.0", tk.END)
    
    if not_in_text_names:
        result_box.insert(tk.END, f"{selected_mode} 模式下，没有完成的同学有：\n" + "\n".join(not_in_text_names))
    else:
        result_box.insert(tk.END, f"在 {selected_mode} 模式下，大家都完成啦。")

# 创建主窗口
root = tk.Tk()
root.title("人名提取与比对工具")

# 添加Logo
# icon = tk.PhotoImage(file="logo.png")
# root.iconphoto(False, icon)  # 设置窗口图标

# 模式选择下拉菜单
mode_var = tk.StringVar(value="全体同学")  # 默认选择第一个模式
mode_menu = tk.OptionMenu(root, mode_var, *modes.keys())
mode_menu.pack(pady=10)

# 粘贴内容的文本框
text_box = scrolledtext.ScrolledText(root, width=50, height=10)
text_box.pack(pady=10)

# 检查按钮
check_button = tk.Button(root, text="检查名字", command=check_names)
check_button.pack(pady=5)

# 显示结果的文本框
result_box = scrolledtext.ScrolledText(root, width=50, height=10)
result_box.pack(pady=10)

# 运行主循环
root.mainloop()
