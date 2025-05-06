import os
import subprocess
from pydub import AudioSegment
import re

# 音频转换函数
def convert_audio(input_path, output_path, target_sample_rate=16000):
    try:
        audio = AudioSegment.from_wav(input_path)
        audio = audio.set_frame_rate(target_sample_rate)
        audio.export(output_path, format="wav")
        print(f"音频文件已转换为 {target_sample_rate} Hz，保存为: {output_path}")
    except Exception as e:
        print(f"音频转换失败: {e}")

# 执行语音识别并提取结果
def recognize_audio(wav_file):
    command = f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/build/bin/Release/sherpa-ncnn.exe ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param ' \
              f'D:/Programs/Codes/XDU/Projects/Sound/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin ' \
              f'"{wav_file}"'
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # 打印输出和错误，帮助排查问题
    print(f"命令输出: {result.stdout}")
    print(f"命令错误: {result.stderr}")
    
    output = result.stdout

    # 提取英文文本部分
    start_index = output.find("text:") + len("text:")  # 从"text:"之后开始提取
    if start_index != -1:
        text = output[start_index:].strip()
        return text
    else:
        print(f"未能在输出中找到文本：{output}")
        return ""

# 处理单词列表：去除数字、时间戳，转换为小写
def process_word_list(text):
    # 使用正则去掉数字和时间戳，保留单词部分
    words = re.findall(r'[a-zA-Z]+', text)
    return [word.lower() for word in words]

# 批量处理音频文件并将所有结果写入一个txt文件
def process_audio_files(input_dir, output_txt_file):
    word_list_counter = 1  # 初始化编号
    with open(output_txt_file, 'w+', encoding='utf-8') as output_file:
        for filename in os.listdir(input_dir):
            if filename.endswith(".wav"):
                input_path = os.path.join(input_dir, filename)
                print(f"处理文件: {filename}")
                
                # 转换音频采样率
                output_audio_filename = filename.replace(".wav", " 16000.wav")
                output_audio_path = os.path.join(input_dir, output_audio_filename)
                convert_audio(input_path, output_audio_path)

                # 识别音频并返回文本
                text = recognize_audio(output_audio_path)
                
                if text:
                    # 处理识别结果并写入到一个单一文件
                    output_file.write(f'#Word List {word_list_counter}\n')
                    word_list = process_word_list(text)
                    for word in word_list:
                        output_file.write(f'{word}\n')
                    output_file.write("\n")  # 每个文件的内容之间空一行
                    
                    word_list_counter += 1  # 更新编号
                else:
                    print(f"未识别到文本: {filename}")

                print(f"处理完: {filename}")

# 设置音频文件夹路径
input_dir = "D:/Users/tang/Downloads/Compressed/6jlxbx21"  # 输入文件夹路径
output_txt_file = "D:/Users/tang/Downloads/Compressed/6jlxbx21/combined_word_list.txt"  # 输出文件路径

# 执行批量处理
process_audio_files(input_dir, output_txt_file)
