import os
import uuid
import pandas as pd
import glob
import re
import pymupdf4llm


# 假定 PDF 文件转换函数，真实环境中需要使用你的 pymupdf4llm 模块
def convert_pdf_to_markdown(pdf_path):
    # 这里是一个模拟函数，实际中你需要使用你的转换函数
    doc = pymupdf4llm.to_markdown(pdf_path)
    return doc


def extract_toc(markdown):
    toc = []
    pattern = r'^(#+)\s+(.*)$'
    lines = markdown.split('\n')
    for line in lines:
        match = re.match(pattern, line)
        if match:
            level = len(match.group(1)) - 1
            title = match.group(2)
            toc.append((level, title))
    return toc


def extract_content_by_sections(markdown):
    sections = {}
    current_section = None
    content_accumulator = []

    lines = markdown.split('\n')
    pattern = r'^(#+)\s+(.*)$'

    for line in lines:
        match = re.match(pattern, line)
        if match:
            level = len(match.group(1)) - 1
            title = match.group(2)
            if level == 0:  # 一级标题
                if current_section is not None:
                    # 将之前一级标题下累积的内容存储起来
                    sections[current_section] = "\n".join(content_accumulator).strip()
                # 更新当前一级标题并重置内容累积器
                current_section = title
                content_accumulator = []
            elif current_section is not None:
                # 累积当前一级标题下的内容
                content_accumulator.append(line)
        elif current_section is not None:
            # 继续累积内容，即使它不是标题
            content_accumulator.append(line)

    # 确保最后一部分也被添加
    if current_section is not None:
        sections[current_section] = "\n".join(content_accumulator).strip()

    return sections


def clean_tags(title):
    # 使用正则表达式去掉前面的数字和点
    cleaned_title = re.sub(r'^\d+\.\d+\s+', '', title)
    return cleaned_title


def generate_csv_for_pdfs(root_dir):
    pdf_files = glob.glob(f'{root_dir}/**/*.pdf', recursive=True)
    data = []

    for pdf_file in pdf_files:
        # 假定的 PDF to Markdown 转换函数
        markdown = convert_pdf_to_markdown(pdf_file)
        toc_content = extract_content_by_sections(markdown)

        # Assume the first section is the title
        titles = list(toc_content.keys())
        first_title = titles[0] if titles else ""
        # Collect second level titles under the first title as sub-titles
        first_section_content = toc_content.get(first_title, "")
        first_section_lines = first_section_content.split('\n')
        sub_titles = [line.strip() for line in first_section_lines if line.startswith('##')]
        sub_titles_cleaned = [re.sub(r'^##\s+', '', title) for title in sub_titles]

        for module_name, content in toc_content.items():
            # 提取二级标题作为 Tags
            tags = [line.strip() for line in content.split('\n') if line.startswith('##')]
            tags = [re.sub(r'^##\s+', '', tag) for tag in tags]  # 清理 '##'

            data.append({
                'ModuleID': str(uuid.uuid4()),
                'Course': os.path.basename(os.path.dirname(pdf_file)),
                'Title': sub_titles_cleaned,
                'URL': os.path.basename(pdf_file),
                'ModuleName': module_name,
                'Tags': ", ".join(tags),
                'Content': content
            })

    df = pd.DataFrame(data)
    csv_file_path = os.path.join(root_dir, 'output.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file generated: {csv_file_path}")



# 假设这是你用于生成摘要的函数
def generate_summary(text):
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": "作为一名语言学专家，请根据如下的输入文本:\n{} \n生成一段摘要".format(text)}]
    )

    return response.choices[0].message.content


# 假设这是你用于生成 Tags 的函数
def generate_tags(prompt):
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[{"role": "user",
                   "content": "作为一名语言学专家，请根据类似的示例:\n{} \n生成这段文本的关键性标签".format(prompt)}]
    )

    return response.choices[0].message.content



if __name__ == '__main__':
    # 基于原始PDF文档，生成格式化的.csv文件
    root_directory = "./course_data"
    generate_csv_for_pdfs(root_directory)

    # 利用大模型做推荐数据的特征工程
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key='2bdc6a66003c448913f9617906c95e9a.0KvV4nPTg9XqhLRj')

    # 加载 CSV 文件
    df = pd.read_csv("./course_data/output.csv")

    # 遍历 DataFrame，生成摘要
    df['Abstract'] = df['Content'].apply(generate_summary)

    # 检查并填充空的 Tags
    for index, row in df.iterrows():
        if pd.isna(row['Tags']) or row['Tags'].strip() == "":
            # 查找具有相同 URL 的其他记录的 Tags
            similar_tags = df[df['URL'] == row['URL']]['Tags'].dropna()
            if not similar_tags.empty:
                tag_prompt = similar_tags.iloc[0]  # 使用第一个非空标签作为提示
                generated_tags = generate_tags(tag_prompt)
                df.at[index, 'Tags'] = generated_tags

    # 保存更新后的 DataFrame 到新的 CSV 文件
    df.to_csv("./course_data/updated_output.csv", index=False)

