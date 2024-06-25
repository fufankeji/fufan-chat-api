# 确保已安装PyPDF2模块
try:
    import PyPDF2
except ImportError:
    import sys

    sys.exit("Please install the PyPDF2 module first, using: pip install PyPDF2")


def extract_text_from_pdf(filename, page_num):
    try:
        with open(filename, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    return text
                else:
                    return "No text found on this page."
            else:
                return f"Page number {page_num} is out of range. This document has {len(reader.pages)} pages."
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    # 示例用法
    filename = "/home/00_rag/fufan-chat-api/data/parse/data/1706.03762v7.pdf"
    page_num = 5
    text = extract_text_from_pdf(filename, page_num)

    print('--------------------------------------------------')
    print(f"Text from file '{filename}' on page {page_num}:")
    print(text if text else "No text available on the selected page.")
    print('--------------------------------------------------')
