import tempfile
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
import json
from unstructured.staging.base import elements_to_json
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print
from bs4 import BeautifulSoup


class UnstructuredProcessor(object):
    def __init__(self):
        # 构造函数：初始化UnstructuredProcessor实例
        pass

    def extract_data(self, file_path, strategy, model_name, options, local=True, debug=False):
        """
        从指定的文件中提取数据。

        :param file_path: str，文件的路径，指定要处理的文件。
        :param strategy: 使用的策略来提取数据。
        :param model_name: 使用的模型名称，这里使用 目标检测模型 yolox
        :param options: dict，额外的选项或参数，用来干预数据提取的过程或结果。
        :param local: bool，一文件处理是否应在本地执行，默认为True。
        :param debug: bool，如果设置为True，则会显示更多的调试信息，帮助理解处理过程中发生了什么，默认为False。

        函数的执行流程：
        - 调用`invoke_pipeline_step`方法，这是一个高阶函数，它接受一个lambda函数和其他几个参数。
        - lambda函数调用`process_file`方法，处理文件并根据指定的策略和模型名提取数据。
        - `invoke_pipeline_step`方法除了执行传入的lambda函数，还可能处理本地执行逻辑，打印进程信息，并依据`local`参数决定执行环境。
        - 最终，数据提取的结果将从`process_file`方法返回，并由`invoke_pipeline_step`方法输出。
        """

        # # 调用数据提取流程，处理PDF文件并提取元素
        elements = self.invoke_pipeline_step(
            lambda: self.process_file(file_path, strategy, model_name),
            "Extracting elements from the document...",
            local
        )

        if debug:
            new_extension = 'json'  # You can change this to any extension you want
            new_file_path = self.change_file_extension(file_path, new_extension)

            content, table_content = self.invoke_pipeline_step(
                lambda: self.load_text_data(elements, new_file_path, options),
                "Loading text data...",
                local
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "file_data.json")

                content, table_content = self.invoke_pipeline_step(
                    lambda: self.load_text_data(elements, temp_file_path, options),
                    "Loading text data...",
                    local
                )

        if debug:
            print("Data extracted from the document:")
            print(content)
            print("\n")
            print("Table content extracted from the document:")
            if table_content:
                print(len(table_content))
            print(table_content)

        print(f"这是content:{content}")
        print(f"这是table_content:{table_content}")
        return content, table_content

    def process_file(self, file_path, strategy, model_name):
        """
        处理文件并提取数据，支持PDF文件和图像文件。

        :param file_path: str，文件的路径，指定要处理的文件。
        :param strategy: 使用的策略来提取数据，影响数据处理的方法和结果。
        :param model_name: 使用的模型名称，这里使用yolox

        方法的执行流程：
        - 初始化`elements`变量为None，用来存储提取的元素。
        - 检查文件路径的后缀，根据文件类型调用相应的处理函数：
          - 如果文件是PDF（.pdf），使用`partition_pdf`函数处理：
            - `filename`：提供文件路径。
            - `strategy`：指定数据提取策略。
            - `infer_table_structure`：是否推断表格结构，这里设为True。
            - `hi_res_model_name`：提供高分辨率模型名称。
            - `languages`：设置处理的语言为英语。
          - 如果文件是图像（.jpg, .jpeg, .png），使用`partition_image`函数处理，参数类似于处理PDF的参数。
        - 返回提取的元素`elements`。

        :return: 返回从文件中提取的元素。
        """

        # 初始化元素变量
        elements = None
        # 根据文件后缀决定处理方式
        # partition_pdf 官方文档：https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf

        # hi_res 策略配合 infer_table_structure=True 的表格识别效果较好
        if file_path.lower().endswith('.pdf'):
            elements = partition_pdf(
                filename=file_path,
                # strategy kwarg 控制用于处理 PDF 的方法。 PDF 的可用策略有 "auto" 、 "hi_res" 、 "ocr_only" 和 "fast"
                strategy=strategy,
                infer_table_structure=True,
                hi_res_model_name=model_name,
                languages=['chi_sim']
            )
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 处理图像文件
            elements = partition_image(
                filename=file_path,
                strategy=strategy,
                infer_table_structure=True,
                hi_res_model_name=model_name,
                languages=['chi_sim']
            )

        return elements

    def change_file_extension(self, file_path, new_extension, suffix=None):
        # Check if the new extension starts with a dot and add one if not
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension

        # Split the file path into two parts: the base (everything before the last dot) and the extension
        # If there's no dot in the filename, it'll just return the original filename without an extension
        base = file_path.rsplit('.', 1)[0]

        # Concatenate the base with the new extension
        if suffix is None:
            new_file_path = base + new_extension
        else:
            new_file_path = base + "_" + suffix + new_extension

        return new_file_path

    def load_text_data(self, elements, file_path, options):
        # 手动将元素保存到 JSON 文件中，确保使用 ensure_ascii=False
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([e.to_dict() for e in elements], file, ensure_ascii=False)

        content, table_content = None, None

        if options is None:
            content = self.process_json_file(file_path)

        if options and "tables" in options and "unstructured" in options:
            content = self.process_json_file(file_path, "form")
            table_content = self.process_json_file(file_path, "table")

        return content, table_content

    def process_json_file(self, file_path, option=None):
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over the JSON data and extract required elements
        extracted_elements = []
        for entry in data:
            if entry["type"] == "Table" and (option is None or option == "table" or option == "form"):
                table_data = entry["metadata"]["text_as_html"]
                if option == "table" and self.table_has_header(table_data):
                    extracted_elements.append(table_data)
                if option is None or option == "form":
                    extracted_elements.append(table_data)
            elif entry["type"] == "Title" and (option is None or option == "form"):
                extracted_elements.append(entry["text"])
                # 叙述文本
            elif entry["type"] == "NarrativeText" and (option is None or option == "form"):
                extracted_elements.append(entry["text"])
                # 未分类
            elif entry["type"] == "UncategorizedText" and (option is None or option == "form"):
                extracted_elements.append(entry["text"])
            elif entry["type"] == "ListItem" and (option is None or option == "form"):
                extracted_elements.append(entry["text"])
            elif entry["type"] == "Image" and (option is None or option == "form"):
                extracted_elements.append(entry["text"])

        if option is None or option == "form":
            # Convert list to single string with two new lines between each element
            extracted_data = "\n\n".join(extracted_elements)
            return extracted_data
     
        return extracted_elements

    def invoke_pipeline_step(self, task_call, task_description, local):
        """
        执行管道步骤，可以在本地或非本地环境中运行任务。

        :param task_call: callable，一个无参数的函数或lambda表达式，它执行实际的任务。
        :param task_description: str，任务的描述，用于进度条或打印输出。
        :param local: bool，指示是否在本地环境中执行任务。如果为True，则使用进度条；如果为False，则仅打印任务描述。

        方法的执行流程：
        - 如果`local`为True，使用`Progress`上下文管理器来显示一个动态的进度条。
          - `SpinnerColumn()`：在进度条中添加一个旋转的指示器。
          - `TextColumn("[progress.description]{task.description}")`：添加一个文本列来显示任务描述。
          - `transient=False`：进度条显示完成后不会消失。
          - 在进度条中添加一个任务，然后调用`task_call()`执行实际的任务，任务的返回结果保存在`ret`中。
        - 如果`local`为False，则直接打印任务描述，不使用进度条，之后调用`task_call()`执行任务，任务的返回结果同样保存在`ret`中。

        :return: 返回从`task_call()`获取的结果。
        """
        if local:
            # 本地执行时，显示带有进度指示的进度条
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
            ) as progress:
                # 添加进度任务，总步长为None表示不确定的任务进度
                progress.add_task(description=task_description, total=None)
                # 调用task_call执行任务，并获取结果
                ret = task_call()
        else:
            print(task_description)
            ret = task_call()

        return ret

    def table_has_header(self, table_html):
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')

        # Check if the table contains a <thead> tag
        if table.find('thead'):
            return True

        # Check if the table contains any <th> tags inside the table (in case there's no <thead>)
        if table.find_all('th'):
            return True

        return False


if __name__ == "__main__":
    processor = UnstructuredProcessor()

    # 提取PDF中的表格数据
    content, table_content = processor.extract_data(
        '/home/00_rag/fufan-chat-api/data/parse/data/invoice_2.pdf',
        'hi_res',       # 
        'yolox',    # https://github.com/Megvii-BaseDetection/YOLOX
        ['tables', 'unstructured'],
        True,
        True)


