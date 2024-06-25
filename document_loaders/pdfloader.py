from document_loaders.interface import Pipeline
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.staging.base import elements_to_json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from pydantic import create_model
from typing import List
from rich.progress import Progress, SpinnerColumn, TextColumn
import tempfile
import json
import warnings
import yaml
import timeit
from rich import print
from typing import Any
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class UnstructuredLightPipeline(Pipeline):

    async def run_pipeline(self,
                           file_path: str,
                           options: List[str] = None,
                           debug: bool = False,
                           local: bool = True) -> Any:
        print(f"\nRunning pipeline with unstructured_langchain \n")

        strategy = 'hi_res'
        model_name = 'yolox'

        extract_tables = False
        # Initialize options as an empty list if it is None
        options = options or []
        if "tables" in options:
            extract_tables = True

        # Extracts the elements from the PDF
        elements = self.invoke_pipeline_step(
            lambda: self.process_file(file_path, strategy, model_name),
            "Extracting elements from the document...",
            local
        )

        if debug:
            new_extension = 'json'  # You can change this to any extension you want
            new_file_path = self.change_file_extension(file_path, new_extension)

            documents = self.invoke_pipeline_step(
                lambda: self.load_text_data(elements, new_file_path, extract_tables),
                "Loading text data...",
                local
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "file_data.json")

                documents = self.invoke_pipeline_step(
                    lambda: self.load_text_data(elements, temp_file_path, extract_tables),
                    "Loading text data...",
                    local
                )

        docs = self.invoke_pipeline_step(
            lambda: self.split_text(documents, chunk_size=200, overlap=50),
            "Splitting text...",
            local
        )

        return docs

    def process_file(self, file_path, strategy, model_name):
        elements = None

        if file_path.lower().endswith('.pdf'):
            elements = partition_pdf(
                filename=file_path,
                strategy=strategy,
                infer_table_structure=True,
                model_name=model_name
            )
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            elements = partition_image(
                filename=file_path,
                strategy=strategy,
                infer_table_structure=True,
                model_name=model_name
            )

        return elements

    def load_text_data(self, elements, file_path, extract_tables):
        # 手动将元素保存到 JSON 文件中，确保使用 ensure_ascii=False
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([e.to_dict() for e in elements], file, ensure_ascii=False)

        # elements_to_json(elements, filename=file_path)
        text_file = self.process_json_file(file_path, extract_tables)

        loader = TextLoader(text_file)
        documents = loader.load()

        return documents

    def split_text(self, text, chunk_size, overlap):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        docs = text_splitter.split_documents(text)

        return docs

    def process_json_file(self, input_data, extract_tables):
        # Read the JSON file
        with open(input_data, 'r', encoding="utf-8") as file:
            data = json.load(file)

        # Iterate over the JSON data and extract required table elements
        extracted_elements = []
        for entry in data:
            if entry["type"] == "Table":
                extracted_elements.append(entry["metadata"]["text_as_html"])
            elif entry["type"] == "Title" and extract_tables is False:
                extracted_elements.append(entry["text"])
            elif entry["type"] == "NarrativeText" and extract_tables is False:
                extracted_elements.append(entry["text"])
            elif entry["type"] == "UncategorizedText" and extract_tables is False:
                extracted_elements.append(entry["text"])

        # Write the extracted elements to the output file
        new_extension = 'txt'  # You can change this to any extension you want
        new_file_path = self.change_file_extension(input_data, new_extension)
        with open(new_file_path, 'w') as output_file:
            for element in extracted_elements:
                output_file.write(element + "\n\n")  # Adding two newlines for separation

        return new_file_path

    def change_file_extension(self, file_path, new_extension):
        # Check if the new extension starts with a dot and add one if not
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension

        # Split the file path into two parts: the base (everything before the last dot) and the extension
        # If there's no dot in the filename, it'll just return the original filename without an extension
        base = file_path.rsplit('.', 1)[0]

        # Concatenate the base with the new extension
        new_file_path = base + new_extension

        return new_file_path

    def beautify_json(self, result):
        try:
            # Convert and pretty print
            data = json.loads(str(result))
            data = json.dumps(data, indent=4)
            return data
        except (json.decoder.JSONDecodeError, TypeError):
            print("The response is not in JSON format:\n")
            print(result)

        return {}

    def invoke_pipeline_step(self, task_call, task_description, local):
        if local:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
            ) as progress:
                progress.add_task(description=task_description, total=None)
                ret = task_call()
        else:
            print(task_description)
            ret = task_call()

        return ret


