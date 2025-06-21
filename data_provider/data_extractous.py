# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_extract.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-02
# * Version     : 1.0.050219
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
# global variable
LOGGING_LABEL = __file__.split('\\')[-1][:-3]
os.environ["LOG_NAME"] = LOGGING_LABEL

from extractous import Extractor, TesseractOcrConfig

from utils.log_util import logger


class data_extract:
    
    def __init__(self, 
                 input_file_path: str, output_file_path: str, 
                 extract_method: str = "OCR", lang: str = None, 
                 string_max_length: int = None,
                 quite: bool = True):
        # file_path = "README.md"
        # file_path = "test.pdf"
        # file_path = "https://www.google.com"
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.extract_method = extract_method.lower()
        self.lang = lang
        self.string_max_length = string_max_length
        self.quite = quite
    
    def _build_extractor(self):
        # create a new extractor
        self.extractor = Extractor()

        # config extract string max length
        if self.string_max_length is not None:
            self.extractor = self.extractor_file.set_extract_string_max_length(1000)
        
        # output an xml
        if self.output_file_path.endswith("xml"):
            self.extractor = self.extractor_file.set_xml_output(True)
        
        # extracting a file with OCR
        if self.extract_method == "ocr":
            self.extractor = self.extractor.set_ocr_config(TesseractOcrConfig().set_language("deu"))

    def extract_file_to_string(self):
        """
        Extract text from a file
        """
        result, metadata = self.extractor.extract_file_to_string(self.input_file_path)
        logger.info(f"result: \n{result}")
        logger.info(f"metadata: \n{metadata}")

    def extract_file_to_buffered_stream(self, file_type: str):
        """
        Extracting a file(URL/bytearray) to a buffered stream
        """
        # for file
        if file_type == "file":
            reader, metadata = self.extractor.extract_file(self.input_file_path)
        # for url
        elif file_type == "url":
            reader, metadata = self.extractor.extract_url(self.input_file_path)
        # for bytearray
        elif file_type == "bytearray":
            with open(self.file_path, "rb") as file:
                buffer = bytearray(file.read())
            reader, metadata = self.extractor.extract_bytes(buffer)
        # reulst
        result = ""
        buffer = reader.read(4096)
        while len(buffer) > 0:
            result += buffer.decode("utf-8")
            buffer = reader.read(4096)
        logger.info(f"result: \n{result}")
        logger.info(f"metadata: \n{metadata}")

    def extract_file_with_OCR(self):
        """
        Extracting a file with OCR
        """
        result, metadata = self.extractor.extract_file_to_string(self.file_path)
        logger.info(f"result: \n{result}")
        logger.info(f"metadata: \n{metadata}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
