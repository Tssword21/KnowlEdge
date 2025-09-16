"""
简历读取模块
支持多种格式简历文件的读取和文本提取
"""
import os
import logging
import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image

class ResumeReader:
    """简历读取器，支持多种格式"""

    def __init__(self):
        """初始化简历读取器"""
        self.supported_formats = {
            "txt": self.read_txt,
            "pdf": self.read_pdf,
            "docx": self.read_docx,
            "doc": self.read_doc,
            "xlsx": self.read_excel,
            "xls": self.read_excel,
            "jpg": self.read_image,
            "jpeg": self.read_image,
            "png": self.read_image
        }
        logging.info("简历读取器初始化完成")

    def read_resume(self, file_path=None):
        """
        读取简历文件

        Args:
            file_path: 文件路径，如果为None则请求用户输入

        Returns:
            简历文本内容
        """
        if file_path is None:
            return self.ask_for_input()

        # 检查文件是否存在
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return None

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip(".")

        # 检查是否支持此格式
        if ext in self.supported_formats:
            try:
                logging.info(f"正在读取 {ext} 格式文件: {file_path}")
                text = self.supported_formats[ext](file_path)
                if text:
                    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                    logging.info(f"成功读取简历，共 {len(text)} 个字符")
                    return text
                else:
                    logging.warning(f"文件内容为空: {file_path}")
                    return ""
            except Exception as e:
                logging.error(f"读取文件时出错: {e}")
                return None
        else:
            logging.error(f"不支持的文件格式: {ext}")
            return None

    def ask_for_input(self):
        """请求用户直接输入简历内容"""
        print("\n请直接输入或粘贴简历内容，完成后输入 ### 并回车:")
        lines = []
        while True:
            line = input()
            if line.strip() == "###":
                break
            lines.append(line)
        
        resume_text = "\n".join(lines)
        logging.info(f"用户输入了简历内容，共 {len(resume_text)} 个字符")
        return resume_text

    def read_txt(self, file_path):
        """读取文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def read_pdf(self, file_path):
        """读取PDF文件"""
        text_parts = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logging.warning(f"PDF第{page_num}页解析失败: {e}")
                    continue
        return ("\n".join(text_parts)).strip()

    def read_docx(self, file_path):
        """读取Word docx文件"""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def read_doc(self, file_path):
        """读取Word doc文件"""
        logging.warning("doc格式支持有限，可能无法正确提取文本")
        # 简单实现，实际应用中可能需要使用其他库
        return f"无法直接读取doc格式，请转换为docx后再试。文件路径: {file_path}"

    def read_excel(self, file_path):
        """读取Excel文件"""
        df = pd.read_excel(file_path)
        return df.to_string()

    def read_image(self, file_path):
        """读取图片文件并进行OCR识别"""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            return text
        except Exception as e:
            logging.error(f"OCR识别出错: {e}")
            return f"OCR识别失败: {str(e)}" 