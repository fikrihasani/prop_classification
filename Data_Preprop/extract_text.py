import os
import sys
import re
from typing import Text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from io import BytesIO
import pandas as pd
from summarize import *

def load_data_to_df():
    df = pd.read_excel("Data/Labeling Skripsi S1.xlsx",sheet_name="Data")
    return df

def clean_text(text):
    text = text.replace('\\n','') if text != "" else ""
    text = text.replace('\\x0c','') if text != "" else ""
    return text

def pdf_to_text(path):
    # manager = PDFResourceManager()
    # retstr = BytesIO()
    # layout = LAParams(all_texts=True)
    # device = TextConverter(manager, retstr, laparams=layout)
    # filepath = open(path, 'rb')
    # interpreter = PDFPageInterpreter(manager, device)

    # for page in PDFPage.get_pages(filepath, check_extractable=True):
    #     interpreter.process_page(page)

    # text = retstr.getvalue()
    # filepath.close()
    # device.close()
    # retstr.close()
    # return text
    if not os.path.exists(path): return ""
    return repr(extract_text(path))

def get_data():
    datas = []
    ret = []
    for root, dirs, files in os.walk("Data"):
        for file in files:
            if file.endswith(".pdf"):
                datas.append(os.path.join(root, file))
    for data in datas:
        ret.append([data,data.split("\\")[2],data.split("\\")[-1] , ""])
    return ret

def get_texts():
    df_excel = load_data_to_df()
    df_excel["Text"] = df_excel.apply(lambda row :(clean_text(str(pdf_to_text("Data\\"+row["Nama Folder"]+"\\"+row["Nama File"]+"\\"+row["Judul"]+".pdf")))).lower().replace("\uf0b7",""), axis = 1)
    df_excel = df_excel[df_excel["Text"]!=""]
    print("summarizing 5 sentence")
    df_excel["Summarized_Text_5"] = df_excel.apply(lambda row:generate_summary(row["Text"],5,False), axis=1)
    df_excel.to_excel("Loaded_Data_Sum5.xlsx")
    print("summarizing 10 sentence")
    df_excel["Summarized_Text_10"] = df_excel.apply(lambda row:generate_summary(row["Text"],10,False), axis=1)
    df_excel.to_excel("Loaded_Data_Sum10.xlsx")
    print("summarizing 15 sentence")
    df_excel["Summarized_Text_15"] = df_excel.apply(lambda row:generate_summary(row["Text"],15,False), axis=1)
    df_excel.to_excel("Loaded_Data_Sum15.xlsx")
    print("summarizing 20 sentence")
    df_excel["Summarized_Text_20"] = df_excel.apply(lambda row:generate_summary(row["Text"],20,False), axis=1)
    df_excel.to_excel("Loaded_Data_Sum20.xlsx")
    df_excel.to_excel("Loaded_Data.xlsx")
    print("data saved")
    return df_excel