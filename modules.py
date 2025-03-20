__author__ = "Saurabh"
__version__ = "1.0.0"
__date__ = "2024-12-16"
__description__ = "This Script is the OCR implementation of the application."


import cv2
import re
import pytesseract
import numpy as np
import pandas as pd
import requests
import json

from io import BytesIO
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'.\Tesseract-OCR\tesseract.exe'

def get_bank_name(cropped_img)-> str:
    """
    This function takes the image as an input, extracts the text from the image and returns the string
    
    parameters
    ----------
    
    cropped_img: array 
                Image from which the text needs to be extracted. 
    
    
    Returns
    -------
    str
       The string extracted from the input image
    """
    # temp_path = 'temp_cropped_image.jpg'
    # cv2.imwrite(temp_path,cropped_img)
    # image = Image.open(temp_path)
    result1 = pytesseract.image_to_string(cropped_img,lang="eng")

    _, bw_image = cv2.threshold(cropped_img, 128, 255, cv2.THRESH_BINARY)
    result2 = pytesseract.image_to_string(bw_image,lang="eng")
    
    result = max([result1,result2], key=len) 
    bnk_nam=result.replace(' ','')
    return bnk_nam


def get_ifsc_code(cropped_img)-> str:
    """
    This function takes the image as an input, extracts the text from the image and returns the string
    
    parameters
    ----------
    
    cropped_img: array 
                Image from which the text needs to be extracted. 
    
    
    Returns
    -------
    str
       The string extracted from the input image
    """    
    # temp_path = 'temp_cropped_image.jpg'
    # cv2.imwrite(temp_path,cropped_img)
    # temp_path = 'temp_cropped_image1.jpg'
    # cv2.imwrite(temp_path,cropped_img)
    # image = Image.open(temp_path)
    result = pytesseract.image_to_string(cropped_img,lang="eng")
    c=result.replace(' ','').replace('\n','').replace(':',' ').replace('-',' ').upper()
    print("===>>>",c)
    ifsc_no=re.findall(r'([A-Z0-9]{11})',c)
    if len(ifsc_no)==0:
        ifsc_no=re.findall(r'([A-Z0-9]{10})',c)
    # if len(ifsc_no)==0:
    #     ifsc_no=re.findall(r'([A-Z]{10})',c)
    if len(ifsc_no)==0:
        ifsc_no=re.findall(r'([A-Z0-9]{09})',c)
    if len(ifsc_no)>0:
        ifsc_no=ifsc_no[-1]
    return ifsc_no


def get_accnt_no(cropped_img)-> str:
    """
    This function takes the image as an input, extracts the text from the image and returns the string
    
    parameters
    ----------
    
    cropped_img: array 
                Image from which the text needs to be extracted. 
    
    
    Returns
    -------
    str
       The string extracted from the input image
    """
    # temp_path = 'temp_cropped_image.jpg'
    # cv2.imwrite(temp_path,cropped_img)
    # image = Image.open(temp_path)
    result1 = pytesseract.image_to_string(cropped_img,lang="eng")
    c=result1.replace(' ','')
    e=re.findall('[0-9]*',c)
    acc_no1 = max(e, key=len)

    _, bw_image = cv2.threshold(cropped_img, 128, 255, cv2.THRESH_BINARY)
    
    result2 = pytesseract.image_to_string(bw_image,lang="eng")
    c=result2.replace(' ','')
    e=re.findall('[0-9]*',c)
    acc_no2 = max(e, key=len)
    

    

    
    acc_no = max([acc_no1,acc_no2], key=len)
    
            
    return acc_no


def get_cheque_number(cropped_img)-> str:
    """
    This function takes the image as an input, extracts the text from the image and returns the string
    
    parameters
    ----------
    
    cropped_img: array 
                Image from which the text needs to be extracted. 
    
    
    Returns
    -------
    str
       The string extracted from the input image
    """
    # temp_path = 'temp_cropped_image.jpg'
    # cv2.imwrite(temp_path,cropped_img)
    # image = Image.open(temp_path)
    result = pytesseract.image_to_string(cropped_img,lang="mcr")
    c=result.replace(' ','')
    e=re.findall('[0-9]*',c)
    cheq_no = max(e, key=len)
    return cheq_no

def get_bank_details(ifsc_code):

    if ifsc_code:
        
        url = f"https://ifsc.razorpay.com/{ifsc_code}"

        response = requests.get(url)

        if response.status_code==200:
            data = json.loads(response.content)
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()
            df['IFSC'] = [ifsc_code]
        return df
    
    else:
        
        df = pd.DataFrame()
        df['IFSC'] = [ifsc_code]
        return df