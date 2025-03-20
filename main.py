from fastapi import FastAPI, File, UploadFile,Form,  Query
from fastapi.staticfiles import StaticFiles
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import shutil
from io import BytesIO
from PIL import Image
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from pydantic import BaseModel
# import easyocr
import re
import os
import pandas as pd
import requests
from modules import get_bank_name,get_ifsc_code,get_accnt_no,get_cheque_number,get_bank_details

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class RequestData(BaseModel):
    name: str
    
class BankDetailsRequest(BaseModel):
    ifsc: str
    account_number: str
    cheque_no: str
# reader = easyocr.Reader(['en'],gpu=True)
# model_path=r".\model\best.pt"
model_path = r"./model/best.pt"
model=YOLO(model_path)

razorpay_url = 'https://ifsc.razorpay.com/'
                
                
                



# Serve React build files


@app.post("/process/")
async def read_path(files: List[UploadFile] = File(...)):
    response_data = {}

    for file in files:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))

        print(type(image))
        data = {}

        image_array_list = []
        bounding_boxes_list = []
        df=pd.DataFrame()

        image_array_list.append(image)
        # Read the image
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Thresholding Segmentation
        _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Predit the AOI from the image
        pred_img=model.predict(source=image_array)
        
        
        a=pred_img[0].orig_img
        # print(pred_img[0].boxes.conf)
        img_org=cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        
        bnk_nme,ifsc_code,acc_no,cheq_no = '','','',''
        
        
        #     # Crop the image wrt the coordinates
        l1 = [float(i) for i in pred_img[0].boxes.cls]
        l2 = [float(i) for i in pred_img[0].boxes.conf]
        results = {}
        for ind,(cls, cnf_scr) in enumerate(zip(l1, l2)):
            cls_name=pred_img[0].names[cls]
            
            if cls_name not in results or cnf_scr > results[cls_name]['cnf']:
                results[cls_name] = {'cnf':cnf_scr,'coordinates':pred_img[0].boxes[ind]}
        bounding_boxes = []
        for result in results.items():
            box = [int(result) for result in result[1]['coordinates'].xyxy[0]]
            bounding_boxes.append(box)
            cls_name = result[0] 
            x_min=int(result[1]['coordinates'].xyxy[0][0])
            y_min=int(result[1]['coordinates'].xyxy[0][1])
            x_max=int(result[1]['coordinates'].xyxy[0][2])
            y_max=int(result[1]['coordinates'].xyxy[0][3])
            cropped_img = img_org[y_min:y_max, x_min:x_max]
            


            # Detect the text using OCR from the respective class.
            if cls_name=='bank_name':
                bnk_nme=get_bank_name(cropped_img)
            elif cls_name=='ifsc_Code':
                print(y_min,y_max, x_min,x_max)
                ifsc_code=get_ifsc_code(cropped_img)
            elif cls_name=='account_number':
                acc_no=get_accnt_no(cropped_img)
            elif cls_name=='cheque_number':
                cheq_no=get_cheque_number(cropped_img)
        
        # Concolidate the extracted text into rows and columns format.
        bounding_boxes_list.append(bounding_boxes)
        if len(bnk_nme)==0:
            bnk_nme= ''
        if len(ifsc_code)==0:
            ifsc_code= ''
        if len(acc_no)==0:
            acc_no= ''
        if len(cheq_no)==0:
            cheq_no= ''
        min_df = pd.DataFrame()
        min_df['Image Name'] = [file.filename]
        min_df['Bank Name'] = bnk_nme
        min_df['IFSC'] = ifsc_code
        min_df['Account No'] = acc_no
        min_df['Cheque No'] = cheq_no
        df = pd.concat([df,min_df])

    # Display the uploaded image
        # image = Image.open(img_path)
        # img_data = main(image)
        img_data = {'Bank_Name':bnk_nme,"IFSC_Code":ifsc_code,'Account_Number':acc_no,"Cheque_No":cheq_no}
        # data.update({filename:img_data})
        response_data[file.filename] = img_data
        print(response_data)
    return response_data



@app.post("/get_bank_details/")
def get_bank_details(data:BankDetailsRequest):
    bank_details = {"IFSC":None,"ACCOUNT_NUMBER":None,"CHEQUE_NUMBER":None,"BANK":None,"ADDRESS":None,
            "CENTRE":None,"DISTRICT":None,"STATE":None,"CONTACT":None,
            "MICR":None,"STATE":None,"ISO3166":None,"CITY":None,"NEFT":None,
            "IMPS":None,"UPI":None,"BRANCH":None,"RTGS":None,"BANKCODE":None}
     
    print("data.ifsc==>>",data.ifsc)
    
    if len(data.ifsc)==0:
        return bank_details
    response = requests.get(razorpay_url+data.ifsc)
    
    result = response.json()
    print("==>>",result)
    if "Not Found" in result:
        bank_details = {"IFSC":None,"ACCOUNT_NUMBER":None,"CHEQUE_NUMBER":None,"BANK":None,"ADDRESS":None,
            "CENTRE":None,"DISTRICT":None,"STATE":None,"CONTACT":None,
            "MICR":None,"STATE":None,"ISO3166":None,"CITY":None,"NEFT":None,
            "IMPS":None,"UPI":None,"BRANCH":None,"RTGS":None,"BANKCODE":None}  
    
    else:
        bank_details = {"IFSC":result['IFSC'],"ACCOUNT_NUMBER":data.account_number,"CHEQUE_NUMBER":data.cheque_no,"BANK":result['BANK'],"ADDRESS":result['ADDRESS'],
            "CENTRE":result['CENTRE'],"DISTRICT":result['DISTRICT'],"STATE":result['STATE'],"CONTACT":result['CONTACT'],
            "MICR":result['MICR'],"STATE":result['STATE'],"ISO3166":result['ISO3166'],"CITY":result['CITY'],"NEFT":str(result['NEFT']),
            "IMPS":str(result['IMPS']),"UPI":str(result['UPI']),"BRANCH":result['BRANCH'],"RTGS":str(result['RTGS']),"BANKCODE":result['BANKCODE']}    
    return bank_details
    
    # return bank_details

app.mount("/", StaticFiles(directory="build", html=True), name="react")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)