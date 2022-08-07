from typing import Union
import json

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

from preprocessing import clean_preprocessing
from featureprocessing import feature_processing
from model import model
from transaction_group import group_transaction

app = FastAPI(
    title="Mono Data Scientist Assessment",
    description="Question Link : `https://withmono.notion.site/Data-Scientist-Assessment-1-4d779e12adb64f53acaa1c7019ac4840`"

)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/group_similar_transaction", tags=["Endpoint"])
def group_similar_transactions(mono_sec_key: str, period: str, account_id: str):
    url = f"https://api.withmono.com/accounts/{account_id}/statement?period={period}"

    headers = {
        "Accept": "application/json",
        "mono-sec-key": mono_sec_key
    }

    response = requests.get(url, headers=headers)
    meta_data = json.loads(response.text)

    raw_data = meta_data['data']

    cleaned_data = clean_preprocessing(raw_data)
    train_data = feature_processing(transactions_data=cleaned_data)
    transactions_data = model(train_data=train_data,
                              transactions_data=cleaned_data)
    transactions_groups = group_transaction(transactions_data)

    json_compatible_item_data = jsonable_encoder(
        {"status": "success", 'data': transactions_groups})
    return JSONResponse(content=json_compatible_item_data)
