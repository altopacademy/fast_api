from fastapi import FastAPI , Request

import random
import json
from fastapi.params import Depends
from fastapi.responses import JSONResponse

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from pydantic import BaseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    
class Item(BaseModel):
    pertanyaan: str

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

app = FastAPI()

inventory = {
    1: {
        "name": "Milk",
        "price": 3.99,
        "brand": "Regular"
    }
}

@app.get("/")
def home():
    return {"Data": "Test"}

@app.get("/about")
def about():
    return {"Data": "About"}

@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    return inventory[item_id]


@app.get("/tanya/")
def respon(pertanyaan: str):
    print(pertanyaan)
    sentence = tokenize(pertanyaan)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob)
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                return {"Jawaban": (intent['responses'])}
    else:
        nama = pertanyaan.replace("x", "*")
        nama2 = nama.replace(":", "/")
        nama2 = nama.replace(" ", "+")
        print(nama2)

        try:
            print(eval(nama2))
            pass
            return eval(nama2)
        except:
             print("Maaf ya, Edubot belum tau jawabannya :(")
             return {"Maaf ya, Edubot belum tau jawabannya :("}
    
    
    
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )