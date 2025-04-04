import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "encoder.pkl")
encoder = load_model(path)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.pkl")
model = load_model(path)

# Create API
app = FastAPI()


# Create welcome message
@app.get("/")
async def get_root():
    """
    GET root endpoint.

    Returns a welcome message to confirm the API is running.
    """
    return {"message": "Welcome to the income prediction API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """
        POST /data/ endpoint.

        Accepts JSON input data, processes it, and returns an income prediction result.

        Parameters:
            data (Data): Input features for model inference.

        Returns:
            dict: A dictionary with a single key "result" containing the predicted income label (<=50K or >50K).
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        training=False
    )
    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
