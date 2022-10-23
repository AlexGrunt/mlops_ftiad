from fastapi import FastAPI, status
from typing import List, Union
from app.api import schemas, crud

app = FastAPI()


@app.get(
    "/models/", response_model=List[schemas.ModelBase], status_code=status.HTTP_200_OK
)
def get_models():
    return [schemas.ModelBase.parse_raw(model) for model in crud.get_models()]


@app.get("/models/info/", status_code=status.HTTP_200_OK)
def get_info():
    return crud.get_info()


@app.put("/model/refit/", status_code=status.HTTP_200_OK)
def refit_model(
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
):
    return crud.refit_model(model)


@app.post("/model/fit/", status_code=status.HTTP_200_OK)
def fit_model(
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
):
    return crud.model_fit(model)


@app.delete("/model/delete/", status_code=status.HTTP_200_OK)
def delete_model(model: schemas.ModelBase):
    return crud.delete_model(model)


@app.get(
    "/models/predict/",
    response_model=schemas.Prediction,
    status_code=status.HTTP_200_OK,
)
def predict_model(model_name: str, model_class: str, classification: bool):
    return schemas.Prediction(prediction=[
        str(pred)
        for pred in crud.get_predictions(
            schemas.ModelBase(
                name=model_name, model_class=model_class, classification=classification
            )
        )
    ])
