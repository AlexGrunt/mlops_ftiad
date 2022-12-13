from fastapi import FastAPI, status
from typing import List, Union
from app.api import schemas, crud, models
from app.api.database import SessionLocal, engine
from app.core.config import SCHEMA_NAME
from sqlalchemy.schema import CreateSchema


app = FastAPI()

class SessionManager:
    def __init__(self):
        self.db = SessionLocal()

    def __enter__(self):
        return self.db

    def __exit__(self, _, _a, _b):
        self.db.close()


@app.on_event("startup")
async def startup_event():
    if not engine.dialect.has_schema(engine, SCHEMA_NAME):
        engine.execute(CreateSchema(SCHEMA_NAME))
    models.Base.metadata.create_all(bind=engine)
    crud.do_etl()


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

@app.post("/model/fit/db", status_code=status.HTTP_200_OK)
def fit_model_db(model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]):
    with SessionManager() as session_local:
        return crud.model_fit(model, db=True, session=session_local)