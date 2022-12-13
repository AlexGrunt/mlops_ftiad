import json
import os
from pathlib import Path
from typing import Union

import dill
import numpy as np
import pandas as pd
import sklearn
from app.api import models, schemas
from app.api.database import engine
from app.core.config import SCHEMA_NAME
from fastapi import HTTPException, status
from sklearn import linear_model
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_models():
    models = []

    pathlist = Path("app/configs/").glob("**/*.json")
    for path in pathlist:
        with open(path, "r") as file:
            model_dict = json.load(file)
            models.append(json.dumps(model_dict))

    return models


def model_exists(model: schemas.BaseModel):
    pathlist = Path("app/configs/").glob("**/*.json")
    for path in pathlist:
        with open(path, "r") as file:
            model_dict = json.load(file)

        if (
            model_dict["name"] == model.name
            and model_dict["model_class"] == model.model_class
        ):
            return True

    return False


def get_dataset(classification: bool, db, session):
    if db and session is not None:
        if classification:
            data = pd.read_sql(session.query(models.WineDataset).all(), engine)
        else:
            data = pd.read_sql(session.query(
                models.DiabetesDataset).all(), engine)
        return data.drop(columns=['target']), data['target']
    if classification:
        data = load_wine()
    else:
        data = load_diabetes()
    return data.data, data.target


def do_etl():
    wine = get_dataset(True)
    wine.to_sql('wine_dataset', con=engine, schema=SCHEMA_NAME,
                if_exists='replace', index=True)
    diabets = get_dataset(False)
    diabets.to_sql('diabetes_dataset', con=engine,
                   schema=SCHEMA_NAME, if_exists='replace', index=True)


def get_tree_model(model: schemas.DecisionTreeModel):
    if model.classification:
        return DecisionTreeClassifier(
            criterion=model.criterion,
            splitter=model.splitter,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state,
            max_leaf_nodes=model.max_leaf_nodes,
        )
    return DecisionTreeRegressor(
        criterion=model.criterion,
        splitter=model.splitter,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
        random_state=model.random_state,
        max_leaf_nodes=model.max_leaf_nodes,
    )


def get_lasso_model(model: schemas.LassoModel):
    return linear_model.Lasso(
        alpha=model.alpha,
        tol=model.tol,
        solver=model.solver,
        random_state=model.random_state,
        max_iter=model.max_iter,
    )


def get_ridge_model(model: schemas.RidgeModel):
    if model.classification:
        return linear_model.RidgeClassifier(
            alpha=model.alpha,
            tol=model.tol,
            solver=model.solver,
            random_state=model.random_state,
        )
    return linear_model.Ridge(
        alpha=model.alpha,
        tol=model.tol,
        solver=model.solver,
        random_state=model.random_state,
    )


def get_estimator(
    model: Union[schemas.RidgeModel,
                 schemas.LassoModel, schemas.DecisionTreeModel]
):
    if model.model_class == "RidgeModel":
        return get_ridge_model(model)
    elif model.model_class == "LassoModel":
        return get_lasso_model(model)
    elif model.model_class == "DecisionTreeModel":
        return get_tree_model(model)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incopatible model class check. Check compatible model class in documentation",
        )


def fit_and_save(
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel], db=False, session=None
):
    X, y = get_dataset(model.classification, db, session)

    estimator = get_estimator(model).fit(X, y)

    with open(f"app/models/{model.name}.dill", "wb") as file:
        dill.dump(estimator, file)

    meta_dict = {
        "name": model.name,
        "model_class": model.model_class,
        "classification": model.classification,
    }

    with open(f"app/configs/{model.name}.json", "w") as file:
        json.dump(meta_dict, file)


def model_fit(
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel], db=False, session=None
):
    if model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model of this class with such name already exists",
        )

    fit_and_save(model, db, session)

    return "Model created successfully"


def delete_model(model: schemas.BaseModel):
    if not model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There are no model of this class with such name! Please, check existing models.",
        )

    os.remove(f"app/configs/{model.name}.json")
    os.remove(f"app/models/{model.name}.dill")

    return "Model deleted successfully"


def refit_model(
    model: Union[schemas.RidgeModel,
                 schemas.LassoModel, schemas.DecisionTreeModel]
):
    if not model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There are no model of this class with such name! Please, check existing models.",
        )

    fit_and_save(model)

    return "Model refitted successfully!"


def get_info():
    return [
        {"model_class": "RidgeModel", "target": [
            "classification", "regression"]},
        {"model_class": "LassoModel", "target": ["regression"]},
        {
            "model_class": "DecisionTreeModel",
            "target": ["classification", "regression"],
        },
    ]


def get_predictions(model: schemas.ModelBase):
    if not model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There are no model of this class with such name! Please, check existing models.",
        )

    with open(f"app/models/{model.name}.dill", "rb") as file:
        estimator = dill.load(file)
    return list(estimator.predict(get_dataset(model.classification)[0]))
