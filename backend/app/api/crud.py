import pandas as pd
import numpy as np
import json
import dill
import os

from pathlib import Path
from sklearn import linear_model
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_wine, load_diabetes
from fastapi import HTTPException, status

from app.api import schemas

from typing import Union


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


def get_dataset(classification: bool):
    if classification:
        data = load_wine()
        return data.data, data.target
    data = load_diabetes()
    return data.data, data.target


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
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
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
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
):
    X, y = get_dataset(model.classification)

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
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
):
    if model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model of this class with such name already exists",
        )

    fit_and_save(model)

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
    model: Union[schemas.RidgeModel, schemas.LassoModel, schemas.DecisionTreeModel]
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
        {"model_class": "RidgeModel", "target": ["classification", "regression"]},
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

    pathlist = Path("app/models/").glob("**/*.json")
    for path in pathlist:
        with open(path, "r") as file:
            model = dill.load(file)
    return list(model.predict(get_dataset(model.classification)[0]))


    
