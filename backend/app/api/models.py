from uuid import uuid1
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Boolean, Integer, Column, ForeignKey, String, Table, Date, Float
from app.api.database import Base
from app.core.config import SCHEMA_NAME


class WineDataset(Base):
    __tablename__ = "wine_dataset"
    __table_args__ = {"schema": SCHEMA_NAME}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid1)
    alcohol = Column(Float)
    malic_acid = Column(Float)
    ash	 = Column(Float)
    alcalinity_of_ash = Column(Float)
    magnesium = Column(Float)
    total_phenols = Column(Float)
    flavanoids = Column(Float)
    nonflavanoid_phenols = Column(Float)
    proanthocyanins = Column(Float)
    color_intensity = Column(Float)
    hue = Column(Float)
    od280_od315_of_diluted_wines = Column(Float)
    proline = Column(Float)
    target = Column(Integer, nullable=True)


class WinePredictions(WineDataset):
    __tablename__ = "wine_predictions"
    __table_args__ = {"schema": SCHEMA_NAME}


class DiabetesDataset(Base):
    __tablename__ = "diabetes_dataset"
    __table_args__ = {"schema": SCHEMA_NAME}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid1)
    age = Column(Float)
    sex =  Column(Float)
    bmi = Column(Float)
    bp = Column(Float)
    s1 = Column(Float)
    s2 = Column(Float)
    s3 = Column(Float)
    s4 = Column(Float)
    s5 = Column(Float)
    s6 = Column(Float)
    target = Column(Float, nullable=True)


class DiabetesPredictions(DiabetesDataset):
    __tablename__ = "diabetes_predictions"
    __table_args__ = {"schema": SCHEMA_NAME}

