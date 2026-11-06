import hashlib
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ----------------- DATABASE CONFIGURATION -----------------
DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FeatureFlagDB(Base):
    __tablename__ = "flags"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, default="")
    is_enabled = Column(Boolean, default=True)
    rollout_percentage = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)

class ExperimentDB(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    variants = Column(String, default="control,variant_green")
    status = Column(String, default="running")
    created_at = Column(DateTime, default=datetime.utcnow)

class EventDB(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    experiment_key = Column(String, index=True, nullable=False)
    user_id = Column(String, nullable=False)
    variant = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # "exposure" or "conversion"
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- PYDANTIC SCHEMAS -----------------
class FlagCreate(BaseModel):
    key: str
    description: Optional[str] = ""
    is_enabled: bool = True
    rollout_percentage: int = Field(default=100, ge=0, le=100)

class FlagUpdate(BaseModel):
    description: Optional[str] = None
    is_enabled: Optional[bool] = None
    rollout_percentage: Optional[int] = Field(default=None, ge=0, le=100)

class ExperimentCreate(BaseModel):
    key: str
    name: str
    variants: List[str] = ["control", "variant_green"]

class ExperimentUpdate(BaseModel):
    status: Optional[str] = None

class EventCreate(BaseModel):
    experiment_key: str
    user_id: str
    event_type: str = "conversion"

# ----------------- FASTAPI CORE SETUP -----------------
app = FastAPI(title="Feature Flagging & A/B Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_bucket(key: str, user_id: str, modulo: int = 100) -> int:
    """Deterministic hashing using SHA-256 for consistent user bucketing."""
    h = hashlib.sha256(f"{key}:{user_id}".encode()).hexdigest()
    return int(h[:8], 16) % modulo

# ----------------- FEATURE FLAG API -----------------
@app.get("/api/flags")
def list_flags(db: Session = Depends(get_db)):
    return db.query(FeatureFlagDB).order_by(FeatureFlagDB.created_at.desc()).all()

@app.post("/api/flags", status_code=status.HTTP_201_CREATED)
def create_flag(payload: FlagCreate, db: Session = Depends(get_db)):
    clean_key = payload.key.strip().lower()
    if db.query(FeatureFlagDB).filter_by(key=clean_key).first():
        raise HTTPException(status_code=400, detail="Flag key already exists")
    flag = FeatureFlagDB(
        key=clean_key,
        description=payload.description or "",
        is_enabled=payload.is_enabled,
        rollout_percentage=payload.rollout_percentage
    )
    db.add(flag)
    db.commit()
    db.refresh(flag)
    return flag

@app.put("/api/flags/{key}")
def update_flag(key: str, payload: FlagUpdate, db: Session = Depends(get_db)):
    flag = db.query(FeatureFlagDB).filter_by(key=key).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")
    if payload.description is not None:
        flag.description = payload.description
    if payload.is_enabled is not None:
        flag.is_enabled = payload.is_enabled
    if payload.rollout_percentage is not None:
        flag.rollout_percentage = payload.rollout_percentage
    db.commit()
    db.refresh(flag)
    return flag

@app.delete("/api/flags/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_flag(key: str, db: Session = Depends(get_db)):
    flag = db.query(FeatureFlagDB).filter_by(key=key).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")
    db.delete(flag)
    db.commit()
    return None

@app.get("/api/eval/{key}")
def evaluate_flag(key: str, user_id: str = "user_default", db: Session = Depends(get_db)):
    flag = db.query(FeatureFlagDB).filter_by(key=key).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")

    if not flag.is_enabled:
        return {
            "flag_key": key,
            "user_id": user_id,
            "enabled": False,
            "bucket": 0,
            "reason": "Flag globally disabled (Kill-Switch Active)"
        }

    score = get_bucket(key, user_id, 100)
    decision = score < flag.rollout_percentage
    reason = (
        f"Bucket score {score} < {flag.rollout_percentage}% rollout"
        if decision
        else f"Bucket score {score} >= {flag.rollout_percentage}% rollout"
    )

    return {
        "flag_key": key,
        "user_id": user_id,
        "enabled": decision,
        "bucket": score,
        "reason": reason
    }

# ----------------- A/B TESTING & EXPERIMENTS API -----------------
@app.get("/api/experiments")
def list_experiments(db: Session = Depends(get_db)):
    exps = db.query(ExperimentDB).order_by(ExperimentDB.created_at.desc()).all()
    results = []
    for exp in exps:
        variants = [v.strip() for v in exp.variants.split(",") if v.strip()]
        analytics = {}
        for var in variants:
            exposures = db.query(EventDB).filter_by(experiment_key=exp.key, variant=var, event_type="exposure").count()
            conversions = db.query(EventDB).filter_by(experiment_key=exp.key, variant=var, event_type="conversion").count()
            rate = round((conversions / exposures * 100), 2) if exposures > 0 else 0.0
            analytics[var] = {"exposures": exposures, "conversions": conversions, "rate": rate}

        results.append({
            "id": exp.id,
            "key": exp.key,
            "name": exp.name,
            "variants": variants,
            "status": exp.status,
            "analytics": analytics
        })
    return results

@app.post("/api/experiments", status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)):
    clean_key = payload.key.strip().lower()
    if db.query(ExperimentDB).filter_by(key=clean_key).first():
        raise HTTPException(status_code=400, detail="Experiment key already exists")
    cleaned_variants = [v.strip() for v in payload.variants if v.strip()]
    if len(cleaned_variants) < 2:
        raise HTTPException(status_code=400, detail="Experiment requires at least 2 variants")

    exp = ExperimentDB(
        key=clean_key,
        name=payload.name,
        variants=",".join(cleaned_variants),
        status="running"
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp

@app.put("/api/experiments/{key}")
def update_experiment(key: str, payload: ExperimentUpdate, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter_by(key=key).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if payload.status:
        exp.status = payload.status
    db.commit()
    return exp

@app.delete("/api/experiments/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(key: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter_by(key=key).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.query(EventDB).filter_by(experiment_key=key).delete()
    db.delete(exp)
    db.commit()
    return None

# ----------------- TELEMETRY & DECISIONS -----------------
@app.get("/api/decision/{exp_key}")
def get_experiment_variant(exp_key: str, user_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter_by(key=exp_key).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    variants = [v.strip() for v in exp.variants.split(",") if v.strip()]
    if exp.status != "running":
        return {"experiment": exp_key, "variant": variants[0], "status": exp.status, "reason": "Experiment paused/ended"}

    bucket = get_bucket(exp_key, user_id, len(variants))
    assigned_variant = variants[bucket]

    # Deduplicate exposure logging: only record exposure once per user
    existing_exposure = db.query(EventDB).filter_by(
        experiment_key=exp_key,
        user_id=user_id,
        event_type="exposure"
    ).first()

    if not existing_exposure:
        db.add(EventDB(experiment_key=exp_key, user_id=user_id, variant=assigned_variant, event_type="exposure"))
        db.commit()

    return {"experiment": exp_key, "user_id": user_id, "variant": assigned_variant, "status": exp.status}

@app.post("/api/events")
def track_conversion(payload: EventCreate, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter_by(key=payload.experiment_key).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    variants = [v.strip() for v in exp.variants.split(",") if v.strip()]
    bucket = get_bucket(payload.experiment_key, payload.user_id, len(variants))
    assigned_variant = variants[bucket]

    # Deduplicate conversion: a user can only convert once per experiment
    existing_conversion = db.query(EventDB).filter_by(
        experiment_key=payload.experiment_key,
        user_id=payload.user_id,
        event_type="conversion"
    ).first()

    if existing_conversion:
        return {"message": "Conversion already recorded for this user", "variant": assigned_variant}

    # Ensure an exposure exists if conversion is called directly
    existing_exposure = db.query(EventDB).filter_by(
        experiment_key=payload.experiment_key,
        user_id=payload.user_id,
        event_type="exposure"
    ).first()

    if not existing_exposure:
        db.add(EventDB(experiment_key=payload.experiment_key, user_id=payload.user_id, variant=assigned_variant, event_type="exposure"))

    db.add(EventDB(
        experiment_key=payload.experiment_key,
        user_id=payload.user_id,
        variant=assigned_variant,
        event_type="conversion"
    ))
    db.commit()
    return {"message": "Conversion recorded", "variant": assigned_variant}

# ----------------- ADMIN RESET ENDPOINT -----------------
@app.post("/api/reset-events")
def reset_all_events(db: Session = Depends(get_db)):
    """Clears all event logs so telemetry restarts at 0% cleanly."""
    db.query(EventDB).delete()
    db.commit()
    return {"message": "All exposure and conversion telemetry reset successfully."}