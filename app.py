from fastapi import FastAPI, HTTPException, Query
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

model = None
target_names = None


@app.on_event("startup")
def train_model():
    global model, target_names
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    sl: float = Query(...),
    sw: float = Query(...),
    pl: float = Query(...),
    pw: float = Query(...),
):
    try:
        pred = int(model.predict([[sl, sw, pl, pw]])[0])
        return {
            "prediction": pred,
            "class_name": str(target_names[pred]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
