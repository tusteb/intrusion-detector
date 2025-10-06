from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictInt, StrictFloat
import joblib
import pandas as pd
import io

bundle = joblib.load("model/xgb_pipeline_bundle.pkl")
model = bundle["model"]
selected_features = bundle["selected_features"]
label_encoder = bundle["label_encoder"]

app = FastAPI()

class InputData(BaseModel):
    Destination_Port: StrictInt
    Init_Win_bytes_forward: StrictInt
    Init_Win_bytes_backward: StrictInt
    Bwd_Packets_s: StrictFloat
    min_seg_size_forward: StrictInt
    Fwd_IAT_Std: StrictFloat
    Flow_IAT_Min: StrictFloat
    Bwd_Packet_Length_Min: StrictInt
    Fwd_Packets_s: StrictFloat
    Fwd_IAT_Min: StrictFloat


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={"error": "Неверное расширение файла. Ожидается .csv"})

    if file.content_type != "text/csv":
        return JSONResponse(
            status_code=400,
            content={"error": f"Неверный MIME-тип: {file.content_type}. Ожидается text/csv"})

    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"error": "Файл пуст или не загружен"})

    try:
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "CSV-файл не содержит данных"})

        missing = [col for col in selected_features if col not in df.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Отсутствуют признаки: {missing}"})

        df = df[selected_features]

        try:
            df = df.astype({col: float for col in df.columns})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Ошибка преобразования типов: {str(e)}"})

        preds = model.predict(df)
        labels = label_encoder.inverse_transform(preds)
        probs = model.predict_proba(df)
        class_names = label_encoder.classes_

        results = []
        for label, prob in zip(labels, probs):
            results.append({"prediction": str(label),
                            "confidence": float(max(prob)),
                            "probabilities": {str(cls): float(p) for cls, p in zip(class_names, prob)}})

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Ошибка сервера: {str(e)}"})


@app.post('/predict')
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.model_dump()])

        df = df.rename(columns={"Destination_Port": "Destination Port",
                                "Bwd_Packets_s": "Bwd Packets/s",
                                "Fwd_IAT_Std": "Fwd IAT Std",
                                "Flow_IAT_Min": "Flow IAT Min",
                                "Bwd_Packet_Length_Min": "Bwd Packet Length Min",
                                "Fwd_Packets_s": "Fwd Packets/s",
                                "Fwd_IAT_Min": "Fwd IAT Min"})

        try:
            df = df.astype({col: float for col in df.columns})
        except Exception as e:
            return {"error": f"Ошибка преобразования типов: {str(e)}"}

        pred = model.predict(df)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]
        probs = model.predict_proba(df)[0]
        class_names = label_encoder.classes_

        return jsonable_encoder({"prediction": str(pred_label),
                                 "confidence": float(max(probs)),
                                 "probabilities": {str(cls): float(p) for cls, p in zip(class_names, probs)}})
    
    except Exception as e:
        return {"error": f"Ошибка сервера: {str(e)}"}
