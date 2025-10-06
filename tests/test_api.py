import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app

client = TestClient(app)
# Ручной ввод (корректный запрос)
def test_predict_valid():
    payload = {
        "Destination_Port": 443,
        "Init_Win_bytes_forward": 8192,
        "Init_Win_bytes_backward": 8192,
        "Bwd_Packets_s": 1500.0,
        "min_seg_size_forward": 40,
        "Fwd_IAT_Std": 5000.0,
        "Flow_IAT_Min": 1000.0,
        "Bwd_Packet_Length_Min": 60,
        "Fwd_Packets_s": 2000.0,
        "Fwd_IAT_Min": 1000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

# Ручной ввод (отсутствуют данные)
def test_predict_missing_field():
    payload = {"Destination_Port": 443}  # остальные поля отсутствуют
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

# Ручной ввод (неверный тип данных)
def test_predict_wrong_type():
    payload = {"Destination_Port": "443",  # строка вместо int
               "Init_Win_bytes_forward": 8192,
               "Init_Win_bytes_backward": 8192,
               "Bwd_Packets_s": 1500.0,
               "min_seg_size_forward": 40,
               "Fwd_IAT_Std": 5000.0,
               "Flow_IAT_Min": 1000.0,
               "Bwd_Packet_Length_Min": 60,
               "Fwd_Packets_s": 2000.0,
               "Fwd_IAT_Min": 1000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

# Загрузка CSV (корректный запрос)
def test_predict_csv_valid():
    with open("data/example.csv", "rb") as f:
        response = client.post("/predict_csv", files={"file": ("example.csv", f, "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    for row in data["results"]:
        assert "prediction" in row
        assert "confidence" in row
        assert "probabilities" in row

# Загрузка CSV (пустой файл)
def test_predict_csv_empty():
    response = client.post("/predict_csv", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data

# Загрузка CSV (файл не читается)
def test_predict_csv_wrong_format():
    response = client.post("/predict_csv", files={"file": ("test.txt", b"not,a,csv", "text/plain")})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data

# Загрузка CSV (текст вместо чисел)
def test_predict_csv_with_text_values():
    with open("data/example_text.csv", "rb") as f:
        response = client.post("/predict_csv", files={"file": ("example_text.csv", f, "text/csv")})
    assert response.status_code == 422 or response.status_code == 400
    data = response.json()
    assert "error" in data

# Загрузка CSV (файл без строк)
def test_predict_csv_empty_file():
    with open("data/example_empty.csv", "rb") as f:
        response = client.post("/predict_csv", files={"file": ("example_empty.csv", f, "text/csv")})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data

# Загрузка CSV (отсутствует один признак)
def test_predict_csv_missing_column():
    with open("data/example_no_last_column.csv", "rb") as f:
        response = client.post("/predict_csv", files={"file": ("example_no_last_column.csv", f, "text/csv")})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "признаки" in data["error"] or "columns" in data["error"].lower()

# Загрузка CSV (неправильное расширение файла)
def test_predict_csv_wrong_extension():
    response = client.post("/predict_csv", files={"file": ("test.txt", b"not,a,csv", "text/plain")})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "расширение" in data["error"].lower() or "ожидается" in data["error"].lower()
