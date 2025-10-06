import pandas as pd


def predict_from_df(df: pd.DataFrame, model, label_encoder, selected_features):
    '''
    Делает предсказание для DataFrame с правильными колонками.
    Возвращает кортеж с лейблами, вероятностями, уверенностями.
    '''
    df = df[selected_features]
    preds = model.predict(df)
    probs = model.predict_proba(df)
    labels = label_encoder.inverse_transform(preds)
    confidences = probs.max(axis=1)
    return labels, probs, confidences


def predict_single(features: dict, model, label_encoder, selected_features):
    '''
    Делает предсказание для одного объекта (словарь признаков).
    Возвращает лейбл, вероятность, уверенность.
    '''
    df = pd.DataFrame([features])
    df = df[selected_features]
    probs = model.predict_proba(df)[0]
    pred_index = probs.argmax()
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = probs[pred_index]
    return pred_label, probs, confidence