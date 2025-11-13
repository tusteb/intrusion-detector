import streamlit as st
import pandas as pd
import requests
from config import API_URL, EXAMPLE_CSV_PATH, CONFIDENCE_THRESHOLDS

# Настройки страницы
st.set_page_config(page_title='Анализ сетевого трафика', layout='centered')
st.title('🚦 Классификация сетевого трафика')
st.markdown('Загрузите CSV с необходимыми признаками или введите параметры сетевого потока вручную, чтобы получить предсказание модели и уровень её уверенности.')

# Загрузка CSV (с примером данных)
st.subheader('📂 Загрузка потока из файла')
with open(EXAMPLE_CSV_PATH, 'rb') as file:
    st.download_button(label='Скачать шаблон CSV',
                       data=file,
                       file_name='example.csv',
                       mime='text/csv',
                       help='Скачайте пример файла с нужными признаками')

uploaded_file = st.file_uploader('Загрузите CSV-файл', type=['csv'])

if uploaded_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button('**Предсказать по CSV**', key='predict_csv', use_container_width=True)

    if predict_clicked:
        with st.spinner('Модель думает...'):
            files = {"file": (uploaded_file.name,
                              uploaded_file.getvalue(),
                              "text/csv")}
            response = requests.post(f"{API_URL}/predict_csv", files=files)

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(f"❌ Ошибка: {result['error']}")
            else:
                st.toast('Предсказание завершено!')
                st.subheader('Результаты предсказания по CSV')

                for i, row in enumerate(result["results"]):
                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        st.markdown(f'**Строка {i+1}:**')
                        st.write(f'Класс: `{row["prediction"]}`')
                        st.write(f'Уверенность: **{row["confidence"]:.2f}**')

                        if row["confidence"] < CONFIDENCE_THRESHOLDS["low"]:
                            st.warning('⚠️ Низкая уверенность — рекомендуется ручная проверка')
                        elif row["confidence"] < CONFIDENCE_THRESHOLDS["medium"]:
                            st.info('🟠 Средняя уверенность')
                        else:
                            st.success('🟢 Высокая уверенность')

                    with col_right:
                        probs_df = pd.DataFrame({
                            'Класс': list(row["probabilities"].keys()),
                            'Вероятность': list(row["probabilities"].values())
                        })
                        st.bar_chart(probs_df.set_index('Класс'))

                    st.markdown('---')
        else:
            try:
                error_detail = response.json().get("error", "Неизвестная ошибка")
                st.error(f"❌ Ошибка сервера: {response.status_code} — {error_detail}")
            except Exception:
                st.error(f"❌ Ошибка сервера: {response.status_code}")


# Ручной ввод
st.subheader('📝 Ввод вручную')

features = {
    "Destination_Port": st.number_input('Destination Port', min_value=0, max_value=65535, value=443,
                                        help='Порт назначения, используемый в сетевом соединении'),
    "Init_Win_bytes_forward": st.number_input('Init_Win_bytes_forward', min_value=0, max_value=1_000_000, value=8192,
                                              help='Начальный размер окна в байтах в прямом направлении'),
    "Init_Win_bytes_backward": st.number_input('Init_Win_bytes_backward', min_value=0, max_value=1_000_000, value=8192,
                                               help='Начальный размер окна в байтах в обратном направлении'),
    "Bwd_Packets_s": st.number_input('Bwd Packets/s', min_value=0.0, max_value=1_000_000.0, value=1500.0,
                                     help='Скорость пакетов в обратном направлении (пакеты в секунду)'),
    "min_seg_size_forward": st.number_input('min_seg_size_forward', min_value=0, max_value=1500, value=40,
                                            help='Минимальный размер сегмента в прямом направлении'),
    "Fwd_IAT_Std": st.number_input('Fwd IAT Std', min_value=0.0, max_value=1_000_000.0, value=5000.0,
                                   help='Стандартное отклонение интервалов между пакетами в прямом направлении'),
    "Flow_IAT_Min": st.number_input('Flow IAT Min', min_value=0.0, max_value=1_000_000.0, value=1000.0,
                                    help='Минимальный интервал между любыми пакетами в потоке'),
    "Bwd_Packet_Length_Min": st.number_input('Bwd Packet Length Min', min_value=0, max_value=1500, value=60,
                                             help='Минимальная длина пакета в обратном направлении'),
    "Fwd_Packets_s": st.number_input('Fwd Packets/s', min_value=0.0, max_value=1_000_000.0, value=2000.0,
                                     help='Скорость пакетов в прямом направлении (пакеты в секунду)'),
    "Fwd_IAT_Min": st.number_input('Fwd IAT Min', min_value=0.0, max_value=1_000_000.0, value=1000.0,
                                   help='Минимальный интервал между пакетами в прямом направлении')}

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button('**Предсказать по введенным данным**', key='predict_manual', use_container_width=True)

if predict_clicked:
    with st.spinner('Модель думает...'):
        response = requests.post(f"{API_URL}/predict", json=features)

    if response.status_code == 200:
        result = response.json()
        if "error" in result:
            st.error(f"❌ Ошибка: {result['error']}")
        else:
            st.toast('Предсказание завершено!')
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader('Результаты предсказания по введенным данным')
                st.write(f'Класс: **{result["prediction"]}**')
                st.write(f'Уверенность модели: **{result["confidence"]:.2f}**')

                if result["confidence"] < CONFIDENCE_THRESHOLDS["low"]:
                    st.warning('⚠️ Низкая уверенность — рекомендуется ручная проверка')
                elif result["confidence"] < CONFIDENCE_THRESHOLDS["medium"]:
                    st.info('🟠 Средняя уверенность')
                else:
                    st.success('🟢 Высокая уверенность')

            with col_right:
                probs_df = pd.DataFrame({'Класс': list(result["probabilities"].keys()),
                                         'Вероятность': list(result["probabilities"].values())})
                st.bar_chart(probs_df.set_index('Класс'))
    else:
        try:
            error_detail = response.json().get("error", "Неизвестная ошибка")
            st.error(f"❌ Ошибка сервера: {response.status_code} — {error_detail}")
        except Exception:
            st.error(f"❌ Ошибка сервера: {response.status_code}")
