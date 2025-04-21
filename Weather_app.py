import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import streamlit as st

st.title("🌤️ Weather Pattern Prediction Dashboard")
st.write(
    "โปรเจคนี้ใช้ข้อมูลอุณหภูมิย้อนหลังจาก Open-Meteo API "
    "เพื่อสร้างโมเดลทำนายอุณหภูมิในอนาคต"
)
cities = {
    'Bangkok': (13.75, 100.52),
    'Chiang Mai': (18.79, 98.98),
    'Phuket': (7.88, 98.39)
}
city = st.selectbox('เลือกเมือง', list(cities.keys()))
lat, lon = cities[city]

st.write(f"📍 เมืองที่เลือก: {city} (ละติจูด: {lat}, ลองจิจูด: {lon})")
st.write("ข้อมูลย้อนหลัง (2024-01-01 ถึง 2024-03-31)")
start_date = "2024-01-01"
end_date = "2024-03-31"


url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m"
)
try:
    data = requests.get(url).json()
    df = pd.DataFrame({
        'time': data['hourly']['time'],
        'temperature': data['hourly']['temperature_2m']
    })
except Exception as e:
    st.error("เกิดข้อผิดพลาดในการดึงข้อมูลจาก API")
    st.stop()

df['time'] = pd.to_datetime(df['time'])
df['temperature'] = pd.to_numeric(df['temperature'])

# 4. Data Preprocessing
df['temp_rolling3'] = df['temperature'].rolling(window=3).mean()
df['target'] = df['temperature'].shift(-1)  # ทำนายอุณหภูมิชั่วโมงถัดไป
df = df.dropna()

# 5. Split Data
st.write("### ข้อมูลที่ใช้ในการฝึกโมเดล")
st.dataframe(df[['time', 'temperature', 'temp_rolling3', 'target']].head(10))
st.write("### ข้อมูลที่ใช้ในการทดสอบโมเดล")
st.dataframe(df[['time', 'temperature', 'temp_rolling3', 'target']].tail(10))
X = df[['temperature', 'temp_rolling3']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Evaluate Model
st.write("### ประสิทธิภาพของโมเดล")
st.write("#### ค่าทำนายอุณหภูมิ (Test Set)")
st.dataframe(pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index(drop=True))
st.write("#### ค่าความแม่นยำ")
st.write("Mean Absolute Error (MAE) คือ ค่าความแตกต่างเฉลี่ยระหว่างค่าทำนายและค่าจริง")
mae = mean_absolute_error(y_test, y_pred)
st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} °C")

# 8. Visualize Predictions
st.write("### กราฟเปรียบเทียบอุณหภูมิจริงและค่าทำนาย (ช่วง Train Set)")
df_pred = df.iloc[-len(y_test):].copy()
df_pred['prediction'] = y_pred
df_pred = df_pred.set_index('time')

st.write("### กราฟเปรียบเทียบอุณหภูมิจริงและค่าทำนาย (ช่วง Test Set)")
st.line_chart(df_pred[['temperature', 'prediction']])

# 9. Future Predictions
with st.expander("ดูข้อมูลตัวอย่าง"):
    st.dataframe(df_pred[['temperature', 'prediction']].head(24))
