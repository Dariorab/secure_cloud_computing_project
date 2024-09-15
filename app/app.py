import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@st.cache_data
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc, model

def inference(scaler, model):

    df = yf.download("ETH-USD", start, end)
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaled_data = scaler.fit_transform(dataset)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

def inference_2(scaler, model):

    end = datetime.now()
    start = end - timedelta(days=365)  # or any other suitable time range

    # Download data
    df = yf.download("ETH-USD", start, end)
    window_size = 60

    test_data = df.Close.values.reshape(-1, 1)
    test_data = scaler.fit_transform(test_data)

    # Prepare input sequence for prediction
    input_sequence = test_data[-window_size:].reshape(1, window_size, 1)
    # Get the model's predicted price values
    predictions = model.predict(input_sequence)

    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(predictions)


    return predictions

#st.title('Ethereum Prediction App')
st.markdown("<h1 style='text-align: center;'>Ethereum Prediction App</h1>", unsafe_allow_html=True)
#st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage. This is a sample application and cannot be used as a substitute for real medical advice.')
#image = Image.open('data/diabetes_image.jpg')
#st.image(image, use_column_width=True)
#st.write('Please click the button to get ETH prediction to improve your investments!')

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

dark_style = '''
<style>
.stApp > header {
    background-color: transparent;
}

.stApp {
    margin: auto;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: auto;
    background-color: black; /* Imposta lo sfondo nero */
    color: white; /* Imposta il testo in bianco */
    animation: gradient 15s ease infinite;
    background-size: 400% 400%;
    background-attachment: fixed;
}

</style>
'''

st.markdown(dark_style, unsafe_allow_html=True)

sc, model = load('models/scaler.joblib', 'models/model.joblib')
result = inference_2(sc, model)

end = datetime.now()
start = end - timedelta(days=365)  # or any other suitable time range

# Download data
df = yf.download("ETH-USD", start, end)

dates = pd.date_range(start=end, periods=len(result[0]), freq="D")
dates2 = pd.date_range(start=end-timedelta(days=60), end=end, freq="D")
fig, ax = plt.subplots(figsize=(70, 40), dpi=150)
#ax.rc('axes', edgecolor='white')
plt.style.use('dark_background')
ax.plot(dates, result[0], color='yellow', lw=7)
ax.plot([dates2[-1], dates[0]], [df[-len(dates2)-1:-1].Close.iloc[-1], result[0][0]], color='yellow', lw=7)
ax.plot(dates2, df[-len(dates2)-1:-1].Close, color='blue', lw=7)

ax.set_title('ETH Price Training and Test Sets', fontsize=30)
ax.set_xlabel('Date', fontsize=40)
ax.set_ylabel('Close', fontsize=40)
ax.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 30})
ax.grid(False)
# Aggiorna etichette e titolo con nuovi colori se necessario
ax.set_title('ETH Price Predictions', fontsize=70)
ax.set_xlabel('Date', fontsize=60)
ax.set_ylabel('Close', fontsize=60)

# Aggiorna lo stile della legenda
# ax.legend(['Predictions'], loc='upper left', prop={'size': 60}, frameon=False)
line_pred = Line2D([0], [0], color='yellow', linewidth=4, label='Predictions')
line_real = Line2D([0], [0], color='blue', linewidth=4, label='Real Data')
ax.legend(handles=[line_pred, line_real], loc='upper left', prop={'size': 70}, frameon=False)
ax.tick_params(axis='x', labelsize=50)  # Imposta la dimensione delle etichette sull'asse x
ax.tick_params(axis='y', labelsize=50)  # Imposta la dimensione delle etichette sull'asse y

st.pyplot(fig)


