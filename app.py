from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor

# Load the model from disk
loaded_model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)

def generate_prediction_graph(prediction):
    plt.figure(figsize=(8, 6))
    plt.plot(prediction, label='Predicted AQI')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predicted AQI')
    plt.legend()

    # Save the plot to a BytesIO object
    plot_image = BytesIO()
    plt.savefig(plot_image, format='png')
    plot_image.seek(0)

    # Encode the plot image as base64
    plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

    return plot_image_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('real_2016.csv')
    my_prediction = loaded_model.predict(df.iloc[:, :-1].values)
    my_prediction = my_prediction.tolist()

    actual_aqi = [
        284.7958333, 219.7208333, 182.1875, 154.0375, 223.2083333, 200.6458333, 285.225, 236.825, 276.9083333, 108,
        107.625, 125.8916667, 181.0125, 152.5541667, 152.3208333, 319.7375, 332.7083333, 279.6, 179.1166667, 54.79166667,
        93.375, 103, 132.2083333, 127.7083333, 109.3333333, 56.45833333, 135.8333333, 122.5, 69.66666667, 79.83333333,
        62.70833333, 1.916666667, 1.833333333, 1.791666667, 117.5833333, 122.7083333, 76.54166667, 59.20833333,
        77.20833333, 87.83333333, 123.7083333, 64.79166667, 62.875, 52.04166667, 128.3333333, 87.58333333, 23.20833333,
        15.16666667, 21.25, 25.75, 105.3333333, 117.75, 20.79166667, 46.91666667, 104.625, 109.625, 0, 30.875, 75.58333333,
        0, 0, 0, 142.5, 133.375, 102.4166667, 158.2916667, 149.5833333, 107.2916667, 106.5, 128.125, 108, 98.41666667,
        94.375, 118.5, 110.4166667, 95.83333333, 93.5, 38.66666667, 45.375, 65.16666667, 67.08333333, 77.66666667, 70.25,
        75.125, 77, 32.95833333, 40.83333333, 90.70833333, 97.29166667, 79.83333333, 40.375, 49.41666667, 43.66666667,
        0, 29.66666667, 47.75, 43.66666667, 29.83333333, 29.66666667, 37.625, 41.45833333, 53.125, 36, 70.58333333,
        52.45833333, 38.25, 26.58333333, 57.58333333, 41.375, 25.54166667, 28.375, 40.33333333, 35.375, 31.91666667,
        36.79166667, 26.83333333, 21.125, 18, 43.58333333, 46.04166667, 28.75, 64.91666667, 30.66666667, 33.16666667,
        30.625, 30.33333333, 31.75, 0, 0, 83.45833333, 71.66666667, 62.375, 24.875, 37.33333333, 41.5, 48.91666667,
        26.66666667, 29, 60.54166667, 41.58333333, 50.91666667, 82.83333333, 95.25, 115.8333333, 136.7083333, 160.6666667,
        173.5833333, 261.6666667, 272.4166667, 226.2916667, 99.70833333, 217.375, 309.375, 166.9166667, 278, 192.3333333,
        235.3333333, 221.5416667, 210.5416667, 251.2083333, 194.7083333, 250.875, 274.875, 160, 217.9166667, 184.2083333,
        176.375, 185.9583333, 195.375, 365.2916667, 312.2083333, 237.625, 202.875, 260.1666667, 403.9583333, 284.1666667,
        288.4166667, 256.8333333, 169, 186.0416667, 185.5833333
    ]

    # Generate the prediction graph
    prediction_graph = generate_prediction_graph(my_prediction)

    # Create a combined plot for my_prediction and actual_aqi
    plt.figure(figsize=(8, 6))
    plt.plot(my_prediction, label='Predicted AQI')
    plt.plot(actual_aqi, label='Actual AQI')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predicted AQI and Actual AQI')
    plt.legend()

    # Save the combined plot to a BytesIO object
    plot_image_combined = BytesIO()
    plt.savefig(plot_image_combined, format='png')
    plot_image_combined.seek(0)

    # Encode the combined plot image as base64
    plot_image_combined_base64 = base64.b64encode(plot_image_combined.getvalue()).decode('utf-8')

    return render_template('result.html',
                           prediction=my_prediction,
                           actual_aqi=actual_aqi,
                           prediction_graph=prediction_graph,
                           plot_image_combined=plot_image_combined_base64)

if __name__ == '__main__':
    app.run(debug=True)
