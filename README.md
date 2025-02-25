# GBR Model Predictor 📊

This is a Streamlit web application that utilizes a pre-trained **Gradient Boosting Regressor (GBR)** model to make predictions based on user-inputted features. Simply enter the required values, and the app will provide a predicted output along with a visualization of the input features.

## Features
✅ User-friendly interface with dynamic input fields for each required feature.  
✅ Scalable input data using **StandardScaler** for accurate predictions.  
✅ Real-time predictions with an option to visualize input data using **bar charts**.  
✅ **Error handling** for model loading and prediction processes.  

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- Streamlit
- scikit-learn
- numpy
- pandas
- joblib

## Installation
### 1. Clone the repository:
```sh
git clone https://github.com/your-username/gbr-model-predictor.git
cd gbr-model-predictor
```

### 2. Install the required dependencies:
```sh
pip install -r requirements.txt
```

### 3. Ensure the pre-trained model file (`best_gbr_model.pkl`) is in the root directory.

## Usage
Run the Streamlit app with the following command:
```sh
streamlit run app.py
```

## Live Demo
Check out the live version of the app here: [GBR Model Predictor](https://pickupprediction-sapna-git.streamlit.app/)

## Deployment
You can deploy this Streamlit app using **Streamlit Cloud** or other cloud platforms. Ensure all dependencies are included in `requirements.txt`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Pull requests are welcome! Feel free to open an issue for feature requests or bug reports.

---
### Built with ❤️ using [Streamlit](https://streamlit.io/)

