import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import mlflow
import mlflow.xgboost
import warnings
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec

# Ignore warnings
warnings.filterwarnings('ignore')

# Set MLflow experiment
mlflow.set_experiment("sales_prediction_xgboost")

# Step 1: Start an MLflow run
with mlflow.start_run():

    # Load and preprocess your dataset
    df = pd.read_csv('depi.csv')

    # Drop irrelevant columns
    columns_to_drop = ['Unnamed: 0', 'FactSalesID', 'SragentID', 'SrSalesID', 'timekeyID', 'SrProductID', 'SrAccountID', 'engage_date', 'close_date']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Filter out certain deal stages
    df = df[~df.deal_stage.isin(['Engaging', 'Prospecting'])]

    # Encode the target variable 'deal_stage'
    deal_stage_mapping = {'Lost': 0, 'Won': 1}
    df['deal_stage_encoded'] = df['deal_stage'].map(deal_stage_mapping)

    # One-Hot Encode the categorical columns (SeriesName, SectorName)
    ohe = OneHotEncoder(drop='first')  # drop='first' to avoid multicollinearity

    # Fit and transform the categorical columns
    encoded_columns = ohe.fit_transform(df[['SeriesName', 'SectorName']]).toarray()

    # Convert the encoded columns back to a DataFrame
    encoded_df = pd.DataFrame(encoded_columns, columns=ohe.get_feature_names_out(['SeriesName', 'SectorName']))

    # Concatenate the one-hot encoded columns back to the original dataframe
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    # Drop the original categorical columns after encoding
    df = df.drop(columns=['SeriesName', 'SectorName'], errors='ignore')

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['sales_price', 'revenue']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Split the data into features and labels
    X = df.drop(columns=['deal_stage', 'deal_stage_encoded'])
    y = df['deal_stage_encoded']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log dataset size
    mlflow.log_param("data_size", len(df))

    # Define XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Log XGBoost parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions and log them
    y_pred = model.predict(X_test)

    # Optionally, log the confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_dict(class_report, "classification_report.json")
    mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, "confusion_matrix.json")

    # Define the model signature using TensorSpec dynamically
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, 1), col) for col in X.columns
    ])

    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1), "predicted_deal_stage")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Create an input example for the model
    input_example = X_test[:1].to_numpy().astype(np.float32)  # Ensure the correct type

    # Log the trained XGBoost model with signature and input example
    mlflow.xgboost.log_model(
        model,
        "xgboost_model",
        signature=signature,
        input_example=input_example
    )
