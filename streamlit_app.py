import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import base64

def set_custom_theme():
    st.markdown(
        """
        <style>
        /* Make default text white */
        html, body, [class*="st-"] {
            color: white !important;
        }

        /* Dropdown inputs + options: black text */
        div[data-baseweb="select"] * {
            color: black !important;
            background-color: white !important;
        }
        div[data-baseweb="menu"] * {
            color: black !important;
            background-color: white !important;
        }

        /* Number input fields */
        input[type="number"] {
            color: black !important;
            background-color: white !important;
        }

        /* Streamlit buttons (including Predict) */
        .stButton>button {
            background-color: red !important;
            color: black !important;
            border: 1px solid #000 !important;
        }

        /* Fix: Top taskbar (Deploy/Rerun icons and labels) */
        header [data-testid="stToolbar"] * {
            color: black !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )





set_custom_theme()

def set_bg_local(file_path):
    with open(file_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_str}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Usage
set_bg_local("bkg.jpg")

st.title("Cybersecurity Loss Prediction")

model_choice = st.radio("Choose model:", ["Neural Network", "Ridge Regression"])

@st.cache_data
def load_data_and_train():
    df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")
    df['id'] = range(1, len(df) + 1)
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.set_primary_key('id')

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=3000)
    synthetic_data = synthetic_data[synthetic_data["Financial Loss (in Million $)"] >= 0.5]

    df.drop(columns='id', inplace=True, errors='ignore')
    synthetic_data.drop(columns='id', inplace=True, errors='ignore')
    df_aug = pd.concat([df, synthetic_data], ignore_index=True)

    categorical_cols = ['Country', 'Attack Type', 'Target Industry', 'Attack Source',
                        'Security Vulnerability Type', 'Defense Mechanism Used']
    
    df_encoded = pd.get_dummies(df_aug, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=['Financial Loss (in Million $)'])
    y = df_encoded['Financial Loss (in Million $)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    return df, X_train_scaled, X_test_scaled, y_train, y_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler, X.columns, categorical_cols, df_aug

df_raw, X_train, X_test, y_train, y_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler, feature_names, categorical_cols, df_aug = load_data_and_train()

# Train models
if model_choice == "Neural Network":
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0002), loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train_scaled, validation_split=0.2, epochs=30,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                         ReduceLROnPlateau(patience=5)], verbose=0)

    y_pred_scaled = model.predict(X_test)
else:
    ridge = Ridge(alpha=1)
    ridge.fit(X_train, y_train_scaled)
    y_pred_scaled = ridge.predict(X_test).reshape(-1, 1)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_test.values.reshape(-1, 1)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

st.subheader(f"{model_choice} Performance on Test Set")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")


# --- INPUT FORM ---

st.subheader("Try Custom Prediction")

# Create empty DataFrame with all feature columns from training
example = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Fill categorical values using one-hot logic
for col in categorical_cols:
    options = sorted(df_raw[col].dropna().unique())
    choice = st.selectbox(f"{col}", options)
    
    for opt in options:
        one_hot_col = f"{col}_{opt}"
        if one_hot_col in example.columns:
            example[one_hot_col] = 1 if opt == choice else 0

# Fill numeric values with special handling
for col in example.columns:
    if all(cat not in col for cat in categorical_cols):

        if "Year" in col:
            year_options = sorted(df_raw['Year'].dropna().unique().astype(int))
            val = st.selectbox(f"{col}", year_options)
            example[col] = val

        elif "Number of Affected Users" in col:
            val = st.number_input(f"{col}", min_value=0, value=1000, step=100, format="%d")
            example[col] = val

        elif "Incident Resolution Time" in col:
            val = st.number_input(f"{col}", min_value=0, value=10, step=1, format="%d")
            example[col] = val

        else:
            val = st.number_input(f"{col}", value=0.0)
            example[col] = val



# Prediction
if st.button("Predict Loss"):
    example_scaled = X_scaler.transform(example)

    if model_choice == "Neural Network":
        pred_scaled = model.predict(example_scaled)
    else:
        pred_scaled = ridge.predict(example_scaled).reshape(-1, 1)

    pred = y_scaler.inverse_transform(pred_scaled)[0][0]
    # st.success(f"Predicted Financial Loss: ${pred:.2f} Million")

    st.markdown(
        f"""
        <div style='
            background-color: #ff4d4d;
            padding: 16px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            color: white;
            text-align: center;
            border: 2px solid #cc0000;
        '>
        🚨 Predicted Financial Loss: ${pred:.2f} Million
        </div>
        """,
        unsafe_allow_html=True
    )
