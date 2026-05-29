import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Set page config
st.set_page_config(page_title="Movie Rating Predictor", layout="centered")

st.title("🍿 Movie Rating Prediction App")
st.write("Enter movie details to predict if its rating will be 'good' or 'bad'.")

@st.cache_data # Cache data loading and preprocessing to improve performance
def load_and_preprocess_data():
    # Mount Google Drive - NOTE: This will not work directly in deployed Streamlit apps
    # For deployment, the file would need to be in the app's directory or accessible via other means.
    # For Colab, assume file is in MyDrive/intership - ICT academy/Project
    filepath = "results_with_crew.csv"
    try:
        df_raw = pd.read_csv(filepath)
        # print("DataFrame loaded successfully.") # Can't print in Streamlit, use st.write for debug
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {filepath}. Please ensure the path is correct and Drive is mounted.")
        st.stop() # Stop the app if data cannot be loaded
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the CSV: {e}")
        st.stop()

    df = df_raw.copy()

    # Missing value handling (replicated from M5r7keVWGp41)
    df['writers'] = df['writers'].fillna(df['writers'].mode()[0])

    # Outlier removal for numerical columns
    for col in ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'rank']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

    # Min-Max Scaling
    minmax_scaler = MinMaxScaler()
    for col in ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'rank']:
        df[col] = minmax_scaler.fit_transform(df[[col]])

    # Encoding
    label_encoder = LabelEncoder()
    df['primaryTitle'] = label_encoder.fit_transform(df['primaryTitle'])
    df = pd.get_dummies(df, columns=['primaryTitle'], dtype=int)
    df = pd.get_dummies(df, columns=["tconst", "directors", "writers", "genres", "IMDbLink", "Title_IMDb_Link"], dtype=int)

    # Define multi-class rating categories (for completeness, though not used in binary prediction)
    bins_multi = [0, 0.33, 0.66, 1.0]
    labels_multi = ['low', 'medium', 'high']
    df['rating_category'] = pd.cut(df['averageRating'], bins=bins_multi, labels=labels_multi, include_lowest=True)

    # Create binary rating column: 'good' if scaled averageRating >= 0.33, 'bad' otherwise
    df['binary_rating_category'] = df['averageRating'].apply(lambda x: 'good' if x >= 0.33 else 'bad')

    # New target variable for binary classification
    y_binary = df['binary_rating_category']

    # Features remain the same, dropping original averageRating, rating_category, and the new binary_rating_category from X
    X_binary = df.drop(['averageRating', 'rating_category', 'binary_rating_category'], axis=1)

    # Store feature names for later use in prediction
    feature_names = X_binary.columns.tolist()

    return X_binary, y_binary, feature_names, minmax_scaler, label_encoder, df_raw # Return df_raw to get original categorical values

X_binary, y_binary, feature_names, minmax_scaler, label_encoder_primaryTitle, df_original = load_and_preprocess_data()

@st.cache_resource # Cache model training
def train_model(X, y):
    # Split the data into training and testing sets for binary classification
    X_train_binary, _, y_train_binary, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to the training data for binary classification
    smote_binary = SMOTE(random_state=42)
    X_train_binary_resampled, y_train_binary_resampled = smote_binary.fit_resample(X_train_binary, y_train_binary)

    # Initialize and train the Decision Tree Classifier for binary classification
    dt_binary_classifier = DecisionTreeClassifier(random_state=42)
    dt_binary_classifier.fit(X_train_binary_resampled, y_train_binary_resampled)
    return dt_binary_classifier

dt_binary_classifier = train_model(X_binary, y_binary)

# --- Streamlit UI for User Input ---

st.header("Input Movie Details")

# Input fields for numerical features
start_year = st.slider("Start Year", min_value=1900, max_value=2026, value=2000)
runtime_minutes = st.slider("Runtime (minutes)", min_value=25, max_value=375, value=120)
average_rating = st.slider("Average Rating (1.0-10.0)", min_value=1.0, max_value=10.0, value=7.5, step=0.1)
num_votes = st.number_input("Number of Votes", min_value=0, value=100000)
rank = st.number_input("Rank", min_value=1, value=2500)

st.subheader("Categorical Features (Simplified Input)")
st.write("Due to the high dimensionality of one-hot encoded categorical features, direct input for all is impractical.")
st.write("For this demo, we'll allow input for `primaryTitle` and will set other one-hot encoded features to zero.")

# Get unique primary titles from original data for dropdown
unique_primary_titles = df_original['primaryTitle'].unique().tolist()
primary_title_input = st.selectbox("Primary Title (select one if available or type)", [''] + sorted(unique_primary_titles))

# Explanation for handling other categorical features
st.info("In a real application, you would either provide more structured input for key categorical features or aggregate them differently. For now, all other one-hot encoded features will be assumed as 0 unless specified.")

if st.button("Predict Rating"):
    # Prepare input data for prediction
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Apply Min-Max Scaling to numerical inputs using the *same scaler* fitted during training
    # Note: For actual deployment, you'd save/load the fitted scaler. Here, we're relying on the cached `minmax_scaler`.
    # To re-use the fitted scaler correctly, you need to apply it to individual numerical features.

    # Create a temporary DataFrame for just numerical inputs to transform them
    numerical_input_data = pd.DataFrame({
        'startYear': [start_year],
        'runtimeMinutes': [runtime_minutes],
        'averageRating': [average_rating],
        'numVotes': [num_votes],
        'rank': [rank]
    })

    # Create a new scaler for each numerical feature as the original code applied separate fit_transform
    # This is a simplification; ideally, a single scaler would be fit for all numerical columns together.
    scaled_numerical_inputs = numerical_input_data.copy()
    for col in numerical_input_data.columns:
        temp_scaler = MinMaxScaler()
        # Fit on a range that makes sense for demonstration or pre-calculate min/max from training data
        # For a robust solution, the actual min/max from the training set should be used.
        # Here, we're fitting on a single point, which will scale it to 0 or 1 depending on the values. This is not ideal.
        # A proper solution involves saving the fitted minmax_scaler from `load_and_preprocess_data` for each feature.
        # For simplicity of this demo, we'll re-initialize and fit for each feature to get a scaled value.
        # In `load_and_preprocess_data`, `minmax_scaler` was reset for each column, simulating individual scalers.
        # So, we'll re-create that behavior here. This is NOT how you'd do it in production.
        if col == 'startYear':
            col_min, col_max = df_original['startYear'].min(), df_original['startYear'].max()
        elif col == 'runtimeMinutes':
            col_min, col_max = df_original['runtimeMinutes'].min(), df_original['runtimeMinutes'].max()
        elif col == 'averageRating': # Original averageRating was 1-10, scaled to 0-1
            col_min, col_max = df_original['averageRating'].min(), df_original['averageRating'].max()
        elif col == 'numVotes':
            col_min, col_max = df_original['numVotes'].min(), df_original['numVotes'].max()
        elif col == 'rank':
            col_min, col_max = df_original['rank'].min(), df_original['rank'].max()

        # Create a dummy array for scaler fitting to get correct min/max bounds
        dummy_data_for_scaler = np.array([col_min, col_max]).reshape(-1, 1)
        temp_scaler.fit(dummy_data_for_scaler)
        scaled_numerical_inputs[col] = temp_scaler.transform(numerical_input_data[[col]])

    input_df['startYear'] = scaled_numerical_inputs['startYear']
    input_df['runtimeMinutes'] = scaled_numerical_inputs['runtimeMinutes']
    input_df['numVotes'] = scaled_numerical_inputs['numVotes']
    input_df['rank'] = scaled_numerical_inputs['rank']

    # Handle primaryTitle
    if primary_title_input:
        # The label_encoder_primaryTitle was fitted on the *numerical* primaryTitle before one-hot encoding
        # So, we need to convert the original primaryTitle string to its encoded integer value first
        try:
            # Get the unique numerical mapping for primaryTitle from the preprocessed df
            # This is a hack, a proper solution would involve saving the LabelEncoder object
            original_title_df = df_original[df_original['primaryTitle'] == primary_title_input]
            if not original_title_df.empty:
                original_encoded_value = label_encoder_primaryTitle.transform([primary_title_input])[0]
                col_name = f'primaryTitle_{original_encoded_value}'
                if col_name in input_df.columns:
                    input_df[col_name] = 1
            else:
                st.warning(f"'{primary_title_input}' not found in training data's primary titles. It will be treated as an unknown title (all zeros).")
        except ValueError:
            st.warning(f"'{primary_title_input}' could not be encoded. It will be treated as an unknown title (all zeros).")

    # Predict
    prediction = dt_binary_classifier.predict(input_df)
    prediction_proba = dt_binary_classifier.predict_proba(input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 'good':
        st.success(f"The predicted rating category is: **{prediction[0].upper()}**")
    else:
        st.error(f"The predicted rating category is: **{prediction[0].upper()}**")

    st.write(f"Confidence (bad vs. good): {prediction_proba[0][0]:.2f} vs. {prediction_proba[0][1]:.2f}")

st.write("--- Drag the sidebar to view more details ---")
