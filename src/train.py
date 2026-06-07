from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import joblib

from data import load_data
from preprocess import get_preprocessor

def main():

    # Load data
    df = load_data()

    X = df.drop("MedHouseValue", axis=1)
    y = df["MedHouseValue"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess
    preprocessor = get_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Models
    lin_model = LinearRegression()
    ridge_model = Ridge(alpha=1.0)

    lin_model.fit(X_train_processed, y_train)
    ridge_model.fit(X_train_processed, y_train)

    # Save everything
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(lin_model, "models/linear_model.pkl")
    joblib.dump(ridge_model, "models/ridge_model.pkl")

    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()