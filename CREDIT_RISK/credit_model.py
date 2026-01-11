from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class CreditRiskModel:
    def __init__(self, model_type="logistic"):
        self.scaler = StandardScaler()
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=4000)
        else:
            self.model = RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                random_state=42
            )

    def split_data(self, df):
        X = df.drop("default", axis=1)
        y = df["default"]
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_pd(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
