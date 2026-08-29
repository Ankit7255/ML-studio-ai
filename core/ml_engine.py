import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(df: pd.DataFrame, target_col: str, task_type: str):
    # Isolate features and target
    features = [col for col in df.columns if col != target_col]
    
    # Dynamic encoding for categorical variables
    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type == 'Classification':
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = accuracy_score(y_test, preds)
    else:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = mean_squared_error(y_test, preds)
        
    return model, metric, X.columns.tolist()