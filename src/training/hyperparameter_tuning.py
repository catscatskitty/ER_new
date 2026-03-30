import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pathlib import Path
import json


class HyperparameterTuner:
    """Подбор гиперпараметров для традиционных моделей"""
    
    def __init__(self, data_root='data/processed/acoustic'):
        self.data_root = Path(data_root)
        self.X_train = np.load(self.data_root / 'features_train.npy')
        self.y_train = np.load(self.data_root / 'labels_train.npy')
    
    def tune_logistic(self):
        """Подбор параметров для Logistic Regression"""
        from sklearn.linear_model import LogisticRegression
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [500, 1000]
        }
        
        grid = GridSearchCV(
            LogisticRegression(), param_grid, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        grid.fit(self.X_train, self.y_train)
        
        return grid.best_params_, grid.best_score_
    
    def tune_random_forest(self):
        """Подбор параметров для Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid = RandomizedSearchCV(
            RandomForestClassifier(), param_grid,
            n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
        )
        grid.fit(self.X_train, self.y_train)
        
        return grid.best_params_, grid.best_score_
    
    def tune_xgboost(self):
        """Подбор параметров для XGBoost"""
        import xgboost as xgb
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        grid = RandomizedSearchCV(
            xgb.XGBClassifier(eval_metric='logloss'), param_grid,
            n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
        )
        grid.fit(self.X_train, self.y_train)
        
        return grid.best_params_, grid.best_score_
    
    def run_all(self, save_path='results/hyperparameters.json'):
        """Запуск подбора для всех моделей"""
        results = {}
        
        print("Tuning Logistic Regression...")
        results['logistic'] = {'params': self.tune_logistic()[0], 'score': self.tune_logistic()[1]}
        
        print("Tuning Random Forest...")
        results['random_forest'] = {'params': self.tune_random_forest()[0], 'score': self.tune_random_forest()[1]}
        
        print("Tuning XGBoost...")
        results['xgboost'] = {'params': self.tune_xgboost()[0], 'score': self.tune_xgboost()[1]}
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {save_path}")
        return results


if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.run_all()