import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pickle

class MLModelSuite:
    
    def __init__(self, model_dir="data/models/ml"):
        self.model_dir = model_dir
        self.models = {} 

    def load_model(self, name, feature_type="mfcc158"):
        cache_key = f"{name}_{feature_type}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        if name == "xgboost":
            path = os.path.join(self.model_dir, f"{name}_{feature_type}.json")
            if os.path.exists(path):
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(path)
                self.models[cache_key] = model
                return model
        
        path = os.path.join(self.model_dir, f"{name}_{feature_type}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self.models[cache_key] = model
            return model
        
        return None

    def train_xgboost(self, X_train, y_train, X_val, y_val, feature_type="mfcc158", robot_weight=1.0):
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report
        
        print(f"\n{'='*60}")
        print(f"Training XGBoost on {feature_type} features...")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Robot class weight: {robot_weight:.2f}")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc', 'error'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': robot_weight,
            'seed': 42,
            'nthread': 12,
            'tree_method': 'hist',
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        print(f"\nStarting training...")
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        y_pred_train = (bst.predict(dtrain) > 0.5).astype(int)
        y_pred_val = (bst.predict(dval) > 0.5).astype(int)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        print(f"\n{'='*60}")
        print(f"XGBoost Training Results:")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"\nClassification Report (Validation):")
        print(classification_report(y_val, y_pred_val, target_names=['Human', 'Robot']))
        
        importance = bst.get_score(importance_type='gain')
        if importance:
            print(f"\nTop 10 Most Important Features (by gain):")
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_imp:
                print(f"  f{feat}: {imp:.2f}")
        
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"xgboost_{feature_type}.json")
        bst.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        print(f"{'='*60}\n")
        
        return bst

    def train_all(self, X_train, y_train, X_val, y_val, feature_type="mfcc158", robot_weight=1.0):
        return self.train_xgboost(X_train, y_train, X_val, y_val, feature_type, robot_weight)

    def predict(self, name, X, feature_type="mfcc158"):
        model = self.load_model(name, feature_type)
        if model is None:
            print(f"Error: Model {name}_{feature_type} not found! Trying mfcc158...")
            model = self.load_model(name, "mfcc158")
            if model is None:
                return None, None
        
        try:
            if name == "xgboost":
                import xgboost as xgb
                dmat = xgb.DMatrix(X)
                
                if hasattr(model, "get_booster"):
                    probs = model.get_booster().predict(dmat)
                else:
                    probs = model.predict(dmat)
                
                if isinstance(probs, np.ndarray) and len(probs.shape) > 0:
                    p1 = float(probs[0])
                else:
                    p1 = float(probs)
                
                pred = 1 if p1 > 0.5 else 0
                conf = [1.0 - p1, p1]
                return pred, conf
            else:
                if hasattr(model, "predict_proba"):
                    conf = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                    return pred, conf
                else:
                    pred = model.predict(X)[0]
                    return pred, None
        except Exception as e:
            print(f"Prediction logic error: {e}")
            return None, None

    def get_feature_importance(self, name, feature_type="mfcc158"):
        model = self.load_model(name, feature_type)
        if model and name == "xgboost":
            if hasattr(model, "get_score"):
                return model.get_score(importance_type='gain')
        return {}

    def cleanup_old_models(self):
        import glob
        
        patterns_to_delete = [
            "xgboost_mfcc8282.json",
            "xgboost_mfcc10282.json", 
            "xgboost_mfcc102110.json",
            "xgboost_all182.json",
            "catboost_*.pkl",
            "rf_*.pkl",
            "logistic_*.pkl"
        ]
        
        deleted = []
        for pattern in patterns_to_delete:
            files = glob.glob(os.path.join(self.model_dir, pattern))
            for f in files:
                try:
                    os.remove(f)
                    deleted.append(os.path.basename(f))
                except Exception as e:
                    print(f"Warning: Could not delete {f}: {e}")
        
        if deleted:
            print(f"Cleaned up {len(deleted)} old model files: {', '.join(deleted)}")
        else:
            print("No old models to clean up.")