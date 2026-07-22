import os
import sys
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import multiprocessing

from src.features import extract_features
from src.models_ml import MLModelSuite
from src.utils import ScalerManager

NUM_WORKERS = min(multiprocessing.cpu_count(), 12)

def load_processed_files(processed_dir):
    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    test_files = []
    test_labels = []
    
    for subset, files, labels in [("train", train_files, train_labels), 
                                   ("val", val_files, val_labels), 
                                   ("test", test_files, test_labels)]:
        subset_dir = os.path.join(processed_dir, subset)
        if not os.path.exists(subset_dir):
            continue
        
        wav_files = glob.glob(os.path.join(subset_dir, "*.wav"))
        for f in wav_files:
            basename = os.path.basename(f)
            try:
                label = int(basename.split("_")[0])
                files.append(f)
                labels.append(label)
            except ValueError:
                print(f"Warning: Cannot parse label from {f}, skipping")
                continue
    
    print(f"Loaded: Train={len(train_files)} (human={train_labels.count(0)}, robot={train_labels.count(1)}), "
          f"Val={len(val_files)} (human={val_labels.count(0)}, robot={val_labels.count(1)}), "
          f"Test={len(test_files)} (human={test_labels.count(0)}, robot={test_labels.count(1)})")
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

def _extract_single_feature(args):
    f, feature_type = args
    try:
        feat = extract_features(f)
        return feat
    except Exception as e:
        print(f"Error extracting {f}: {e}")
        return None

def extract_features_from_files(files, feature_type="mfcc176", n_workers=None):
    if n_workers is None:
        n_workers = NUM_WORKERS
    
    features = []
    
    if n_workers > 1 and len(files) > 10:
        from tqdm import tqdm
        
        print(f"Extracting features with {n_workers} workers...")
        
        args = [(f, feature_type) for f in files]
        
        with multiprocessing.Pool(processes=n_workers) as pool:
            features = list(tqdm(
                pool.imap(_extract_single_feature, args, chunksize=10),
                total=len(args),
                desc="Extracting features"
            ))
        
        features = [f for f in features if f is not None]
    else:
        print(f"Extracting features sequentially...")
        for i, f in enumerate(files):
            try:
                feat = extract_features(f)
                features.append(feat)
            except Exception as e:
                print(f"Error extracting {f}: {e}")
    
    return np.array(features, dtype=np.float32)

def train_model(processed_dir="data/processed", model_dir="data/models/ml", 
                feature_type="mfcc176", robot_weight=None):
    print(f"Loading files from {processed_dir}...")
    train_files, train_labels, val_files, val_labels, test_files, test_labels = load_processed_files(processed_dir)
    
    if not train_files:
        print("No training files found! Please run preprocessing first.")
        return None
    
    print(f"\nExtracting {feature_type} features...")
    X_train = extract_features_from_files(train_files, feature_type, n_workers=NUM_WORKERS)
    y_train = np.array(train_labels)
    
    print(f"Train feature shape: {X_train.shape}, labels distribution: {np.bincount(y_train)}")
    
    if val_files:
        X_val = extract_features_from_files(val_files, feature_type, n_workers=NUM_WORKERS)
        y_val = np.array(val_labels)
    else:
        print("No validation files, splitting train set 80/20...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
    
    print(f"Feature shapes: Train={X_train.shape}, Val={X_val.shape}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({'mean': scaler.mean_, 'std': scaler.scale_}, f)
    print(f"Scaler saved to {scaler_path}")
    
    if robot_weight is None:
        n_human = np.sum(y_train == 0)
        n_robot = np.sum(y_train == 1)
        robot_weight = max(1.0, n_human / n_robot) if n_robot > 0 else 1.0
        print(f"Class imbalance detected: human={n_human}, robot={n_robot}")
        print(f"Using robot_weight={robot_weight:.2f}")
    
    ml_suite = MLModelSuite(model_dir=model_dir)
    model = ml_suite.train_xgboost(
        X_train_scaled, y_train, 
        X_val_scaled, y_val, 
        feature_type=feature_type,
        robot_weight=robot_weight
    )
    
    if test_files:
        print(f"\nEvaluating on test set...")
        X_test = extract_features_from_files(test_files, feature_type, n_workers=NUM_WORKERS)
        y_test = np.array(test_labels)
        X_test_scaled = scaler.transform(X_test)
        
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test_scaled)
        y_pred = (model.predict(dtest) > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"\nTest Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'Robot']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'='*50}")
    
    print(f"\nTop 15 Most Important Features:")
    importance = ml_suite.get_feature_importance("xgboost", feature_type=feature_type)
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        for feat, imp in sorted_imp:
            print(f"  {feat}: {imp:.2f}")
    
    return model

def quick_train(raw_dir="data/raw", processed_dir="data/processed", feature_type="mfcc176"):
    from src.preprocessing import prepare_dataset
    
    print(f"Preparing dataset from {raw_dir}...")
    prepare_dataset(raw_dir, processed_dir)
    
    print(f"\nTraining model...")
    train_model(processed_dir, feature_type=feature_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model for voice classification")
    parser.add_argument("--processed_dir", type=str, default="data/processed", 
                       help="Directory with processed audio files")
    parser.add_argument("--model_dir", type=str, default="data/models/ml", 
                       help="Directory to save trained model")
    parser.add_argument("--feature_type", type=str, default="mfcc176", 
                       help="Feature type to use (mfcc176 includes CQCC+LFCC+contrast)")
    parser.add_argument("--robot_weight", type=float, default=None, 
                       help="Weight for robot class (None for auto)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker processes (None for auto)")
    
    args = parser.parse_args()
    
    if args.workers:
        NUM_WORKERS = args.workers
    
    train_model(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        feature_type=args.feature_type,
        robot_weight=args.robot_weight
    )