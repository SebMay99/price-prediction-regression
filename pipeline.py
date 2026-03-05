import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import os
import pickle
warnings.filterwarnings("ignore")

# ── Try importing XGBoost (optional) ───────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — install with: pip install xgboost")

OUTPUTS_DIR = "outputs"
MODELS_DIR  = "models"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, target_col: str = "SalePrice") -> None:
    """Generate and save EDA charts."""
    print("\n📊  Running EDA …")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis — House Prices", fontsize=16, fontweight="bold")

    # (a) Target distribution
    axes[0, 0].hist(df[target_col], bins=50, color="#4F81BD", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Target Distribution (SalePrice)")
    axes[0, 0].set_xlabel("Sale Price ($)")
    axes[0, 0].set_ylabel("Frequency")

    # (b) Log-transformed target
    axes[0, 1].hist(np.log1p(df[target_col]), bins=50, color="#C0504D", edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Log-Transformed SalePrice")
    axes[0, 1].set_xlabel("log(1 + SalePrice)")
    axes[0, 1].set_ylabel("Frequency")

    # (c) Missing values heatmap (top 20 columns)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    if not missing.empty:
        axes[1, 0].barh(missing.index, missing.values, color="#9BBB59", edgecolor="white")
        axes[1, 0].set_title("Top 20 Columns with Missing Values")
        axes[1, 0].set_xlabel("Count")
    else:
        axes[1, 0].text(0.5, 0.5, "No missing values 🎉", ha="center", va="center", fontsize=14)
        axes[1, 0].set_title("Missing Values")

    # (d) Correlation with target (top 15 numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False).head(15)
    axes[1, 1].barh(corr.index[::-1], corr.values[::-1], color="#4BACC6", edgecolor="white")
    axes[1, 1].set_title(f"Top 15 Numeric Features — Correlation with {target_col}")
    axes[1, 1].set_xlabel("|Pearson r|")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "eda_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and handle missing data."""
    print("\n⚙️  Feature Engineering …")
    df = df.copy()

    # ── New derived features ──────────────────────────────────────────────────
    if "YearBuilt" in df.columns:
        df["HouseAge"]        = df["YrSold"].fillna(2010) - df["YearBuilt"]
    if "YearRemodAdd" in df.columns:
        df["YearsSinceRemod"] = df["YrSold"].fillna(2010) - df["YearRemodAdd"]
    if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
        df["TotalSF"]         = df["GrLivArea"] + df["TotalBsmtSF"].fillna(0)
    if {"BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"}.issubset(df.columns):
        df["TotalBaths"]      = (df["BsmtFullBath"].fillna(0)
                                 + 0.5 * df["BsmtHalfBath"].fillna(0)
                                 + df["FullBath"].fillna(0)
                                 + 0.5 * df["HalfBath"].fillna(0))
    if {"1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["HasSecondFloor"]  = (df["2ndFlrSF"] > 0).astype(int)
    if {"GarageArea"}.issubset(df.columns):
        df["HasGarage"]       = (df["GarageArea"].fillna(0) > 0).astype(int)
    if {"PoolArea"}.issubset(df.columns):
        df["HasPool"]         = (df["PoolArea"].fillna(0) > 0).astype(int)

    new_feats = [c for c in ["HouseAge","YearsSinceRemod","TotalSF",
                              "TotalBaths","HasSecondFloor","HasGarage","HasPool"]
                 if c in df.columns]
    print(f"   Created {len(new_feats)} new features: {new_feats}")

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("None")
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ── Impute remaining numeric NaNs ─────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer  = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    print(f"   Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELING
# ══════════════════════════════════════════════════════════════════════════════

def train_models(X_train, X_test, y_train, y_test) -> dict:
    """Train multiple models, return results dict."""
    print("\n🤖  Training models …")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                                    min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=4, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, random_state=42,
                                          eval_metric="rmse", verbosity=0)

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results  = {}
    best_rmse, best_name, best_model = np.inf, None, None

    for name, model in models.items():
        Xtr = X_train_s if name in ("Linear Regression", "Ridge Regression") else X_train
        Xte = X_test_s  if name in ("Linear Regression", "Ridge Regression") else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        results[name] = {"model": model, "preds": preds, "rmse": rmse, "mae": mae, "r2": r2}
        flag = ""
        if rmse < best_rmse:
            best_rmse, best_name, best_model = rmse, name, model
            flag = " ⭐"
        print(f"   {name:<22}  RMSE=${rmse:>10,.0f}  MAE=${mae:>9,.0f}  R²={r2:.4f}{flag}")

    # Save best model + scaler
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "name": best_name}, f)
    print(f"\n   Best model: {best_name}  (RMSE=${best_rmse:,.0f})")
    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, best_name: str, X_test, y_test, feature_names) -> None:
    """Generate evaluation plots and save to outputs/."""
    print("\n📈  Generating evaluation plots …")

    # ── (a) Model comparison bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(results.keys())
    rmses  = [results[n]["rmse"] for n in names]
    colors = ["#4F81BD" if n != best_name else "#C0504D" for n in names]
    bars   = ax.bar(names, rmses, color=colors, edgecolor="white", width=0.55)
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in rmses], padding=4, fontsize=9)
    ax.set_title("Model Comparison — RMSE (lower is better)", fontweight="bold")
    ax.set_ylabel("RMSE ($)")
    ax.set_ylim(0, max(rmses) * 1.2)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    best = results[best_name]

    # ── (b) Predicted vs Actual ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, best["preds"], alpha=0.4, color="#4F81BD", edgecolors="none", s=20)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_title(f"{best_name} — Predicted vs Actual", fontweight="bold")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── (c) Residual plot ──────────────────────────────────────────────────────
    residuals = y_test - best["preds"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(best["preds"], residuals, alpha=0.4, color="#9BBB59", edgecolors="none", s=20)
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title(f"{best_name} — Residuals vs Fitted", fontweight="bold")
    axes[0].set_xlabel("Fitted Values ($)")
    axes[0].set_ylabel("Residuals ($)")
    axes[1].hist(residuals, bins=50, color="#4BACC6", edgecolor="white", alpha=0.85)
    axes[1].set_title("Residuals Distribution")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── (d) Feature importance ────────────────────────────────────────────────
    model_obj = best["model"]
    if hasattr(model_obj, "feature_importances_"):
        imp = pd.Series(model_obj.feature_importances_, index=feature_names)
        top = imp.sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top.index[::-1], top.values[::-1], color="#4F81BD", edgecolor="white")
        ax.set_title(f"{best_name} — Top 20 Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Top 5 features: {list(top.index[:5])}")

    print(f"   All plots saved to ./{OUTPUTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(filepath: str, target_col: str = "SalePrice") -> None:
    df_raw = load_data(filepath)
    run_eda(df_raw, target_col)

    df = engineer_features(df_raw)

    X = df.drop(columns=[target_col, "Id"], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n   Train: {X_train.shape}  |  Test: {X_test.shape}")

    results, best_name = train_models(X_train, X_test, y_train, y_test)
    plot_results(results, best_name, X_test, y_test, X.columns.tolist())

    print("\n✅  Pipeline complete!")
    print(f"   Artifacts: ./{OUTPUTS_DIR}/")
    print(f"   Best model: ./{MODELS_DIR}/best_model.pkl")


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    run_pipeline(filepath)