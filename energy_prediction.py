"""
Energy Consumption Prediction - Full ML Pipeline
=================================================
Predicts hourly energy consumption using historical data,
weather features, time features, and multiple ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

import joblib
import json
import os

# ─────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────

def generate_energy_dataset(n_days=730, seed=42):
    """Simulate 2 years of hourly energy consumption data."""
    np.random.seed(seed)
    
    hours = n_days * 24
    date_range = pd.date_range(start='2022-01-01', periods=hours, freq='h')
    
    hour_of_day   = date_range.hour
    day_of_week   = date_range.dayofweek
    month         = date_range.month
    day_of_year   = date_range.dayofyear
    is_weekend    = (day_of_week >= 5).astype(int)
    
    # Holiday approximation (major US holidays)
    holidays = set()
    for year in [2022, 2023]:
        for md in [(1,1),(7,4),(11,25),(12,25),(5,30),(9,5)]:
            holidays.add(f"{year}-{md[0]:02d}-{md[1]:02d}")
    is_holiday = pd.Series(
        [1 if d.strftime('%Y-%m-%d') in holidays else 0 for d in date_range]
    )
    
    # Weather simulation
    temp = (
        15 + 15*np.sin(2*np.pi*(day_of_year - 80)/365)   # seasonal
        + 5*np.sin(2*np.pi*hour_of_day/24)                 # diurnal
        + np.random.normal(0, 3, hours)                     # noise
    )
    humidity    = np.clip(60 + 20*np.sin(2*np.pi*month/12) + np.random.normal(0,10,hours), 20, 100)
    wind_speed  = np.abs(8 + np.random.normal(0, 4, hours))
    solar_rad   = np.maximum(0, 500*np.sin(np.pi*hour_of_day/24)**2
                             * np.sin(np.pi*day_of_year/365) + np.random.normal(0,50,hours))
    
    # Base consumption pattern
    hour_profile = np.array([
        0.55,0.50,0.47,0.45,0.45,0.48,
        0.55,0.70,0.85,0.92,0.90,0.88,
        0.85,0.85,0.88,0.92,0.95,1.00,
        0.98,0.95,0.90,0.82,0.72,0.62
    ])
    base = hour_profile[hour_of_day] * 800   # kWh base

    # Factors
    hvac       = 150  * np.maximum(0, temp - 22)  # cooling load
    heating    = 200  * np.maximum(0, 15 - temp)  # heating load
    weekend_adj = -80 * is_weekend
    holiday_adj = -100 * is_holiday.values
    solar_save  = -0.1  * solar_rad

    consumption = (base + hvac + heating + weekend_adj
                   + holiday_adj + solar_save
                   + np.random.normal(0, 30, hours))
    consumption = np.maximum(consumption, 100)   # floor
    
    df = pd.DataFrame({
        'datetime':      date_range,
        'energy_kwh':    consumption,
        'temperature':   temp,
        'humidity':      humidity,
        'wind_speed':    wind_speed,
        'solar_radiation': solar_rad,
        'hour':          hour_of_day,
        'day_of_week':   day_of_week,
        'month':         month,
        'day_of_year':   day_of_year,
        'is_weekend':    is_weekend,
        'is_holiday':    is_holiday.values,
        'year':          date_range.year,
        'quarter':       date_range.quarter,
    })
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    
    # Cyclical time encoding
    df['hour_sin']   = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']   = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']    = np.sin(2*np.pi*df['day_of_week']/7)
    df['dow_cos']    = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin']  = np.sin(2*np.pi*df['month']/12)
    df['month_cos']  = np.cos(2*np.pi*df['month']/12)
    df['doy_sin']    = np.sin(2*np.pi*df['day_of_year']/365)
    df['doy_cos']    = np.cos(2*np.pi*df['day_of_year']/365)
    
    # Lag features
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f'energy_lag_{lag}h'] = df['energy_kwh'].shift(lag)
    
    # Rolling statistics
    for window in [6, 24, 168]:
        df[f'energy_roll_mean_{window}h'] = df['energy_kwh'].shift(1).rolling(window).mean()
        df[f'energy_roll_std_{window}h']  = df['energy_kwh'].shift(1).rolling(window).std()
    
    # Weather interactions
    df['temp_humidity']  = df['temperature'] * df['humidity'] / 100
    df['feels_like']     = df['temperature'] - 0.4*(df['temperature'] - 10)*(1 - df['humidity']/100)
    df['heating_degree'] = np.maximum(0, 18 - df['temperature'])
    df['cooling_degree'] = np.maximum(0, df['temperature'] - 22)
    
    df = df.dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'temperature','humidity','wind_speed','solar_radiation',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'month_sin','month_cos','doy_sin','doy_cos',
    'is_weekend','is_holiday','quarter',
    'energy_lag_1h','energy_lag_2h','energy_lag_3h',
    'energy_lag_24h','energy_lag_48h','energy_lag_168h',
    'energy_roll_mean_6h','energy_roll_mean_24h','energy_roll_mean_168h',
    'energy_roll_std_6h','energy_roll_std_24h',
    'temp_humidity','feels_like','heating_degree','cooling_degree',
]

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f"  {name:<30} MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {'model': model, 'name': name, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'mape': mape, 'y_pred': y_pred}


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────

def create_dashboard(df, results, X_test, y_test, best_result, feature_cols):
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0f1117')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)
    
    ACCENT  = '#00d4ff'
    GREEN   = '#00ff88'
    ORANGE  = '#ff8c42'
    PURPLE  = '#b87cff'
    RED     = '#ff4d6d'
    BGCELL  = '#1a1d2e'
    TEXT    = '#e0e6f0'
    GRID    = '#2a2d3e'
    
    def style_ax(ax, title):
        ax.set_facecolor(BGCELL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=ACCENT, fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    
    # ── Plot 1: Time-series overview ──
    ax1 = fig.add_subplot(gs[0, :2])
    sample = df.tail(24*14)
    ax1.fill_between(sample['datetime'], sample['energy_kwh'],
                     alpha=0.15, color=ACCENT)
    ax1.plot(sample['datetime'], sample['energy_kwh'],
             color=ACCENT, linewidth=1, label='Actual')
    style_ax(ax1, '⚡ Energy Consumption — Last 14 Days')
    ax1.set_ylabel('kWh', color=TEXT)
    ax1.tick_params(axis='x', rotation=30)
    
    # ── Plot 2: Prediction vs Actual (best model) ──
    ax2 = fig.add_subplot(gs[0, 2])
    sample_n = min(500, len(y_test))
    idx = range(sample_n)
    ax2.plot(idx, y_test.values[:sample_n], color=ACCENT, linewidth=0.8, label='Actual', alpha=0.85)
    ax2.plot(idx, best_result['y_pred'][:sample_n], color=GREEN, linewidth=0.8,
             label='Predicted', alpha=0.85, linestyle='--')
    ax2.legend(fontsize=7, facecolor=BGCELL, labelcolor=TEXT)
    style_ax(ax2, f"🎯 {best_result['name']}: Predict vs Actual")
    ax2.set_ylabel('kWh', color=TEXT)
    
    # ── Plot 3: Model comparison bar ──
    ax3 = fig.add_subplot(gs[1, 0])
    names = [r['name'].replace(' ', '\n') for r in results]
    maes  = [r['mae'] for r in results]
    colors = [GREEN if r['mae'] == min(maes) else ACCENT for r in results]
    bars = ax3.barh(names, maes, color=colors, edgecolor=GRID, height=0.55)
    for bar, v in zip(bars, maes):
        ax3.text(v + 1, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}', va='center', color=TEXT, fontsize=7)
    style_ax(ax3, '📊 Model MAE Comparison')
    ax3.set_xlabel('MAE (kWh)', color=TEXT)
    ax3.invert_yaxis()
    
    # ── Plot 4: R² comparison ──
    ax4 = fig.add_subplot(gs[1, 1])
    r2s = [r['r2'] for r in results]
    colors4 = [GREEN if r == max(r2s) else PURPLE for r in r2s]
    bars4 = ax4.barh(names, r2s, color=colors4, edgecolor=GRID, height=0.55)
    for bar, v in zip(bars4, r2s):
        ax4.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{v:.4f}', va='center', color=TEXT, fontsize=7)
    style_ax(ax4, '📈 Model R² Comparison')
    ax4.set_xlabel('R² Score', color=TEXT)
    ax4.invert_yaxis()
    
    # ── Plot 5: Residual distribution ──
    ax5 = fig.add_subplot(gs[1, 2])
    residuals = y_test.values - best_result['y_pred']
    ax5.hist(residuals, bins=50, color=PURPLE, edgecolor=BGCELL, alpha=0.85)
    ax5.axvline(0, color=RED, linewidth=1.5, linestyle='--')
    ax5.axvline(np.mean(residuals), color=ORANGE, linewidth=1, linestyle=':',
                label=f'Mean={np.mean(residuals):.1f}')
    ax5.legend(fontsize=7, facecolor=BGCELL, labelcolor=TEXT)
    style_ax(ax5, f'🔍 Residual Distribution — {best_result["name"]}')
    ax5.set_xlabel('Residual (kWh)', color=TEXT)
    ax5.set_ylabel('Count', color=TEXT)
    
    # ── Plot 6: Scatter predicted vs actual ──
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(y_test, best_result['y_pred'], alpha=0.3, s=6, color=ACCENT)
    mn, mx = y_test.min(), y_test.max()
    ax6.plot([mn,mx],[mn,mx], color=RED, linewidth=1.5, linestyle='--', label='Perfect fit')
    ax6.legend(fontsize=7, facecolor=BGCELL, labelcolor=TEXT)
    style_ax(ax6, '🔵 Predicted vs Actual Scatter')
    ax6.set_xlabel('Actual (kWh)', color=TEXT)
    ax6.set_ylabel('Predicted (kWh)', color=TEXT)
    
    # ── Plot 7: Feature importance ──
    ax7 = fig.add_subplot(gs[2, 1])
    if hasattr(best_result['model'], 'feature_importances_'):
        fi = pd.Series(best_result['model'].feature_importances_, index=feature_cols)
        top = fi.nlargest(12)
        ax7.barh(range(len(top)), top.values, color=ORANGE, edgecolor=BGCELL)
        ax7.set_yticks(range(len(top)))
        ax7.set_yticklabels(top.index, fontsize=7, color=TEXT)
        ax7.invert_yaxis()
    style_ax(ax7, '🏆 Top Feature Importances')
    ax7.set_xlabel('Importance', color=TEXT)
    
    # ── Plot 8: Hourly average profile ──
    ax8 = fig.add_subplot(gs[2, 2])
    profile = df.groupby('hour')['energy_kwh'].mean()
    ax8.fill_between(profile.index, profile.values, alpha=0.2, color=GREEN)
    ax8.plot(profile.index, profile.values, color=GREEN, linewidth=2, marker='o', markersize=4)
    style_ax(ax8, '🕐 Average Hourly Profile')
    ax8.set_xlabel('Hour of Day', color=TEXT)
    ax8.set_ylabel('Avg kWh', color=TEXT)
    ax8.set_xticks(range(0, 24, 3))
    
    # Title
    fig.suptitle('⚡ Energy Consumption Prediction — ML Dashboard',
                 fontsize=16, color=TEXT, fontweight='bold', y=0.98)
    
    out = '/mnt/user-data/outputs/energy_ml_dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Dashboard saved → {out}")
    return out


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ENERGY CONSUMPTION PREDICTION — FULL ML PIPELINE")
    print("=" * 60)
    
    # Generate & engineer
    print("\n[1/5] Generating dataset...")
    df_raw = generate_energy_dataset(n_days=730)
    print(f"      Raw shape: {df_raw.shape}  |  Date range: {df_raw['datetime'].min().date()} → {df_raw['datetime'].max().date()}")
    
    print("\n[2/5] Engineering features...")
    df = engineer_features(df_raw)
    print(f"      Engineered shape: {df.shape}  |  Features: {len(FEATURE_COLS)}")
    
    # Split (temporal)
    X = df[FEATURE_COLS]
    y = df['energy_kwh']
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    print(f"\n      Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
    
    # Models
    print("\n[3/5] Training & evaluating models...")
    models = [
        ("Random Forest",         RandomForestRegressor(n_estimators=150, max_depth=16, n_jobs=-1, random_state=42)),
        ("Gradient Boosting",     GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)),
        ("Ridge Regression",      Ridge(alpha=10.0)),
        ("Lasso Regression",      Lasso(alpha=1.0, max_iter=5000)),
        ("ElasticNet",            ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
    ]
    
    results = []
    for name, model in models:
        use_scaled = name in ("Ridge Regression", "Lasso Regression", "ElasticNet")
        Xtr = X_train_s if use_scaled else X_train.values
        Xte = X_test_s  if use_scaled else X_test.values
        res = evaluate_model(name, model, Xtr, y_train, Xte, y_test)
        results.append(res)
    
    best_result = min(results, key=lambda r: r['mae'])
    print(f"\n  ★ Best model: {best_result['name']}  (MAE={best_result['mae']:.2f} kWh, R²={best_result['r2']:.4f})")
    
    # Save model
    print("\n[4/5] Saving best model...")
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    joblib.dump(best_result['model'], '/mnt/user-data/outputs/best_model.pkl')
    joblib.dump(scaler, '/mnt/user-data/outputs/scaler.pkl')
    
    metrics = {r['name']: {'MAE': round(r['mae'],2), 'RMSE': round(r['rmse'],2),
                           'R2': round(r['r2'],4), 'MAPE': round(r['mape'],2)}
               for r in results}
    with open('/mnt/user-data/outputs/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Dashboard
    print("\n[5/5] Generating dashboard...")
    create_dashboard(df, results, X_test, y_test, best_result, FEATURE_COLS)
    
    print("\n" + "=" * 60)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 60)
    return df, results, best_result

if __name__ == '__main__':
    df, results, best_result = main()
