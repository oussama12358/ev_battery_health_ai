import sys
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / 'models/saved'
PROCESSED_DIR = ROOT_DIR / 'data/processed'

sys.path.insert(0, str(ROOT_DIR))
from utils.feature_engineering import FEATURE_COLS
from utils.risk_scoring import assess_risk

st.set_page_config(
    page_title='EV Battery Health AI',
    page_icon='🔋',
    layout='wide',
    initial_sidebar_state='expanded',
)


def load_models():
    models = {}
    for name in ['rf_soh', 'rf_rul', 'xgb_soh', 'xgb_rul']:
        path = MODEL_DIR / f'{name}.pkl'
        if path.exists():
            models[name] = joblib.load(path)
    scaler_path = MODEL_DIR / 'scaler.pkl'
    if scaler_path.exists():
        models['scaler'] = joblib.load(scaler_path)
    return models


@st.cache_data(show_spinner=False)
def load_data():
    path = PROCESSED_DIR / 'cycle_features.parquet'
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    path = MODEL_DIR / 'all_metrics.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_predictions(prefix: str):
    path = MODEL_DIR / f'ml_predictions_{prefix}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    return None


def render_sidebar(models_loaded: bool) -> str:
    with st.sidebar:
        st.image('https://img.icons8.com/fluency/96/electric-vehicle.png', width=72)
        st.title('EV Battery Health AI')
        st.caption('Clean, lightweight battery health dashboard')
        st.divider()

        page = st.radio('View', ['Overview', 'Predict'], index=0)

        st.divider()
        if models_loaded:
            st.success('Models loaded')
        else:
            st.error('Models not found. Run: python training/run_all.py')

        st.caption('Built with Streamlit | Simple dashboard')
    return page


def format_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics.items():
        if isinstance(values, dict):
            rows.append({
                'model': model_name,
                'mae': values.get('mae'),
                'rmse': values.get('rmse'),
                'r2': values.get('r2'),
            })
    return pd.DataFrame(rows)


def page_overview(data: pd.DataFrame | None, metrics: dict):
    st.title('Overview')

    if data is None:
        st.warning('No processed dataset loaded. Run the training pipeline first.')
        return

    st.markdown('### Dataset summary')
    batteries = int(data['battery_id'].nunique())
    cycles = int(len(data))
    avg_soh = float(data['soh'].mean())
    below_eol = int((data['soh'] < 0.80).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Batteries', batteries)
    c2.metric('Total cycles', f'{cycles:,}')
    c3.metric('Avg SoH', f'{avg_soh*100:.1f}%')
    c4.metric('Below EOL', f'{below_eol:,}')

    st.markdown('---')
    st.markdown('### Model performance')
    if metrics:
        df_metrics = format_metrics(metrics)
        if not df_metrics.empty:
            st.dataframe(df_metrics.sort_values('model'), use_container_width=True)
            if 'xgb_soh' in metrics and 'rf_soh' in metrics:
                xgb_soh = metrics.get('xgb_soh', {}).get('mae')
                rf_soh = metrics.get('rf_soh', {}).get('mae')
                if xgb_soh is not None and rf_soh is not None:
                    winner = 'XGBoost' if xgb_soh < rf_soh else 'Random Forest'
                    st.write(f'Best SoH model: **{winner}**')
        else:
            st.info('Model metric file found but could not parse the metrics.')
    else:
        st.info('No model metrics available.')

    st.markdown('---')
    st.markdown('### Recent predictions')
    recent = []
    for prefix in ['xgb', 'rf']:
        df_pred = load_predictions(prefix)
        if df_pred is not None:
            df_pred = df_pred.copy()
            df_pred['model'] = prefix.upper()
            recent.append(df_pred)

    if recent:
        all_preds = pd.concat(recent, ignore_index=True)
        st.dataframe(
            all_preds.sort_values('timestamp_utc', ascending=False).head(8),
            use_container_width=True,
        )
    else:
        st.info('No saved prediction logs yet. Use the Predict page.')


def page_predict(models: dict):
    st.title('Predict')
    st.markdown('Enter cycle features to predict SoH, RUL, and battery risk.')

    if not models:
        st.error('No models are loaded. Run the training pipeline first.')
        return

    model_choice = st.selectbox('Model', ['XGBoost', 'Random Forest'])
    prefix = 'xgb' if model_choice == 'XGBoost' else 'rf'
    soh_model = models.get(f'{prefix}_soh')
    rul_model = models.get(f'{prefix}_rul')

    if soh_model is None or rul_model is None:
        st.error(f'The {model_choice} models are not available.')
        return

    with st.expander('Input features', expanded=True):
        cycle = st.number_input('Cycle #', min_value=0, max_value=1000, value=50)
        dis_v_mean = st.slider('Discharge voltage mean (V)', 2.7, 4.2, 3.65, 0.01)
        dis_v_std = st.slider('Discharge voltage std', 0.0, 0.5, 0.12, 0.01)
        dis_v_min = st.slider('Discharge voltage min (V)', 2.5, 3.5, 2.75, 0.01)
        dis_v_max = st.slider('Discharge voltage max (V)', 3.8, 4.2, 4.15, 0.01)
        dis_i_mean = st.slider('Discharge current mean (A)', -3.0, -0.5, -1.5, 0.1)
        dis_t_mean = st.slider('Discharge temp mean (°C)', 15.0, 55.0, 28.0, 0.5)
        dis_t_max = st.slider('Discharge temp max (°C)', 20.0, 70.0, 36.0, 0.5)
        ir_mean = st.slider('Internal resistance mean (Ω)', 0.08, 0.35, 0.12, 0.005)
        dis_duration_s = st.number_input('Discharge duration (s)', min_value=1000, max_value=7200, value=4800)
        dis_charge_throughput_ah = st.number_input('Discharge throughput (Ah)', min_value=0.0, max_value=3.0, value=1.75)
        cumulative_throughput_ah = st.number_input('Cumulative throughput (Ah)', min_value=0.0, max_value=800.0, value=50.0)
        chg_v_mean = st.slider('Charge voltage mean (V)', 3.8, 4.2, 3.90, 0.01)
        chg_v_max = st.slider('Charge voltage max (V)', 4.0, 4.3, 4.15, 0.01)
        chg_i_mean = st.slider('Charge current mean (A)', 0.5, 2.5, 1.5, 0.1)
        chg_t_mean = st.slider('Charge temp mean (°C)', 15.0, 55.0, 25.0, 0.5)
        chg_duration_s = st.number_input('Charge duration (s)', min_value=1000, max_value=8000, value=7200)
        time_to_4v_s = st.number_input('Time to 4.0V (s)', min_value=0, max_value=7200, value=3600)

    defaults = {
        'cycle': float(cycle),
        'dis_v_mean': float(dis_v_mean),
        'dis_v_std': float(dis_v_std),
        'dis_v_min': float(dis_v_min),
        'dis_v_max': float(dis_v_max),
        'dis_i_mean': float(dis_i_mean),
        'dis_t_mean': float(dis_t_mean),
        'dis_t_max': float(dis_t_max),
        'ir_mean': float(ir_mean),
        'dis_duration_s': float(dis_duration_s),
        'dis_charge_throughput_ah': float(dis_charge_throughput_ah),
        'cumulative_throughput_ah': float(cumulative_throughput_ah),
        'chg_v_mean': float(chg_v_mean),
        'chg_v_max': float(chg_v_max),
        'chg_i_mean': float(chg_i_mean),
        'chg_t_mean': float(chg_t_mean),
        'chg_duration_s': float(chg_duration_s),
        'time_to_4v_s': float(time_to_4v_s),
        'v_drop_rate': 0.0,
        'ir_max': float(ir_mean * 1.05),
        'dis_v_end': float(dis_v_min),
        'dis_i_end': float(dis_i_mean * 0.95),
        'chg_v_end': float(chg_v_mean),
        'chg_i_end': float(chg_i_mean * 0.95),
        'delta_dis_v_mean': 0.0,
        'delta_dis_i_mean': 0.0,
        'delta_chg_v_mean': 0.0,
        'delta_chg_i_mean': 0.0,
        'delta_dis_t_mean': 0.0,
        'delta_ir_mean': 0.0,
        'delta_dis_charge_throughput_ah': 0.0,
        'cycle_remaining': float(cycle),
        'cycle_fraction': float(min(cycle / max(cycle, 1), 1.0)),
        'rul_ratio': 0.5,
    }

    feature_row = {name: float(defaults.get(name, 0.0)) for name in FEATURE_COLS}

    if st.button('Run prediction'):
        X = np.array([feature_row[name] for name in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
        if 'scaler' in models:
            X = models['scaler'].transform(X)

        soh_pred = float(np.clip(soh_model.predict(X)[0], 0.0, 1.0))
        rul_pred = int(np.clip(rul_model.predict(X)[0], 0, 999))
        risk = assess_risk(soh_pred, rul_pred)

        st.markdown('### Prediction result')
        st.metric('State of Health', f'{soh_pred*100:.1f}%')
        st.metric('Remaining Useful Life', f'{rul_pred} cycles')
        st.metric('Risk label', risk.risk_label)
        st.write(risk.recommendation)

        record = {'timestamp_utc': datetime.utcnow().isoformat(), 'model': prefix}
        record.update({name: float(feature_row.get(name, 0.0)) for name in FEATURE_COLS})
        record.update({'soh_pred': soh_pred, 'rul_pred': rul_pred, 'risk_label': risk.risk_label, 'risk_score': risk.risk_score})

        path = MODEL_DIR / f'ml_predictions_{prefix}.parquet'
        try:
            if path.exists():
                df_existing = pd.read_parquet(path)
                df_out = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
            else:
                df_out = pd.DataFrame([record])
            df_out.to_parquet(path, index=False)
            st.success(f'Saved prediction to {path.name}')
        except Exception as exc:
            st.warning(f'Could not save prediction log: {exc}')


def main():
    models = load_models()
    data = load_data()
    metrics = load_metrics()

    page = render_sidebar(models_loaded=bool(models))

    if page == 'Overview':
        page_overview(data, metrics)
    else:
        page_predict(models)


if __name__ == '__main__':
    main()
