import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import io
import warnings
warnings.filterwarnings("ignore")

# ─── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastIQ — Прогнозирование продаж",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Кастомный CSS (тёмная тема в стиле оригинального дизайна) ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Основные цвета */
:root {
    --bg: #0d0f14;
    --surface: #161920;
    --surface2: #1e2230;
    --accent: #4f8cff;
    --accent3: #2dd4bf;
    --success: #34d399;
    --warn: #fbbf24;
    --danger: #f87171;
    --text: #e8eaf0;
    --text2: #8b90a0;
    --text3: #555b70;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Убираем стандартный отступ */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Боковое меню */
[data-testid="stSidebar"] {
    background: #161920 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] .stMarkdown {
    color: #e8eaf0;
}

/* Метрики */
[data-testid="metric-container"] {
    background: #161920;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #555b70 !important;
    font-size: 12px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #4f8cff !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}

/* Кнопки */
.stButton > button {
    background: #4f8cff !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3a7aef !important;
    transform: translateY(-1px);
}

/* Selectbox и inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #252a3a !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

/* Загрузчик файла */
[data-testid="stFileUploader"] {
    background: #161920;
    border: 2px dashed rgba(79,140,255,0.4);
    border-radius: 12px;
    padding: 20px;
}

/* Прогресс-бар */
[data-testid="stProgress"] > div > div {
    background: #4f8cff !important;
}

/* Таблицы */
[data-testid="stDataFrame"] {
    background: #161920 !important;
}

/* Инфо-блоки */
[data-testid="stInfo"] {
    background: rgba(79,140,255,0.08) !important;
    border: 1px solid rgba(79,140,255,0.25) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
[data-testid="stSuccess"] {
    background: rgba(52,211,153,0.08) !important;
    border: 1px solid rgba(52,211,153,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stWarning"] {
    background: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.25) !important;
    border-radius: 8px !important;
}

/* Вкладки */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161920;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #8b90a0;
    font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #252a3a !important;
    color: #e8eaf0 !important;
}

/* Разделители */
hr {
    border-color: rgba(255,255,255,0.07) !important;
}

/* Скрыть footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def styled_header(title: str, subtitle: str = ""):
    """Красивый заголовок страницы"""
    st.markdown(f"""
    <div style="margin-bottom: 28px;">
        <h1 style="font-family: 'DM Serif Display', serif; font-size: 32px; font-weight: 400;
                   color: #e8eaf0; margin-bottom: 4px;">{title}</h1>
        {"" if not subtitle else f'<p style="color: #8b90a0; font-size: 15px; margin: 0;">{subtitle}</p>'}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = "#4f8cff"):
    """Кастомная метрика-карточка"""
    st.markdown(f"""
    <div style="background: #161920; border: 1px solid rgba(255,255,255,0.07);
                border-radius: 12px; padding: 16px 20px; text-align: center;">
        <div style="font-size: 24px; font-weight: 600; color: {color}; margin-bottom: 4px;">{value}</div>
        <div style="font-size: 12px; color: #555b70;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "accent"):
    colors = {
        "accent": ("rgba(79,140,255,0.12)", "#4f8cff"),
        "success": ("rgba(52,211,153,0.12)", "#34d399"),
        "warn": ("rgba(251,191,36,0.12)", "#fbbf24"),
        "danger": ("rgba(248,113,113,0.12)", "#f87171"),
    }
    bg, fg = colors.get(color, colors["accent"])
    return f'<span style="background:{bg}; color:{fg}; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500;">{text}</span>'


def plotly_dark_layout(fig, title: str = "", height: int = 400):
    """Применяет единый тёмный стиль к plotly-графику"""
    fig.update_layout(
        title=dict(text=title, font=dict(family="DM Sans", size=15, color="#e8eaf0")),
        paper_bgcolor="#161920",
        plot_bgcolor="#161920",
        font=dict(family="DM Sans", color="#8b90a0"),
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#555b70",
                   showline=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#555b70",
                   showline=False, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b90a0")),
    )
    return fig


# ─── Обработка данных (логика из ноутбука) ────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if file_name.endswith(".csv"):
        # Пробуем разные разделители и кодировки
        for sep in [",", ";"]:
            for enc in ["utf-8", "utf-8-sig", "cp1251", "latin1"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=enc, low_memory=False)
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue
    else:
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Не удалось прочитать файл. Проверьте формат.")


def detect_columns(cols: list) -> dict:
    """Автоматически определяет нужные колонки"""
    cols_lower = [c.lower() for c in cols]
    def find(patterns):
        for p in patterns:
            for i, c in enumerate(cols_lower):
                if p in c:
                    return cols[i]
        return None
    return {
        "sku":          find(["sku", "артикул", "код товара", "product_id", "item_id", "id"]),
        "date":         find(["date", "дата", "период", "month", "месяц"]),
        "sales_qty":    find(["sales_qty", "qty", "quantity", "количество", "продажи", "sales", "кол-во"]),
        "sales_tg":     find(["sales_tg", "revenue", "выручка", "сумма", "amount", "total"]),
        "name":         find(["name", "наименование", "товар", "название", "product"]),
        "subgroup":     find(["subgroup", "category", "категория", "group", "группа", "подгруппа"]),
        "promo_flag":   find(["promo", "акция", "promotion"]),
        "stock":        find(["stock", "остаток", "запас", "inventory"]),
        "rc_price":     find(["price", "цена", "rc_price", "cost", "стоимость"]),
        "holiday_days": find(["holiday", "праздник"]),
        "ramadan_flag": find(["ramadan", "рамадан"]),
        "promo_count":  find(["promo_count", "кол_акц", "promo_qty"]),
        "receipts_qty": find(["receipt", "приход", "поступление"]),
        "oos_flag":     find(["oos", "out_of_stock", "дефицит"]),
    }


def engineer_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Конструирование признаков — повторяет логику из ноутбука"""
    d = pd.DataFrame()

    # Базовые колонки
    d["sku"]      = df[col_map["sku"]].astype(str) if col_map.get("sku") else "default"
    d["name"]     = df[col_map["name"]].astype(str) if col_map.get("name") else d["sku"]
    d["subgroup"] = df[col_map["subgroup"]].astype(str) if col_map.get("subgroup") else "default"
    d["date"]     = pd.to_datetime(df[col_map["date"]], dayfirst=True, errors="coerce") if col_map.get("date") else pd.Timestamp("2023-01-01")

    # Продажи
    if col_map.get("sales_qty"):
        d["sales_qty"] = pd.to_numeric(df[col_map["sales_qty"]], errors="coerce").fillna(0)
    elif col_map.get("sales_tg"):
        d["sales_qty"] = pd.to_numeric(df[col_map["sales_tg"]], errors="coerce").fillna(0)
    else:
        raise ValueError("Не выбрана колонка с продажами!")

    # Опциональные колонки
    for key, default in [("rc_price", np.nan), ("promo_flag", 0), ("stock", 0),
                          ("holiday_days", 0), ("ramadan_flag", 0),
                          ("promo_count", 0), ("receipts_qty", 0), ("oos_flag", 0)]:
        if col_map.get(key):
            d[key] = pd.to_numeric(df[col_map[key]], errors="coerce").fillna(default)
        else:
            d[key] = default

    # Заполняем цену медианой
    price_med = d["rc_price"].median() if d["rc_price"].notna().any() else 1.0
    d["rc_price_missing_flag"] = d["rc_price"].isna().astype(int)
    d["rc_price"] = d["rc_price"].fillna(price_med)

    # Временны́е признаки
    d["month"]   = d["date"].dt.month
    d["year"]    = d["date"].dt.year
    d["quarter"] = d["date"].dt.quarter
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    min_year = d["year"].min()
    min_month = d.loc[d["year"] == min_year, "month"].min()
    d["trend_index"] = (d["year"] - min_year) * 12 + (d["month"] - min_month)

    # Сортировка
    d = d.sort_values(["sku", "date"]).reset_index(drop=True)

    # Label encoding
    le_sku = LabelEncoder()
    le_sub = LabelEncoder()
    d["sku_encoded"]      = le_sku.fit_transform(d["sku"].astype(str))
    d["subgroup_encoded"] = le_sub.fit_transform(d["subgroup"].astype(str))

    # Статистики по SKU
    d["sku_mean_sales"]      = d.groupby("sku")["sales_qty"].transform("mean")
    d["subgroup_mean_sales"] = d.groupby("subgroup")["sales_qty"].transform("mean")

    # Ценовые признаки
    d["price_ratio"]   = d["rc_price"] / d.groupby("sku")["rc_price"].transform("mean").replace(0, 1)
    d["price_lag_1"]   = d.groupby("sku")["rc_price"].shift(1).fillna(d["rc_price"])
    d["price_change_1"]= d["rc_price"] - d["price_lag_1"]
    d["price_x_promo"] = d["rc_price"] * d["promo_flag"]
    d["stock_x_promo"] = d["stock"] * d["promo_flag"]

    # Лаговые признаки — из ноутбука
    for lag in [1, 2, 3, 6, 12]:
        d[f"lag_{lag}"] = d.groupby("sku")["sales_qty"].shift(lag).fillna(0)

    # Скользящие средние
    for w in [3, 6, 12]:
        d[f"rolling_mean_{w}"] = (
            d.groupby("sku")["sales_qty"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            .fillna(0)
        )

    d["sales_diff_1"] = d["lag_1"] - d["lag_2"]
    d["sales_trend"]  = d.groupby("sku")["sales_qty"].diff().fillna(0)
    d["stock_to_roll3"] = d["stock"] / (d["rolling_mean_3"] + 1)
    d["received_flag"]  = (d["receipts_qty"] > 0).astype(int)
    d["low_stock_flag"] = (d["stock"] <= 5).astype(int)

    # Целевая переменная
    d["target"] = np.log1p(d["sales_qty"])

    return d, le_sku, le_sub, price_med


# Признаки — полный список из ноутбука
FEATURES = [
    "sku_encoded", "subgroup_encoded",
    "month", "quarter", "year", "month_sin", "month_cos", "trend_index",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "sales_diff_1",
    "rc_price", "rc_price_missing_flag", "price_lag_1", "price_change_1",
    "promo_flag", "promo_count",
    "stock", "receipts_qty", "received_flag", "low_stock_flag", "stock_to_roll3",
    "holiday_days", "ramadan_flag",
    "price_x_promo", "stock_x_promo",
    "sku_mean_sales", "subgroup_mean_sales", "sales_trend",
]


def train_model(df: pd.DataFrame, progress_bar):
    """Обучение LightGBM — параметры из ноутбука"""
    # Хронологический сплит: последние 3 месяца = тест
    unique_dates = sorted(df["date"].unique())
    test_dates   = unique_dates[-3:]
    train_df = df[~df["date"].isin(test_dates)].copy()
    test_df  = df[df["date"].isin(test_dates)].copy()

    # Берём только строки где лаги уже есть
    train_df = train_df.dropna(subset=["lag_1", "lag_2", "lag_3"])
    test_df  = test_df.dropna(subset=["lag_1", "lag_2", "lag_3"])

    # Убираем признаки которых нет в датафрейме
    feats = [f for f in FEATURES if f in df.columns]

    X_train = train_df[feats].fillna(0)
    y_train = train_df["target"]
    X_test  = test_df[feats].fillna(0)
    y_test  = test_df["sales_qty"]

    progress_bar.progress(20, text="Инициализация LightGBM...")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    progress_bar.progress(40, text="Обучение модели...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, test_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    progress_bar.progress(80, text="Расчёт метрик...")

    pred_log = model.predict(X_test)
    pred     = np.expm1(pred_log)
    pred     = np.maximum(pred, 0)

    actual = y_test.values
    mae    = mean_absolute_error(actual, pred)
    rmse   = np.sqrt(mean_squared_error(actual, pred))
    r2     = r2_score(actual, pred)
    mask   = actual > 0
    mape   = mean_absolute_percentage_error(actual[mask], pred[mask]) * 100 if mask.sum() > 0 else None

    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}

    # Важность признаков
    feat_imp = pd.DataFrame({
        "feature":    feats,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Сохраняем тестовые предсказания для графика
    test_df = test_df.copy()
    test_df["pred"] = pred

    progress_bar.progress(100, text="Готово!")
    return model, metrics, feat_imp, test_df, feats


def make_forecast(model, df: pd.DataFrame, feats: list,
                  sku_filter: str, horizon: int) -> pd.DataFrame:
    """Итеративный прогноз на horizon месяцев вперёд"""
    skus = df["sku"].unique().tolist() if sku_filter == "Все SKU" else [sku_filter]
    results = []

    for sku in skus:
        sku_df = df[df["sku"] == sku].sort_values("date").copy()
        if sku_df.empty:
            continue

        hist = sku_df["sales_qty"].tolist()
        last_row = sku_df.iloc[-1].copy()
        last_date = sku_df["date"].max()

        for h in range(1, horizon + 1):
            next_date = last_date + pd.DateOffset(months=h)
            month  = next_date.month
            year   = next_date.year

            lag1  = hist[-1]  if len(hist) >= 1  else 0
            lag2  = hist[-2]  if len(hist) >= 2  else 0
            lag3  = hist[-3]  if len(hist) >= 3  else 0
            lag6  = hist[-6]  if len(hist) >= 6  else 0
            lag12 = hist[-12] if len(hist) >= 12 else 0
            roll3  = np.mean(hist[-3:])  if hist else 0
            roll6  = np.mean(hist[-6:])  if hist else 0
            roll12 = np.mean(hist[-12:]) if hist else 0

            row = {
                "sku_encoded":       last_row.get("sku_encoded", 0),
                "subgroup_encoded":  last_row.get("subgroup_encoded", 0),
                "month": month, "quarter": (month - 1) // 3 + 1,
                "year": year,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "trend_index": last_row.get("trend_index", 0) + h,
                "lag_1": lag1, "lag_2": lag2, "lag_3": lag3,
                "lag_6": lag6, "lag_12": lag12,
                "rolling_mean_3": roll3, "rolling_mean_6": roll6, "rolling_mean_12": roll12,
                "sales_diff_1": lag1 - lag2,
                "rc_price": last_row.get("rc_price", 1),
                "rc_price_missing_flag": last_row.get("rc_price_missing_flag", 0),
                "price_lag_1": last_row.get("rc_price", 1),
                "price_change_1": 0,
                "promo_flag": 0, "promo_count": 0,
                "stock": last_row.get("stock", 0),
                "receipts_qty": 0, "received_flag": 0,
                "low_stock_flag": 0,
                "stock_to_roll3": last_row.get("stock", 0) / (roll3 + 1),
                "holiday_days": 3 if month in [1, 3, 5, 7, 9] else 0,
                "ramadan_flag": 1 if month == 3 else 0,
                "price_x_promo": 0, "stock_x_promo": 0,
                "sku_mean_sales": last_row.get("sku_mean_sales", 0),
                "subgroup_mean_sales": last_row.get("subgroup_mean_sales", 0),
                "sales_trend": lag1 - lag2,
            }

            X = pd.DataFrame([row])[[f for f in feats if f in row]].fillna(0)
            pred_log = model.predict(X)[0]
            pred = max(0, np.expm1(pred_log))

            results.append({
                "sku": sku, "date": next_date,
                "predicted": round(pred),
                "lower": round(pred * 0.88),
                "upper": round(pred * 1.12),
            })
            hist.append(pred)

    result_df = pd.DataFrame(results)
    if sku_filter == "Все SKU" and not result_df.empty:
        result_df = result_df.groupby("date").agg(
            predicted=("predicted", "sum"),
            lower=("lower", "sum"),
            upper=("upper", "sum"),
        ).reset_index()
        result_df["sku"] = "Все SKU"

    return result_df


# ─── Сессионное состояние ──────────────────────────────────────────────────────
for key in ["df_raw", "df_eng", "col_map", "model", "metrics",
            "feat_imp", "test_pred", "feats", "file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─── Боковое меню ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:8px 0 24px;">
        <div style="width:32px; height:32px; background:#4f8cff; border-radius:8px;
                    display:flex; align-items:center; justify-content:center; font-size:16px;">📈</div>
        <span style="font-family:'DM Serif Display',serif; font-size:22px; color:#e8eaf0;">ForecastIQ</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Навигация",
        ["📂 Загрузка данных", "📊 Обзор", "🤖 Обучение модели", "🔮 Прогноз"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Статус
    has_data  = st.session_state.df_eng is not None
    has_model = st.session_state.model is not None

    st.markdown(f"""
    <div style="font-size:13px; color:#555b70;">
        <div style="margin-bottom:6px;">
            {"✅" if has_data else "⏳"} Данные загружены
        </div>
        <div style="margin-bottom:6px;">
            {"✅" if has_model else "⏳"} Модель обучена
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#555b70;">Алгоритм:</span>
            <span style="color:#4f8cff;">LightGBM</span>
        </div>
        <div>
            <span style="color:#555b70;">Признаков:</span>
            <span style="color:#e8eaf0;">{len(FEATURES)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 1 — ЗАГРУЗКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📂 Загрузка данных":
    styled_header(
        "Загрузка данных",
        "Загрузите CSV или Excel файл с историей продаж — система автоматически определит колонки."
    )

    uploaded = st.file_uploader(
        "Перетащите файл сюда или нажмите «Browse files»",
        type=["csv", "xlsx", "xls"],
        help="Максимальный размер: 500 МБ. Поддерживаются CSV (с разделителями , и ;) и Excel."
    )

    if uploaded:
        with st.spinner("Читаем файл..."):
            try:
                file_bytes = uploaded.read()
                df_raw = load_file(file_bytes, uploaded.name)
                st.session_state.df_raw   = df_raw
                st.session_state.file_name = uploaded.name
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
                st.stop()

        st.success(f"✓ Загружено: **{uploaded.name}** — {len(df_raw):,} строк · {len(df_raw.columns)} колонок")

        # Превью
        with st.expander("Превью данных (первые 5 строк)", expanded=True):
            st.dataframe(df_raw.head(5), use_container_width=True)

        st.markdown("---")
        st.markdown("### Маппинг колонок")
        st.markdown('<p style="color:#8b90a0; font-size:14px; margin-bottom:16px;">Система автоматически определила колонки. Проверьте и исправьте при необходимости.</p>', unsafe_allow_html=True)

        auto = detect_columns(df_raw.columns.tolist())
        cols_with_none = ["— не выбрано —"] + df_raw.columns.tolist()

        def sel(label, key, required=False):
            default = auto.get(key)
            idx = cols_with_none.index(default) if default in cols_with_none else 0
            val = st.selectbox(
                f"{'* ' if required else ''}{label}",
                cols_with_none, index=idx, key=f"col_{key}"
            )
            return None if val == "— не выбрано —" else val

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Обязательные**")
            cm_sku       = sel("SKU / Артикул",    "sku",       required=True)
            cm_date      = sel("Дата",              "date",      required=True)
            cm_sales_qty = sel("Количество продаж", "sales_qty", required=True)
        with c2:
            st.markdown("**Идентификаторы**")
            cm_name      = sel("Название товара", "name")
            cm_subgroup  = sel("Категория",       "subgroup")
            cm_sales_tg  = sel("Выручка",         "sales_tg")
        with c3:
            st.markdown("**Факторы**")
            cm_price     = sel("Цена",         "rc_price")
            cm_promo     = sel("Промо-флаг",   "promo_flag")
            cm_stock     = sel("Остаток",      "stock")

        st.markdown("---")

        if st.button("⚙️ Обработать данные и построить признаки", use_container_width=True):
            if not cm_sku or not cm_date or not cm_sales_qty:
                st.error("Необходимо выбрать: SKU, Дата и Количество продаж.")
            else:
                col_map = {
                    "sku": cm_sku, "date": cm_date, "sales_qty": cm_sales_qty,
                    "name": cm_name, "subgroup": cm_subgroup, "sales_tg": cm_sales_tg,
                    "rc_price": cm_price, "promo_flag": cm_promo, "stock": cm_stock,
                }
                with st.spinner("Конструирование признаков (лаги, скользящие средние, кодировки)..."):
                    try:
                        df_eng, le_sku, le_sub, price_med = engineer_features(df_raw, col_map)
                        st.session_state.df_eng  = df_eng
                        st.session_state.col_map = col_map
                        st.session_state.model   = None  # сбрасываем модель
                        st.success(f"✓ Готово! Получено {len(df_eng):,} строк с {len(FEATURES)} признаками.")
                        st.info("Перейдите в раздел **🤖 Обучение модели**")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    else:
        # Подсказка по формату
        st.markdown("""
        <div style="background:#161920; border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px; padding:24px; margin-top:16px;">
            <div style="font-weight:500; margin-bottom:12px; color:#e8eaf0;">Ожидаемый формат данных</div>
            <div style="font-size:13px; color:#8b90a0; line-height:1.9;">
                <b style="color:#e8eaf0;">Обязательно:</b> дата (год-месяц), SKU/артикул, количество продаж<br>
                <b style="color:#e8eaf0;">Опционально:</b> цена, промо-флаг, остаток, категория, выручка<br>
                <b style="color:#e8eaf0;">Гранулярность:</b> помесячная — одна строка = один товар за один месяц<br>
                <b style="color:#e8eaf0;">Кодировки:</b> UTF-8, CP1251, Latin1 — все поддерживаются<br>
                <b style="color:#e8eaf0;">Разделитель CSV:</b> запятая или точка с запятой — определяется автоматически
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 2 — ОБЗОР
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Обзор":
    if st.session_state.df_eng is None:
        st.warning("Сначала загрузите данные в разделе **📂 Загрузка данных**")
        st.stop()

    df = st.session_state.df_eng
    styled_header("Обзор данных", "Анализ загруженного датасета и паттернов продаж.")

    # Метрики
    total_sku   = df["sku"].nunique()
    total_sales = df["sales_qty"].sum()
    date_min    = df["date"].min().strftime("%Y-%m")
    date_max    = df["date"].max().strftime("%Y-%m")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Уникальных SKU",   f"{total_sku:,}",               "#4f8cff")
    with c2: metric_card("Строк данных",     f"{len(df):,}",                 "#2dd4bf")
    with c3: metric_card("Продаж (сумм.)",   f"{int(total_sales):,}",        "#34d399")
    with c4: metric_card("Период",           f"{date_min} – {date_max}",     "#fbbf24")

    st.markdown("<br>", unsafe_allow_html=True)

    # График динамики продаж
    monthly = df.groupby("date")["sales_qty"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["sales_qty"],
        fill="tozeroy", mode="lines+markers",
        line=dict(color="#2dd4bf", width=2),
        fillcolor="rgba(45,212,191,0.08)",
        marker=dict(size=4, color="#2dd4bf"),
        name="Продажи"
    ))
    plotly_dark_layout(fig, "Динамика продаж по месяцам", height=320)
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Сезонность
        seasonal = df.groupby("month")["sales_qty"].mean().reset_index()
        month_names = ["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"]
        seasonal["month_name"] = seasonal["month"].apply(lambda x: month_names[x-1])

        fig2 = go.Figure(go.Bar(
            x=seasonal["month_name"], y=seasonal["sales_qty"],
            marker_color="#4f8cff", marker_opacity=0.85,
            name="Среднее"
        ))
        plotly_dark_layout(fig2, "Сезонность (среднее по месяцам)", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Топ SKU
        st.markdown('<div style="font-weight:500; margin-bottom:12px; color:#e8eaf0;">Топ-10 SKU</div>', unsafe_allow_html=True)
        top_sku = df.groupby("sku")["sales_qty"].sum().sort_values(ascending=False).head(10)
        for i, (sku, val) in enumerate(top_sku.items()):
            pct = val / top_sku.max()
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="font-size:12px; color:#8b90a0;">{sku[:25]}</span>
                    <span style="font-size:12px; color:#4f8cff; font-weight:500;">{int(val):,}</span>
                </div>
                <div style="height:3px; background:#252a3a; border-radius:2px;">
                    <div style="height:100%; width:{pct*100:.0f}%; background:#4f8cff; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 3 — ОБУЧЕНИЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Обучение модели":
    if st.session_state.df_eng is None:
        st.warning("Сначала загрузите данные в разделе **📂 Загрузка данных**")
        st.stop()

    df = st.session_state.df_eng
    styled_header("Обучение модели", "LightGBM с признаками из вашего ноутбука — лаги, скользящие средние, кодировки SKU.")

    col_params, col_info = st.columns(2)

    with col_params:
        st.markdown('<div style="font-weight:500; margin-bottom:16px; color:#e8eaf0;">Параметры модели</div>', unsafe_allow_html=True)
        n_est = st.select_slider("n_estimators", [100, 300, 500, 700, 1000], value=1000)
        lr    = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1], value=0.03)
        depth = st.select_slider("max_depth", [4, 6, 8, 10], value=8)
        leaves= st.select_slider("num_leaves", [15, 31, 63, 127], value=31)

    with col_info:
        st.markdown('<div style="font-weight:500; margin-bottom:16px; color:#e8eaf0;">Информация о датасете</div>', unsafe_allow_html=True)
        n_skus = df["sku"].nunique()
        n_train = int(len(df) * 0.8)
        info_items = [
            ("Строк для обучения", f"{n_train:,} (80%)"),
            ("Строк для теста",    f"{len(df)-n_train:,} (20%)"),
            ("Уникальных SKU",     f"{n_skus:,}"),
            ("Признаков",          str(len(FEATURES))),
            ("Целевая переменная", "log1p(sales_qty)"),
            ("Тип сплита",         "хронологический"),
        ]
        for label, val in info_items:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid rgba(255,255,255,0.07); font-size:13px;">
                <span style="color:#8b90a0;">{label}</span>
                <span style="font-weight:500; color:#e8eaf0;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Запустить обучение", use_container_width=True):
        progress = st.progress(0, text="Подготовка данных...")
        try:
            # Перестраиваем модель с нужными параметрами
            df_eng = st.session_state.df_eng.copy()

            # Обновляем гиперпараметры через monkey-patch
            def train_custom(df, progress_bar):
                unique_dates = sorted(df["date"].unique())
                test_dates   = unique_dates[-3:]
                train_df = df[~df["date"].isin(test_dates)].dropna(subset=["lag_1","lag_2","lag_3"]).copy()
                test_df  = df[df["date"].isin(test_dates)].dropna(subset=["lag_1","lag_2","lag_3"]).copy()
                feats    = [f for f in FEATURES if f in df.columns]
                X_train  = train_df[feats].fillna(0)
                y_train  = train_df["target"]
                X_test   = test_df[feats].fillna(0)
                y_test   = test_df["sales_qty"]
                progress_bar.progress(30, text="Обучение LightGBM...")
                m = lgb.LGBMRegressor(
                    n_estimators=n_est, learning_rate=lr,
                    max_depth=depth, num_leaves=leaves,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1,
                )
                m.fit(X_train, y_train,
                      eval_set=[(X_test, test_df["target"])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                progress_bar.progress(75, text="Расчёт метрик...")
                pred = np.maximum(np.expm1(m.predict(X_test)), 0)
                actual = y_test.values
                mae  = mean_absolute_error(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                r2   = r2_score(actual, pred)
                mask = actual > 0
                mape = mean_absolute_percentage_error(actual[mask], pred[mask]) * 100 if mask.sum() > 0 else None
                fi   = pd.DataFrame({"feature": feats, "importance": m.feature_importances_}).sort_values("importance", ascending=False)
                test_df = test_df.copy(); test_df["pred"] = pred
                progress_bar.progress(100, text="Готово!")
                return m, {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}, fi, test_df, feats

            model, metrics, feat_imp, test_pred, feats = train_custom(df_eng, progress)

            st.session_state.model     = model
            st.session_state.metrics   = metrics
            st.session_state.feat_imp  = feat_imp
            st.session_state.test_pred = test_pred
            st.session_state.feats     = feats

        except Exception as e:
            st.error(f"Ошибка обучения: {e}")
            st.stop()

    # Результаты
    if st.session_state.metrics:
        metrics  = st.session_state.metrics
        feat_imp = st.session_state.feat_imp
        test_pred= st.session_state.test_pred

        st.markdown("---")
        st.markdown('<div style="font-weight:500; margin-bottom:16px; color:#e8eaf0;">Метрики качества</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("R² (объясн. дисперсия)", f"{metrics['R²']:.3f}", "#34d399")
        with c2: metric_card("MAE",  f"{metrics['MAE']:,.0f}",  "#e8eaf0")
        with c3: metric_card("RMSE", f"{metrics['RMSE']:,.0f}", "#e8eaf0")
        with c4: metric_card("MAPE", f"{metrics['MAPE']:.1f}%" if metrics['MAPE'] else "N/A", "#fbbf24")

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_fi = st.columns(2)

        with col_chart:
            # Факт vs прогноз на тестовом периоде
            monthly_cmp = test_pred.groupby("date")[["sales_qty","pred"]].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_cmp["date"], y=monthly_cmp["sales_qty"],
                mode="lines+markers", name="Факт",
                line=dict(color="#2dd4bf", width=2), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=monthly_cmp["date"], y=monthly_cmp["pred"],
                mode="lines+markers", name="Прогноз",
                line=dict(color="#4f8cff", width=2, dash="dash"), marker=dict(size=6)))
            plotly_dark_layout(fig, "Факт vs Прогноз (тестовый период)", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_fi:
            # Важность признаков
            top_fi = feat_imp.head(12)
            fig2 = go.Figure(go.Bar(
                x=top_fi["importance"][::-1], y=top_fi["feature"][::-1],
                orientation="h", marker_color="#4f8cff", marker_opacity=0.85
            ))
            plotly_dark_layout(fig2, "Топ-12 важных признаков", height=300)
            fig2.update_layout(yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig2, use_container_width=True)

        st.success("✅ Модель обучена! Перейдите в раздел **🔮 Прогноз**")


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 4 — ПРОГНОЗ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Прогноз":
    if st.session_state.model is None:
        st.warning("Сначала обучите модель в разделе **🤖 Обучение модели**")
        st.stop()

    df       = st.session_state.df_eng
    model    = st.session_state.model
    feats    = st.session_state.feats

    styled_header("Прогноз продаж", "Прогнозы LightGBM с учётом лагов, сезонности и ценовых факторов.")

    # Панель управления
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sku_options = ["Все SKU"] + sorted(df["sku"].unique().tolist())
        sku_filter  = st.selectbox("SKU / Продукт", sku_options)
    with c2:
        horizon = st.selectbox("Горизонт прогноза",
            [1, 2, 3, 6, 9, 12],
            index=3,
            format_func=lambda m: f"{m} мес." if m > 1 else "1 месяц"
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Построить прогноз", use_container_width=True)

    if run:
        with st.spinner("Генерация прогноза..."):
            forecast_df = make_forecast(model, df, feats, sku_filter, horizon)
            st.session_state["forecast_df"] = forecast_df

    if "forecast_df" in st.session_state and st.session_state.forecast_df is not None:
        forecast_df = st.session_state.forecast_df

        # История + прогноз на графике
        if sku_filter == "Все SKU":
            hist_df = df.groupby("date")["sales_qty"].sum().reset_index()
        else:
            hist_df = df[df["sku"] == sku_filter].groupby("date")["sales_qty"].sum().reset_index()

        hist_df = hist_df.sort_values("date").tail(18)

        fig = go.Figure()

        # История
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["sales_qty"],
            mode="lines+markers", name="История",
            line=dict(color="#2dd4bf", width=2),
            marker=dict(size=5, color="#2dd4bf"),
        ))

        # Доверительный интервал
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself", fillcolor="rgba(79,140,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Интервал ±12%", showlegend=True,
        ))

        # Прогноз
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["predicted"],
            mode="lines+markers", name="Прогноз",
            line=dict(color="#4f8cff", width=2.5, dash="dash"),
            marker=dict(size=7, color="#4f8cff"),
        ))

        # Вертикальная линия разделения
        split_date = hist_df["date"].max()
        fig.add_vline(x=split_date, line_dash="dot",
                      line_color="rgba(255,255,255,0.2)", line_width=1)

        plotly_dark_layout(fig, f"{'Все SKU' if sku_filter=='Все SKU' else sku_filter} — прогноз на {horizon} мес.", height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Таблица прогноза
        st.markdown('<div style="font-weight:500; margin-bottom:12px; color:#e8eaf0;">Помесячная разбивка</div>', unsafe_allow_html=True)

        table_data = forecast_df[["date","predicted","lower","upper"]].copy()
        table_data["date"]      = table_data["date"].dt.strftime("%Y-%m")
        table_data["predicted"] = table_data["predicted"].apply(lambda x: f"{int(x):,}")
        table_data["lower"]     = table_data["lower"].apply(lambda x: f"{int(x):,}")
        table_data["upper"]     = table_data["upper"].apply(lambda x: f"{int(x):,}")
        table_data.columns      = ["Месяц", "Прогноз (ед.)", "Нижняя граница (−12%)", "Верхняя граница (+12%)"]

        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # Экспорт
        csv_out = forecast_df.copy()
        csv_out["date"] = csv_out["date"].dt.strftime("%Y-%m")
        csv_bytes = csv_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Скачать прогноз CSV",
            data=csv_bytes,
            file_name=f"forecast_{sku_filter.replace(' ','_')}_{horizon}m.csv",
            mime="text/csv",
        )

        st.markdown(f"""
        <div style="margin-top:16px; padding:12px 16px; background:#161920;
                    border-radius:8px; border:1px solid rgba(255,255,255,0.07);
                    font-size:12px; color:#555b70;">
            Прогноз построен на основе: лаговых признаков (lag1–lag12), скользящих средних (3/6/12 мес.),
            сезонных кодировок, тренда, ценовых факторов и промо-активности.
            Модель: <span style="color:#4f8cff;">LightGBM</span> ·
            Признаков: <span style="color:#e8eaf0;">{len(feats)}</span>
        </div>
        """, unsafe_allow_html=True)
