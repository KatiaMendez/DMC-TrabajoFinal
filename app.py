"""
InsuranceCompany | Exploración de Datos (EDA)
Aplicación Streamlit para el Caso de Estudio N.° 3 – Python for Analytics.

Ejecución:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InsuranceCompany | EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="deep")
EXPECTED_COLUMNS = {
    "id",
    "perc_premium_paid_by_cash_credit",
    "age_in_days",
    "Income",
    "Count_3-6_months_late",
    "Count_6-12_months_late",
    "Count_more_than_12_months_late",
    "application_underwriting_score",
    "no_of_premiums_paid",
    "sourcing_channel",
    "residence_area_type",
    "premium",
    "renewal",
}


# ---------------------------------------------------------------------------
# Capa de análisis (POO)
# ---------------------------------------------------------------------------
class DataAnalyzer:
    """Centraliza validaciones y operaciones descriptivas sobre un DataFrame."""

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("El objeto recibido no es un DataFrame de pandas.")
        if data.empty:
            raise ValueError("El archivo no contiene registros para analizar.")
        if data.shape[1] == 0:
            raise ValueError("El archivo no contiene columnas.")
        self.df = data.copy()

    def variable_types(self) -> tuple[list[str], list[str]]:
        numeric = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical = [c for c in self.df.columns if c not in numeric]
        return numeric, categorical

    def missing_summary(self) -> pd.DataFrame:
        result = pd.DataFrame({
            "Variable": self.df.columns,
            "Valores faltantes": self.df.isna().sum().values,
        })
        result["% faltantes"] = (
            result["Valores faltantes"] / max(len(self.df), 1) * 100
        ).round(2)
        return result.sort_values(
            ["Valores faltantes", "Variable"], ascending=[False, True]
        ).reset_index(drop=True)

    def descriptive_stats(self, columns: list[str] | None = None) -> pd.DataFrame:
        data = self.df if columns is None else self.df[columns]
        numeric = data.select_dtypes(include=np.number)
        if numeric.empty:
            return pd.DataFrame()
        stats = numeric.describe().T
        stats["mediana"] = numeric.median()
        modes = numeric.mode(dropna=True)
        stats["moda"] = modes.iloc[0] if not modes.empty else np.nan
        return stats

    def categorical_summary(self, column: str) -> pd.DataFrame:
        counts = self.df[column].astype("string").fillna("(Sin dato)").value_counts(
            dropna=False
        )
        result = counts.rename_axis("Categoría").reset_index(name="Frecuencia")
        result["Proporción (%)"] = (
            result["Frecuencia"] / max(result["Frecuencia"].sum(), 1) * 100
        ).round(2)
        return result

    def renewal_column(self) -> str | None:
        return "renewal" if "renewal" in self.df.columns else None


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normaliza encabezados y detecta problemas sin descartar filas silenciosamente."""
    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if any(not c for c in df.columns):
        raise ValueError("Hay columnas sin nombre. Corrige los encabezados del CSV.")
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Hay nombres de columnas duplicados: "
            + ", ".join(map(str, duplicated))
            + ". Renómbralos antes de continuar."
        )

    # Las cadenas vacías se consideran valores faltantes.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    if df.columns.tolist() != raw.columns.tolist():
        notes.append("Se eliminaron espacios al inicio/final de los nombres de columnas.")
    if df.isna().any().any():
        notes.append("Se detectaron valores faltantes; no se imputaron automáticamente.")
    if "id" in df.columns and df["id"].duplicated().any():
        notes.append(
            f"La columna id contiene {int(df['id'].duplicated().sum())} "
            "identificadores repetidos."
        )
    return df, notes


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_number(value: Any, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/D"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def plot_to_streamlit(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_insight(text: str) -> None:
    st.info(text)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        or c in {"renewal", "id"}
    ]


def convert_renewal_for_display(series: pd.Series) -> pd.Series:
    """Convierte 0/1 en etiquetas legibles, sin alterar la columna original."""
    mapping = {1: "Renovó (1)", 0: "No renovó (0)", "1": "Renovó (1)", "0": "No renovó (0)"}
    return series.map(mapping).fillna(series.astype("string").fillna("(Sin dato)"))


# ---------------------------------------------------------------------------
# Encabezado y navegación
# ---------------------------------------------------------------------------
st.title("📊 InsuranceCompany | Análisis Exploratorio de Datos")
st.caption(
    "Caso de Estudio N.° 3 · Especialización en Python for Analytics · 2026"
)

with st.sidebar:
    st.sidebar.image("LogoKRMC-rectangular.png")
    st.header("Navegación")
    module = st.radio(
        "Selecciona un módulo",
        ["Inicio", "Carga del dataset", "Análisis exploratorio (EDA)", "Conclusiones"],
        index=0,
    )
    st.divider()
    st.caption("Herramientas: Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn")
    st.caption("El análisis es exploratorio; no genera modelos predictivos.")


# El uploader está disponible en todo momento. No se usa un dataset local
# implícito: los análisis solo se habilitan tras cargar y validar un CSV.
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Carga InsuranceCompany.csv",
        type=["csv"],
        help="Carga un archivo CSV codificado en UTF-8. El archivo se procesa en memoria.",
    )

df: pd.DataFrame | None = None
analyzer: DataAnalyzer | None = None
load_error: str | None = None
load_notes: list[str] = []

if uploaded_file is not None:
    try:
        if uploaded_file.size == 0:
            raise ValueError("El archivo está vacío.")
        if uploaded_file.size > 200 * 1024 * 1024:
            raise ValueError("El archivo supera el límite de 200 MB.")
        uploaded_file.seek(0)
        try:
            raw_df = pd.read_csv(uploaded_file, low_memory=False)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="latin-1", low_memory=False)
        if raw_df.empty:
            raise ValueError("El CSV no contiene filas de datos.")
        df, load_notes = clean_dataframe(raw_df)
        analyzer = DataAnalyzer(df)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"No se pudo procesar el archivo. Detalle: {exc}"

if load_error:
    st.error(f"**No se pudo cargar el dataset:** {load_error}")
    st.caption("Verifica que el archivo sea un CSV válido, con encabezados y separadores consistentes.")
elif df is not None and analyzer is not None:
    for note in load_notes:
        st.warning(note)
    missing_expected = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing_expected:
        st.warning(
            "El archivo no coincide completamente con el esquema esperado de "
            "InsuranceCompany. Algunas visualizaciones se adaptarán a las columnas "
            "disponibles. Faltan: " + ", ".join(missing_expected)
        )
    if "renewal" not in df.columns:
        st.warning(
            "No se encontró la columna **renewal**. Los análisis generales estarán "
            "disponibles, pero los análisis y conclusiones sobre renovación se omitirán."
        )


# ---------------------------------------------------------------------------
# Módulo 1: Home
# ---------------------------------------------------------------------------
if module == "Inicio":
    st.subheader("Bienvenido")
    st.write(
        "Esta aplicación permite explorar de forma interactiva los datos históricos "
        "de clientes de una compañía de seguros, revisar su calidad y observar "
        "patrones relacionados con la renovación de pólizas."
    )
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Objetivo")
        st.write(
            "Aplicar técnicas de análisis exploratorio de datos (EDA), estadística "
            "descriptiva y visualización para apoyar la comprensión del dataset y "
            "la toma de decisiones. No se construyen modelos predictivos."
        )
        st.markdown("#### ¿Qué contiene el dataset?")
        st.write(
            "Variables demográficas y económicas, historial de pagos tardíos, "
            "canal de captación, tipo de residencia, prima, puntaje de evaluación "
            "y un indicador de renovación."
        )
    with right:
        st.markdown("#### Ficha del proyecto")
        st.markdown(
            """
            - **Autora:** Katia Roxana Mendez Cortez
            - **Programa:** Especialización en Python for Analytics
            - **Año:** 2026
            - **Tecnologías:** Python, Streamlit, Pandas, NumPy, Matplotlib y Seaborn
            """
        )
    st.divider()
    if df is None:
        st.info("Para comenzar, carga el archivo CSV desde el panel lateral. Los análisis se habilitarán cuando el archivo sea válido.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{len(df):,}")
        c2.metric("Columnas", f"{df.shape[1]:,}")
        c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")
        st.success("Dataset cargado y listo para explorar. Abre «Carga del dataset» o «Análisis exploratorio (EDA)».")


# ---------------------------------------------------------------------------
# Módulo 2: Carga del dataset
# ---------------------------------------------------------------------------
elif module == "Carga del dataset":
    st.subheader("Carga y validación del dataset")
    st.write("Carga el CSV desde el panel lateral. Antes del análisis se validan su lectura, estructura y encabezados.")
    if load_error:
        st.error(load_error)
    elif df is None:
        st.warning("Todavía no has cargado un archivo. Usa el cargador del panel lateral para continuar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{df.shape[0]:,}")
        c2.metric("Variables", f"{df.shape[1]:,}")
        c3.metric("Valores faltantes", f"{int(df.isna().sum().sum()):,}")
        st.markdown("#### Vista previa")
        preview_rows = st.slider(
            "Filas para mostrar", min_value=5, max_value=min(50, max(5, len(df))),
            value=min(10, max(5, len(df))), step=5,
        )
        st.dataframe(df.head(preview_rows), use_container_width=True)
        st.markdown("#### Estructura de columnas")
        schema = pd.DataFrame({
            "Variable": df.columns,
            "Tipo de dato": [str(df[c].dtype) for c in df.columns],
            "No nulos": [int(df[c].notna().sum()) for c in df.columns],
            "Nulos": [int(df[c].isna().sum()) for c in df.columns],
            "Únicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar dataset cargado (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="InsuranceCompany_validado.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Módulo 3: EDA, con los 10 ítems requeridos
# ---------------------------------------------------------------------------
elif module == "Análisis exploratorio (EDA)":
    st.subheader("Análisis exploratorio de datos")
    if load_error:
        st.error("Corrige el archivo cargado antes de ejecutar el análisis.")
    elif df is None or analyzer is None:
        st.warning("El análisis está bloqueado hasta que cargues un CSV válido desde el panel lateral.")
    else:
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        renewal_col = analyzer.renewal_column()
        tabs = st.tabs([
            "1. Información", "2. Tipos", "3. Descriptiva", "4. Faltantes",
            "5. Distribuciones", "6. Categóricas", "7. Numérica vs renovación",
            "8. Categórica vs renovación", "9. Análisis dinámico", "10. Hallazgos",
        ])

        # Ítem 1
        with tabs[0]:
            st.markdown("#### Información general del dataset")
            st.write(f"El dataset contiene **{len(df):,} filas** y **{df.shape[1]:,} columnas**.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Tipos de datos**")
                st.dataframe(
                    pd.DataFrame({"Variable": df.columns, "Tipo": [str(x) for x in df.dtypes]}),
                    use_container_width=True, hide_index=True,
                )
            with col_b:
                st.markdown("**Conteo de valores nulos**")
                st.dataframe(
                    analyzer.missing_summary(), use_container_width=True, hide_index=True
                )
            st.markdown("**Resumen equivalente a `DataFrame.info()`**")
            info_buffer = BytesIO()
            # StringIO es más adecuado para la salida textual de .info().
            from io import StringIO
            buffer = StringIO()
            df.info(buf=buffer)
            st.code(buffer.getvalue(), language="text")
            st.caption("La información se calcula sobre el archivo cargado; no se eliminan filas automáticamente.")

        # Ítem 2
        with tabs[1]:
            st.markdown("#### Clasificación de variables")
            st.write("Clasificación basada en el tipo de dato detectado. Los identificadores y variables binarias pueden requerir interpretación de negocio.")
            nums, cats = analyzer.variable_types()
            c1, c2 = st.columns(2)
            c1.metric("Variables numéricas", len(nums))
            c2.metric("Variables categóricas / texto", len(cats))
            c1.markdown("**Numéricas**")
            c1.dataframe(pd.DataFrame({"Variable": nums}), use_container_width=True, hide_index=True)
            c2.markdown("**Categóricas / texto**")
            c2.dataframe(pd.DataFrame({"Variable": cats}), use_container_width=True, hide_index=True)
            if "renewal" in df.columns and pd.api.types.is_numeric_dtype(df["renewal"]):
                st.info("Nota: `renewal` es numérica en el CSV, pero representa una categoría/indicador de negocio.")

        # Ítem 3
        with tabs[2]:
            st.markdown("#### Estadísticas descriptivas")
            stats = analyzer.descriptive_stats()
            if stats.empty:
                st.info("No hay columnas numéricas para calcular estadísticas.")
            else:
                st.dataframe(stats.round(3), use_container_width=True)
                chosen = st.selectbox("Variable numérica para interpretar", numeric_cols, key="stats_var")
                series = safe_numeric(df[chosen]).dropna()
                if series.empty:
                    st.info("La variable seleccionada no contiene valores numéricos válidos.")
                else:
                    mean_v, median_v, std_v = series.mean(), series.median(), series.std()
                    a, b, c = st.columns(3)
                    a.metric("Media", format_number(mean_v))
                    b.metric("Mediana", format_number(median_v))
                    c.metric("Desviación estándar", format_number(std_v))
                    if mean_v > median_v:
                        st.write("La media es mayor que la mediana; puede existir asimetría hacia valores altos o presencia de valores extremos.")
                    elif mean_v < median_v:
                        st.write("La media es menor que la mediana; puede existir asimetría hacia valores bajos.")
                    else:
                        st.write("La media y la mediana son iguales o muy cercanas; revisa el histograma para comprender la forma de la distribución.")

        # Ítem 4
        with tabs[3]:
            st.markdown("#### Análisis de valores faltantes")
            missing = analyzer.missing_summary()
            total_missing = int(df.isna().sum().sum())
            st.metric("Celdas faltantes", f"{total_missing:,}")
            if total_missing == 0:
                st.success("No se detectaron valores faltantes en el dataset.")
            else:
                missing_nonzero = missing[missing["Valores faltantes"] > 0]
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=missing_nonzero, x="% faltantes", y="Variable", ax=ax)
                ax.set_title("Porcentaje de valores faltantes por variable")
                ax.set_xlabel("% faltantes")
                ax.set_ylabel("")
                plot_to_streamlit(fig)
                st.dataframe(missing_nonzero, use_container_width=True, hide_index=True)
                render_insight(
                    "Los valores faltantes pueden afectar comparaciones y estadísticas. "
                    "Antes de imputar o excluir registros, conviene validar su causa con el área dueña del dato."
                )

        # Ítem 5
        with tabs[4]:
            st.markdown("#### Distribución de variables numéricas")
            if not numeric_cols:
                st.info("No se detectaron variables numéricas.")
            else:
                selected_num = st.selectbox("Variable a visualizar", numeric_cols, key="hist_var")
                values = safe_numeric(df[selected_num]).dropna()
                if values.empty:
                    st.warning("No hay observaciones numéricas válidas para esta variable.")
                else:
                    bins = st.slider("Número de intervalos (bins)", 5, 80, 30, key="hist_bins")
                    fig, ax = plt.subplots(figsize=(9, 4))
                    sns.histplot(values, bins=bins, kde=(len(values) > 1 and values.nunique() > 1), ax=ax)
                    ax.set_title(f"Distribución de {selected_num}")
                    ax.set_xlabel(selected_num)
                    ax.set_ylabel("Frecuencia")
                    plot_to_streamlit(fig)
                    st.write(
                        f"Observaciones válidas: **{len(values):,}** · "
                        f"Media: **{format_number(values.mean())}** · "
                        f"Mediana: **{format_number(values.median())}** · "
                        f"Mínimo: **{format_number(values.min())}** · "
                        f"Máximo: **{format_number(values.max())}**."
                    )
                    st.caption("El histograma muestra concentración, dispersión y posibles valores extremos; la interpretación depende del significado de la variable.")

        # Ítem 6
        with tabs[5]:
            st.markdown("#### Análisis de variables categóricas")
            if not categorical_cols:
                st.info("No se detectaron variables categóricas.")
            else:
                cat_col = st.selectbox("Variable categórica", categorical_cols, key="cat_var")
                summary = analyzer.categorical_summary(cat_col)
                max_categories = st.slider("Categorías a mostrar", 3, 30, 10, key="cat_topn")
                shown = summary.head(max_categories)
                st.dataframe(shown, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(9, 4))
                sns.barplot(data=shown, x="Frecuencia", y="Categoría", ax=ax)
                ax.set_title(f"Frecuencia por categoría: {cat_col}")
                plot_to_streamlit(fig)
                st.caption("Las proporciones se calculan sobre todos los registros, incluidos los valores faltantes agrupados como «Sin dato».")

        # Ítem 7
        with tabs[6]:
            st.markdown("#### Análisis bivariado: numérica vs renovación")
            if renewal_col is None:
                st.info("Este análisis requiere la columna `renewal`, no disponible en el archivo.")
            elif not numeric_cols:
                st.info("No hay variables numéricas disponibles.")
            else:
                candidates = [c for c in numeric_cols if c != "id" and c != renewal_col] or numeric_cols
                num_var = st.selectbox("Variable numérica", candidates, key="num_renewal")
                temp = df[[num_var, renewal_col]].copy()
                temp[num_var] = safe_numeric(temp[num_var])
                temp = temp.dropna()
                if temp.empty:
                    st.warning("No hay datos completos para comparar las variables seleccionadas.")
                else:
                    temp["_grupo_renovacion"] = convert_renewal_for_display(temp[renewal_col])
                    fig, ax = plt.subplots(figsize=(9, 4))
                    sns.boxplot(data=temp, x="_grupo_renovacion", y=num_var, ax=ax)
                    ax.set_title(f"{num_var} según renovación")
                    ax.set_xlabel("Renovación")
                    ax.set_ylabel(num_var)
                    plot_to_streamlit(fig)
                    st.dataframe(
                        temp.groupby("_grupo_renovacion", observed=True)[num_var]
                        .agg(["count", "mean", "median", "std", "min", "max"])
                        .round(2).reset_index(),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption("La comparación describe diferencias entre grupos; por sí sola no demuestra causalidad.")

        # Ítem 8
        with tabs[7]:
            st.markdown("#### Análisis bivariado: categórica vs renovación")
            if renewal_col is None:
                st.info("Este análisis requiere la columna `renewal`.")
            else:
                cat_candidates = [c for c in categorical_cols if c != renewal_col]
                # Incluye variables binarias numéricas como renewal, pero no fuerza otras numéricas a categorías.
                if not cat_candidates:
                    st.info("No hay variables categóricas disponibles para comparar.")
                else:
                    cat_var = st.selectbox("Variable categórica", cat_candidates, key="cat_renewal")
                    temp = df[[cat_var, renewal_col]].copy()
                    temp[cat_var] = temp[cat_var].astype("string").fillna("(Sin dato)")
                    temp["_renovacion"] = convert_renewal_for_display(temp[renewal_col])
                    table = pd.crosstab(temp[cat_var], temp["_renovacion"])
                    proportions = pd.crosstab(
                        temp[cat_var], temp["_renovacion"], normalize="index"
                    ).mul(100).round(2)
                    st.markdown("**Frecuencias**")
                    st.dataframe(table, use_container_width=True)
                    st.markdown("**Porcentaje dentro de cada categoría**")
                    st.dataframe(proportions, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    proportions.plot(kind="bar", stacked=True, ax=ax)
                    ax.set_title(f"Proporción de renovación por {cat_var}")
                    ax.set_xlabel(cat_var)
                    ax.set_ylabel("Porcentaje (%)")
                    ax.legend(title="Renovación", bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.xticks(rotation=30, ha="right")
                    plot_to_streamlit(fig)
                    st.caption("Se muestran proporciones por categoría para facilitar comparaciones cuando los tamaños de grupo son distintos.")

        # Ítem 9
        with tabs[8]:
            st.markdown("#### Análisis basado en parámetros seleccionados")
            st.write("Selecciona variables y, opcionalmente, filtra valores para actualizar la vista.")
            selected_columns = st.multiselect(
                "Columnas a incluir", options=df.columns.tolist(),
                default=df.columns.tolist()[: min(5, len(df.columns))],
                key="dynamic_cols",
            )
            if not selected_columns:
                st.warning("Selecciona al menos una columna para mostrar el análisis.")
            else:
                filtered = df[selected_columns].copy()
                if st.checkbox("Aplicar filtro a una variable categórica", key="use_filter"):
                    filter_candidates = [c for c in selected_columns if c in categorical_cols]
                    if not filter_candidates:
                        st.warning("Incluye al menos una columna categórica en la selección para aplicar este filtro.")
                    else:
                        filter_col = st.selectbox("Variable para filtrar", filter_candidates, key="filter_col")
                        values = sorted(df[filter_col].dropna().astype(str).unique().tolist())
                        selected_values = st.multiselect(
                            "Valores incluidos", values, default=values[: min(10, len(values))],
                            key="filter_values",
                        )
                        if not selected_values:
                            filtered = filtered.iloc[0:0]
                        else:
                            filtered = filtered[df[filter_col].astype(str).isin(selected_values)]
                st.metric("Registros en la vista", f"{len(filtered):,}")
                st.dataframe(filtered.head(500), use_container_width=True)
                if len(filtered) > 500:
                    st.caption("Se muestran hasta 500 filas para mantener una interfaz ágil. La descarga incluye todos los registros filtrados.")
                st.download_button(
                    "Descargar selección en CSV",
                    data=filtered.to_csv(index=False).encode("utf-8-sig"),
                    file_name="insurancecompany_seleccion.csv",
                    mime="text/csv",
                    key="download_dynamic",
                )
                numeric_selected = [c for c in selected_columns if c in numeric_cols]
                if numeric_selected:
                    st.markdown("**Resumen estadístico de las columnas numéricas seleccionadas**")
                    st.dataframe(filtered[numeric_selected].describe().T.round(2), use_container_width=True)

        # Ítem 10
        with tabs[9]:
            st.markdown("#### Hallazgos clave")
            st.write("Resumen automático de calidad y distribución. Estos indicadores son descriptivos y deben interpretarse con conocimiento del negocio.")
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = int(df.isna().sum().sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Filas", f"{len(df):,}")
            m2.metric("Columnas", f"{df.shape[1]:,}")
            m3.metric("% de celdas faltantes", f"{(missing_cells / total_cells * 100) if total_cells else 0:.2f}%")
            findings: list[str] = []
            if missing_cells:
                top_missing = analyzer.missing_summary().iloc[0]
                findings.append(
                    f"La variable con mayor cantidad de faltantes es **{top_missing['Variable']}** "
                    f"({int(top_missing['Valores faltantes']):,}; {top_missing['% faltantes']:.2f}%)."
                )
            else:
                findings.append("No se observan valores faltantes en el dataset.")
            findings.append(
                f"Se identificaron **{len(numeric_cols)} variables numéricas** y "
                f"**{len(categorical_cols)} variables categóricas/texto** según sus tipos de datos."
            )
            if "id" in df.columns:
                findings.append(
                    f"Hay **{int(df['id'].duplicated().sum()):,} identificadores repetidos** en la columna `id` "
                    "(si el identificador debe ser único, conviene revisar estos casos)."
                )
            if renewal_col:
                renewal_counts = df[renewal_col].value_counts(dropna=False)
                renewal_pct = df[renewal_col].value_counts(normalize=True, dropna=False) * 100
                st.markdown("**Distribución de renewal**")
                renewal_display = pd.DataFrame({
                    "Valor": renewal_counts.index.astype(str),
                    "Frecuencia": renewal_counts.values,
                    "Proporción (%)": [round(float(renewal_pct.loc[v]), 2) for v in renewal_counts.index],
                })
                st.dataframe(renewal_display, use_container_width=True, hide_index=True)
                if df[renewal_col].nunique(dropna=True) == 2:
                    findings.append(
                        "La variable `renewal` presenta dos valores observados; "
                        "revisa las frecuencias y proporciones anteriores para dimensionar cada grupo."
                    )
                else:
                    findings.append(
                        "`renewal` tiene más de dos valores distintos o una codificación diferente a la esperada; "
                        "verifica su definición antes de interpretar renovaciones."
                    )
            for i, finding in enumerate(findings, 1):
                st.markdown(f"{i}. {finding}")
            st.caption("Los hallazgos se generan a partir del archivo cargado y no sustituyen la validación del área de negocio.")


# ---------------------------------------------------------------------------
# Módulo 4: Conclusiones (5 conclusiones descriptivas, calculadas sobre datos)
# ---------------------------------------------------------------------------
elif module == "Conclusiones":
    st.subheader("Conclusiones finales")
    st.write(
        "Las siguientes conclusiones se generan a partir de los datos cargados. "
        "Revísalas y complementa su interpretación con el contexto del negocio."
    )
    if load_error:
        st.error(load_error)
    elif df is None or analyzer is None:
        st.warning("Carga primero un CSV válido para generar conclusiones basadas en datos.")
    else:
        nums, cats = analyzer.variable_types()
        missing_summary = analyzer.missing_summary()
        total_missing = int(df.isna().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])
        conclusions: list[str] = []
        conclusions.append(
            f"**Tamaño de la información.** El dataset contiene {len(df):,} registros y "
            f"{df.shape[1]} variables, lo que permite realizar comparaciones descriptivas entre grupos."
        )
        if total_missing:
            row = missing_summary.iloc[0]
            conclusions.append(
                f"**Calidad de datos.** Se identificaron {total_missing:,} celdas faltantes "
                f"({total_missing / max(total_cells, 1) * 100:.2f}% del total). "
                f"La variable con mayor porcentaje de faltantes es `{row['Variable']}` "
                f"({row['% faltantes']:.2f}%)."
            )
        else:
            conclusions.append("**Calidad de datos.** No se detectaron celdas faltantes; esto reduce una fuente de sesgo, aunque no garantiza que los datos sean correctos.")
        conclusions.append(
            f"**Tipos de variables.** Se encontraron {len(nums)} variables numéricas y "
            f"{len(cats)} variables categóricas/texto. Conviene confirmar el significado de negocio "
            "de identificadores y variables codificadas numéricamente."
        )
        if "renewal" in df.columns:
            renewal = df["renewal"].value_counts(normalize=True, dropna=False) * 100
            most_common = renewal.idxmax()
            conclusions.append(
                f"**Renovación.** El valor más frecuente de `renewal` es `{most_common}` "
                f"({renewal.max():.2f}% de los registros, incluidos los nulos si existieran). "
                "La codificación debe confirmarse con el diccionario de datos."
            )
        else:
            conclusions.append(
                "**Renovación.** No se encontró la variable `renewal`; no es posible formular una conclusión "
                "sobre renovaciones con este archivo."
            )
        if "id" in df.columns:
            duplicates = int(df["id"].duplicated().sum())
            conclusions.append(
                f"**Identificadores.** Se detectaron {duplicates:,} valores repetidos en `id`. "
                "Si se espera un identificador único por póliza/cliente, estos registros requieren revisión."
            )
        else:
            conclusions.append(
                "**Identificadores.** El archivo no incluye una columna `id`; no se pudo verificar unicidad de registros."
            )
        for i, conclusion in enumerate(conclusions[:5], 1):
            st.markdown(f"**Conclusión {i}.** {conclusion}")
        st.divider()
        st.markdown("#### Recomendaciones para la toma de decisiones")
        st.markdown(
            """
            - Validar la calidad y definición de las variables antes de tomar decisiones.
            - Comparar grupos con tamaños y proporciones, no solo con frecuencias absolutas.
            - Contrastar los patrones observados con el conocimiento de los equipos comerciales y de cobranzas.
            - Evitar inferir causalidad a partir de relaciones descriptivas.
            - Documentar los criterios de limpieza o tratamiento de faltantes antes de cualquier análisis posterior.
            """
        )

st.sidebar.divider()
st.sidebar.caption("InsuranceCompany EDA · Aplicación académica")
