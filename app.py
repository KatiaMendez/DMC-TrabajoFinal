
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="InsuranceCompany | EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", palette="deep")

MAX_FILE_SIZE = 200 * 1024 * 1024
MAX_PLOT_ROWS = 10_000
MAX_TABLE_ROWS = 500
MAX_CATEGORIES = 20

EXPECTED_COLUMNS = {
    "id", "perc_premium_paid_by_cash_credit", "age_in_days",
    "Income", "Count_3-6_months_late", "Count_6-12_months_late",
    "Count_more_than_12_months_late", "application_underwriting_score",
    "no_of_premiums_paid", "sourcing_channel", "residence_area_type",
    "premium", "renewal"
}


# =========================================================
# 2. ESTILOS
# =========================================================
st.markdown("""
<style>
[data-testid="stSidebar"] .block-container {
    padding-top: 0.6rem;
    padding-bottom: 0.4rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
}

[data-testid="stSidebar"] hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    padding-top: 1;
    padding-bottom: 1;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================
def missing_count(df):
    """Normaliza encabezados y valores faltantes."""
    df = df.copy()

    original_columns = df.columns.tolist()
    df.columns = [str(c).strip() for c in df.columns]

    if any(not c for c in df.columns):
        raise ValueError("Existen columnas sin nombre.")

    if df.columns.duplicated().any():
        raise ValueError("Existen nombres de columnas duplicados.")

    for col in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    notes = []

    if original_columns != df.columns.tolist():
        notes.append("Se eliminaron espacios de los encabezados.")

    missing_count = int(df.isna().sum().sum())

    if missing_count:
        notes.append(
            f"Se detectaron {missing_count:,} celdas faltantes. "
            "No se imputaron automáticamente."
        )

    if "id" in df.columns:
        duplicates = int(df["id"].duplicated().sum())

        if duplicates:
            notes.append(
                f"Se encontraron {duplicates:,} identificadores repetidos."
            )

    return df, notes



def load_dataset(file):
    """Carga, valida y prepara el archivo CSV."""
    if file.size == 0:
        raise ValueError("El archivo está vacío.")

    if file.size > MAX_FILE_SIZE:
        raise ValueError("El archivo supera el límite de 200 MB.")

    try:
        file.seek(0)

        try:
            df = pd.read_csv(file, low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, low_memory=False, encoding="latin-1")

    except pd.errors.EmptyDataError as exc:
        raise ValueError("El archivo no contiene datos legibles.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Error en la estructura del CSV: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "El CSV no contiene registros. Verifica sus encabezados y separador."
        )

    # missing_count devuelve el DataFrame limpio y las notas.
    df, notes = missing_count(df)

    return df, notes



def missing_summary(df):
    """Resume los valores faltantes por variable."""
    result = pd.DataFrame({
        "Variable": df.columns,
        "Valores faltantes": df.isna().sum().values
    })

    result["% faltantes"] = (
        result["Valores faltantes"] / len(df) * 100
    ).round(2)

    return result.sort_values(
        "Valores faltantes",
        ascending=False
    ).reset_index(drop=True)


def categorical_summary(df, column):
    """Calcula frecuencias sin mostrar todas las categorías."""
    values = df[column].astype("string").fillna("(Sin dato)")

    result = (
        values.value_counts()
        .rename_axis("Categoría")
        .reset_index(name="Frecuencia")
    )

    result["Proporción (%)"] = (
        result["Frecuencia"] / len(df) * 100
    ).round(2)

    return result


def renewal_labels(series):
    """Convierte 0 y 1 en etiquetas legibles."""
    values = pd.to_numeric(series, errors="coerce")

    result = pd.Series(
        "(Sin dato)",
        index=series.index,
        dtype="object"
    )

    result.loc[values == 1] = "Renovó"
    result.loc[values == 0] = "No renovó"

    other = values.notna() & ~values.isin([0, 1])

    result.loc[other] = (
        series.loc[other].astype(str)
    )

    return result


def number(value):
    """Formatea números y valores ausentes."""
    return "N/D" if pd.isna(value) else f"{value:,.2f}"


def show_plot(fig):
    """Muestra y cierra la figura para liberar memoria."""
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_sample(series, max_rows=MAX_PLOT_ROWS):
    """Muestrea datos solo para gráficos."""
    series = series.dropna()

    if len(series) > max_rows:
        return series.sample(
            n=max_rows,
            random_state=42
        )

    return series


def download_csv(df, filename, key):
    """Descarga un DataFrame como CSV."""
    st.download_button(
        "Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


def get_column_types(df):
    """
    Separa variables numéricas y categóricas.
    Los identificadores y renewal no se tratan como
    variables categóricas ordinarias.
    """
    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        and col not in ["id", "renewal"]
    ]

    return numeric, categorical


def get_numeric_candidates(df):
    """Excluye identificadores e indicadores de renovación."""
    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return [
        col for col in numeric
        if col not in ["id", "renewal"]
    ]


def get_categorical_candidates(df):
    """Excluye identificadores y renovación."""
    return [
        col for col in df.columns
        if col not in ["id", "renewal"]
        and not pd.api.types.is_numeric_dtype(df[col])
    ]


# =========================================================
# 4. ENCABEZADO Y SIDEBAR
# =========================================================
st.title("📊 InsuranceCompany | EDA")
st.caption("Caso de Estudio N.° 3 · 2026")

with st.sidebar:
    st.subheader("Python for Analytics")

    try:
        st.image("LogoKRMC-rectangular.png", width=205)
    except Exception:
        st.caption("InsuranceCompany EDA")

    st.divider()

    st.subheader("Menú principal")

    pagina = st.selectbox(
        "Selecciona una sección:",
        [
            "Inicio",
            "Carga del dataset",
            "Análisis exploratorio (EDA)",
            "Conclusiones"
        ],
        key="menu_navegacion"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Carga archivo para análisis",
        type=["csv"],
        help="Archivo CSV de hasta 200 MB."
    )

    st.divider()
    st.caption("EDA · Aplicación académica")


# =========================================================
# 5. CARGA Y VALIDACIÓN
# =========================================================
df = None
load_error = None
notes = []

if uploaded_file is not None:
    try:
        df, notes = load_dataset(uploaded_file)

    except (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError
    ) as exc:
        load_error = str(exc)

    except Exception:
        load_error = (
            "No se pudo procesar el archivo. "
            "Verifica que sea un CSV válido."
        )

if load_error:
    st.error(f"Error al cargar el dataset: {load_error}")


# =========================================================
# 6. INICIO
# =========================================================
if pagina == "Inicio":

    left, right = st.columns(2)

    with left:
        st.markdown("#### Objetivo")
        st.write(
            "Analizar la estructura, calidad y distribución de los "
            "datos para identificar patrones relacionados con la "
            "renovación de pólizas. No se construyen modelos predictivos."
        )

        st.markdown("#### Dataset")
        st.write(
            "Este dataset contiene información histórica de clientes de "
            "una compañía de seguros. Incluye variables demográficas, económicas,"
            "historial de pagos, comportamiento de morosidad, canal de captación,"
            "tipo de residencia, valor de la prima y puntaje de evaluación del cliente."
        )

    with right:
        st.markdown("#### Ficha del proyecto")
        st.write("**Autora:**")
        st.write("    Katia Roxana Mendez Cortez")
        st.write("**Programa:**")
        st.write("    Especialización en Python for Analytics")
        st.write("**Año:** 2026")
        st.write("**Tecnologías:**")
        st.write("    Python, Streamlit, Pandas, NumPy y Seaborn")

    if df is None:
        st.info("Para continuar carga un CSV válido desde el panel lateral.")

    else:
        st.info("Archivo CSV cargado, para analizar seleccione otra sección del menú.")


# =========================================================
# 7. CARGA DEL DATASET
# =========================================================
elif pagina == "Carga del dataset":
    st.subheader("Carga y validación del dataset")

    if df is None:
        st.warning("Carga un archivo CSV válido para continuar.")

    else:
        if df is not None:
            for note in notes:
                st.warning(note)
        
            missing_columns = sorted(
                EXPECTED_COLUMNS - set(df.columns)
            )
        
            if missing_columns:
                st.warning(
                    "Faltan columnas del esquema esperado: "
                    + ", ".join(missing_columns)
                )
        
            numeric, categorical = get_column_types(df)
        
        a, b, c = st.columns(3)

        a.metric("Registros", f"{len(df):,}")
        b.metric("Variables", f"{df.shape[1]:,}")
        c.metric(
            "Valores faltantes",
            f"{int(df.isna().sum().sum()):,}"
        )

        st.markdown("#### Vista previa")

        rows = st.slider(
            "Filas para mostrar",
            min_value=1,
            max_value=min(50, len(df)),
            value=min(10, len(df)),
            key="preview_rows"
        )

        st.dataframe(
            df.head(rows),
            use_container_width=True,
            hide_index=True
        )

        schema = pd.DataFrame({
            "Variable": df.columns,
            "Tipo de dato": df.dtypes.astype(str).values,
            "No nulos": df.notna().sum().values,
            "Nulos": df.isna().sum().values,
            "Valores únicos": df.nunique().values
        })

        st.markdown("#### Estructura de columnas")

        st.dataframe(
            schema,
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            df,
            "InsuranceCompany_validado.csv",
            "download_clean"
        )



# =========================================================
# 8. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =========================================================
elif pagina == "Análisis exploratorio (EDA)":
    st.subheader("Análisis exploratorio de datos")
    st.caption(
        "Selecciona una pestaña para consultar cada uno de los "
        "diez análisis del dataset."
    )

    if df is None:
        st.warning("Carga un CSV válido para habilitar el análisis.")

    else:
        numeric, categorical = get_column_types(df)

        # Cada análisis tiene su propio tab.
        # Se mantienen los diez análisis originales.
        tabs = st.tabs([
            "1. Información",
            "2. Tipos",
            "3. Descriptiva",
            "4. Faltantes",
            "5. Distribuciones",
            "6. Categóricas",
            "7. Numérica vs renovación",
            "8. Categórica vs renovación",
            "9. Análisis dinámico",
            "10. Hallazgos"
        ])

        # -------------------------------------------------
        # TAB 1. INFORMACIÓN
        # -------------------------------------------------
        with tabs[0]:
            st.markdown("### Información general")
            st.info("Este dataset contiene información histórica de clientes de una compañía de seguros. Incluye variables demográficas, económicas,historial de pagos, comportamiento de morosidad, canal de captación,tipo de residencia, valor de la prima y puntaje de evaluación del cliente.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Registros", f"{len(df):,}")
            col2.metric("Variables", f"{df.shape[1]:,}")
            col3.metric(
                "Celdas faltantes",
                f"{int(df.isna().sum().sum()):,}"
            )

            left, right = st.columns(2)

            with left:
                st.markdown("#### Tipos de datos")

                types_df = pd.DataFrame({
                    "Variable": df.columns,
                    "Tipo": df.dtypes.astype(str).values
                })

                st.dataframe(
                    types_df,
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.markdown("#### Valores faltantes")

                st.dataframe(
                    missing_summary(df),
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("Ver información técnica"):
                buffer = StringIO()
                df.info(buf=buffer)
                st.code(buffer.getvalue())

        # -------------------------------------------------
        # TAB 2. TIPOS
        # -------------------------------------------------
        with tabs[1]:
            st.markdown("### Tipos de variables")

            left, right = st.columns(2)

            with left:
                st.metric("Variables numéricas", len(numeric))
                st.dataframe(
                    pd.DataFrame({"Variable": numeric}),
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.metric(
                    "Variables categóricas/texto",
                    len(categorical)
                )
                st.dataframe(
                    pd.DataFrame({"Variable": categorical}),
                    use_container_width=True,
                    hide_index=True
                )

            if "id" in df.columns:
                st.info(
                    "`id` es un identificador y no se trata como "
                    "una variable categórica ordinaria."
                )

            if "renewal" in df.columns:
                st.info(
                    "`renewal` es un indicador de negocio, aunque "
                    "esté almacenado como número."
                )

        # -------------------------------------------------
        # TAB 3. DESCRIPTIVA
        # -------------------------------------------------
        with tabs[2]:
            st.markdown("### Estadísticas descriptivas")

            if not numeric:
                st.info("No hay variables numéricas disponibles.")

            else:
                stats = df[numeric].describe().T
                stats["Mediana"] = df[numeric].median()

                modes = df[numeric].mode()
                if not modes.empty:
                    stats["Moda"] = modes.iloc[0]

                st.dataframe(
                    stats.round(3),
                    use_container_width=True
                )

                candidates = get_numeric_candidates(df)

                if candidates:
                    variable = st.selectbox(
                        "Variable para interpretar",
                        candidates,
                        key="tab3_variable"
                    )

                    values = pd.to_numeric(
                        df[variable],
                        errors="coerce"
                    ).dropna()

                    if values.empty:
                        st.warning(
                            "No hay valores numéricos válidos."
                        )

                    else:
                        col1, col2, col3 = st.columns(3)

                        col1.metric("Media", number(values.mean()))
                        col2.metric("Mediana", number(values.median()))
                        col3.metric(
                            "Desviación estándar",
                            number(values.std())
                        )

                        if values.mean() > values.median():
                            st.write(
                                "La media supera la mediana; podría "
                                "existir asimetría hacia valores altos."
                            )
                        elif values.mean() < values.median():
                            st.write(
                                "La media es inferior a la mediana; "
                                "podría existir asimetría hacia valores bajos."
                            )
                        else:
                            st.write(
                                "La media y la mediana coinciden "
                                "o son muy cercanas."
                            )
                else:
                    st.info(
                        "No hay variables numéricas distintas de "
                        "`id` y `renewal` para interpretar."
                    )

        # -------------------------------------------------
        # TAB 4. FALTANTES
        # -------------------------------------------------
        with tabs[3]:
            st.markdown("### Análisis de valores faltantes")

            missing = missing_summary(df)
            missing = missing[
                missing["Valores faltantes"] > 0
            ].copy()

            total_missing = int(df.isna().sum().sum())
            total_cells = df.size
            missing_pct = (
                total_missing / total_cells * 100
                if total_cells else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Celdas faltantes", f"{total_missing:,}")
            col2.metric("Porcentaje faltante", f"{missing_pct:.2f}%")
            col3.metric("Variables afectadas", f"{len(missing):,}")

            if missing.empty:
                st.success("No se detectaron valores faltantes.")

            else:
                left, right = st.columns([1.2, 1])

                with left:
                    fig, ax = plt.subplots(figsize=(8, 4))

                    sns.barplot(
                        data=missing.head(30),
                        x="% faltantes",
                        y="Variable",
                        ax=ax
                    )

                    ax.set_title("Porcentaje de valores faltantes")
                    show_plot(fig)

                with right:
                    st.dataframe(
                        missing,
                        use_container_width=True,
                        hide_index=True
                    )

                st.info(
                    "Valida la causa de los faltantes antes de "
                    "imputar o excluir registros."
                )

            st.markdown("#### Identificadores duplicados")

            if "id" in df.columns:
                duplicate_mask = df["id"].duplicated(keep=False)

                col1, col2 = st.columns(2)
                col1.metric(
                    "Identificadores repetidos",
                    f"{int(df['id'].duplicated().sum()):,}"
                )
                col2.metric(
                    "Filas involucradas",
                    f"{int(duplicate_mask.sum()):,}"
                )

                if duplicate_mask.any():
                    with st.expander("Consultar registros repetidos"):
                        st.dataframe(
                            df.loc[duplicate_mask].head(MAX_TABLE_ROWS),
                            use_container_width=True,
                            hide_index=True
                        )

        # -------------------------------------------------
        # TAB 5. DISTRIBUCIONES
        # -------------------------------------------------
        with tabs[4]:
            st.markdown("### Distribuciones numéricas")

            candidates = get_numeric_candidates(df)

            if not candidates:
                st.info("No hay variables numéricas disponibles.")

            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    variable = st.selectbox(
                        "Variable numérica",
                        candidates,
                        key="tab5_variable"
                    )

                with col2:
                    bins = st.slider(
                        "Número de intervalos",
                        min_value=5,
                        max_value=80,
                        value=30,
                        key="tab5_bins"
                    )

                values = pd.to_numeric(
                    df[variable],
                    errors="coerce"
                ).dropna()

                if values.empty:
                    st.warning("No hay valores numéricos válidos.")

                else:
                    plot_values = plot_sample(values)

                    fig, ax = plt.subplots(figsize=(9, 4))
                    sns.histplot(
                        plot_values,
                        bins=bins,
                        kde=plot_values.nunique() > 1,
                        ax=ax
                    )

                    ax.set_title(f"Distribución de {variable}")
                    ax.set_xlabel(variable)
                    ax.set_ylabel("Frecuencia")
                    show_plot(fig)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Media", number(values.mean()))
                    col2.metric("Mediana", number(values.median()))
                    col3.metric("Mínimo", number(values.min()))
                    col4.metric("Máximo", number(values.max()))

                    st.caption(
                        f"Se graficaron {len(plot_values):,} de "
                        f"{len(values):,} observaciones válidas."
                    )

        # -------------------------------------------------
        # TAB 6. CATEGÓRICAS
        # -------------------------------------------------
        with tabs[5]:
            st.markdown("### Análisis de variables categóricas")

            candidates = get_categorical_candidates(df)

            if not candidates:
                st.info(
                    "No hay variables categóricas de texto disponibles."
                )

            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    variable = st.selectbox(
                        "Variable categórica",
                        candidates,
                        key="tab6_variable"
                    )

                with col2:
                    top = st.slider(
                        "Categorías a mostrar",
                        min_value=3,
                        max_value=MAX_CATEGORIES,
                        value=10,
                        key="tab6_top"
                    )

                summary = categorical_summary(df, variable).head(top)

                left, right = st.columns([1, 1.3])

                with left:
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

                with right:
                    fig, ax = plt.subplots(figsize=(8, 4))

                    sns.barplot(
                        data=summary,
                        x="Frecuencia",
                        y="Categoría",
                        ax=ax
                    )

                    ax.set_title(f"Principales categorías: {variable}")
                    show_plot(fig)

                st.caption(
                    f"Categorías únicas: {df[variable].nunique(dropna=True):,}. "
                    f"Se muestran las {len(summary)} más frecuentes."
                )

        # -------------------------------------------------
        # TAB 7. NUMÉRICA VS RENOVACIÓN
        # -------------------------------------------------
        with tabs[6]:
            st.markdown("### Variables numéricas según renovación")

            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")

            else:
                candidates = get_numeric_candidates(df)

                if not candidates:
                    st.info(
                        "No hay otras variables numéricas disponibles."
                    )

                else:
                    variable = st.selectbox(
                        "Variable numérica",
                        candidates,
                        key="tab7_variable"
                    )

                    temp = df[[variable, "renewal"]].copy()
                    temp[variable] = pd.to_numeric(
                        temp[variable],
                        errors="coerce"
                    )

                    temp["Renovación"] = renewal_labels(temp["renewal"])
                    temp = temp.dropna(subset=[variable])
                    temp = temp[
                        temp["Renovación"].isin(
                            ["Renovó", "No renovó"]
                        )
                    ]

                    if temp.empty:
                        st.warning("No hay datos válidos para comparar.")

                    else:
                        plot_df = plot_sample(
                            temp[variable]
                        ).to_frame()

                        # Recupera las etiquetas de renovación
                        # correspondientes a los índices muestreados.
                        plot_df["Renovación"] = temp.loc[
                            plot_df.index, "Renovación"
                        ]

                        fig, ax = plt.subplots(figsize=(8, 4))

                        sns.boxplot(
                            data=plot_df,
                            x="Renovación",
                            y=variable,
                            ax=ax
                        )

                        ax.set_title(f"{variable} según renovación")
                        show_plot(fig)

                        summary = (
                            temp.groupby("Renovación")[variable]
                            .agg([
                                "count", "mean", "median",
                                "std", "min", "max"
                            ])
                        )

                        st.markdown("**Estadísticas de todos los registros válidos**")
                        st.dataframe(
                            summary.round(2),
                            use_container_width=True
                        )

                        st.caption(
                            f"Registros comparados: {len(temp):,}; "
                            f"registros graficados: {len(plot_df):,}."
                        )

                        st.info(
                            "Las diferencias descriptivas no demuestran causalidad."
                        )

        # -------------------------------------------------
        # TAB 8. CATEGÓRICA VS RENOVACIÓN
        # -------------------------------------------------
        with tabs[7]:
            st.markdown("### Renovación por categoría")

            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")

            else:
                candidates = get_categorical_candidates(df)

                if not candidates:
                    st.info("No hay variables categóricas para comparar.")

                else:
                    variable = st.selectbox(
                        "Variable categórica",
                        candidates,
                        key="tab8_variable"
                    )

                    temp = df[[variable, "renewal"]].copy()
                    temp["Renovación"] = renewal_labels(temp["renewal"])

                    temp = temp[
                        temp["Renovación"].isin(
                            ["Renovó", "No renovó"]
                        )
                    ].copy()

                    temp[variable] = (
                        temp[variable]
                        .astype("string")
                        .fillna("(Sin dato)")
                    )

                    if temp.empty:
                        st.warning("No hay datos válidos para comparar.")

                    else:
                        top_categories = (
                            temp[variable]
                            .value_counts()
                            .head(MAX_CATEGORIES)
                            .index
                        )

                        temp = temp[
                            temp[variable].isin(top_categories)
                        ]

                        counts = pd.crosstab(
                            temp[variable],
                            temp["Renovación"]
                        )

                        percentages = pd.crosstab(
                            temp[variable],
                            temp["Renovación"],
                            normalize="index"
                        ).mul(100)

                        for label in ["Renovó", "No renovó"]:
                            if label not in counts.columns:
                                counts[label] = 0
                            if label not in percentages.columns:
                                percentages[label] = 0.0

                        counts = counts[["Renovó", "No renovó"]]
                        percentages = percentages[
                            ["Renovó", "No renovó"]
                        ].round(2)

                        left, right = st.columns(2)

                        with left:
                            st.markdown("**Frecuencias**")
                            st.dataframe(
                                counts,
                                use_container_width=True
                            )

                        with right:
                            st.markdown("**Porcentaje por categoría**")
                            st.dataframe(
                                percentages,
                                use_container_width=True
                            )

                        fig, ax = plt.subplots(figsize=(9, 4))

                        percentages.plot(
                            kind="bar",
                            stacked=True,
                            ax=ax
                        )

                        ax.set_title(f"Renovación por {variable}")
                        ax.set_ylabel("Porcentaje (%)")
                        ax.set_xlabel(variable)
                        ax.set_ylim(0, 100)
                        ax.tick_params(axis="x", rotation=30)
                        ax.legend(
                            title="Renovación",
                            bbox_to_anchor=(1.02, 1),
                            loc="upper left"
                        )

                        show_plot(fig)

                        st.caption(
                            "Se muestran hasta las 20 categorías más "
                            "frecuentes; los porcentajes se calculan "
                            "dentro de cada categoría."
                        )

        # -------------------------------------------------
        # TAB 9. ANÁLISIS DINÁMICO
        # -------------------------------------------------
        with tabs[8]:
            st.markdown("### Análisis dinámico de datos")

            columns = st.multiselect(
                "Columnas a incluir",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:min(5, len(df.columns))],
                key="tab9_columns"
            )

            if not columns:
                st.warning("Selecciona al menos una columna.")

            else:
                filtered = df[columns].copy()

                apply_filter = st.checkbox(
                    "Aplicar filtro por categoría",
                    key="tab9_apply_filter"
                )

                if apply_filter:
                    options = [
                        col for col in columns
                        if col in get_categorical_candidates(df)
                    ]

                    if not options:
                        st.warning(
                            "Selecciona al menos una columna categórica "
                            "para poder filtrar."
                        )

                    else:
                        variable = st.selectbox(
                            "Filtrar por",
                            options,
                            key="tab9_filter_variable"
                        )

                        series = (
                            df[variable]
                            .astype("string")
                            .fillna("(Sin dato)")
                        )

                        available_values = (
                            series.value_counts()
                            .head(100)
                            .index.tolist()
                        )

                        selected = st.multiselect(
                            "Valores incluidos",
                            options=available_values,
                            default=available_values[:min(10, len(available_values))],
                            key="tab9_filter_values"
                        )

                        if selected:
                            mask = series.isin(selected)
                            filtered = filtered.loc[mask]
                        else:
                            filtered = filtered.iloc[0:0]

                        st.caption(
                            "Se ofrecen hasta 100 categorías, "
                            "priorizando las más frecuentes."
                        )

                col1, col2 = st.columns(2)
                col1.metric("Registros filtrados", f"{len(filtered):,}")
                col2.metric("Columnas seleccionadas", len(columns))

                st.dataframe(
                    filtered.head(MAX_TABLE_ROWS),
                    use_container_width=True,
                    hide_index=True
                )

                if len(filtered) > MAX_TABLE_ROWS:
                    st.caption(
                        f"Se muestran los primeros {MAX_TABLE_ROWS:,} "
                        "registros; la descarga incluye todos los filtrados."
                    )

                download_csv(
                    filtered,
                    "insurancecompany_seleccion.csv",
                    "tab9_download"
                )

                selected_numeric = [
                    col for col in columns
                    if col in numeric
                ]

                if selected_numeric and not filtered.empty:
                    st.markdown("#### Resumen estadístico")

                    st.dataframe(
                        filtered[selected_numeric]
                        .describe()
                        .T.round(2),
                        use_container_width=True
                    )

        # -------------------------------------------------
        # TAB 10. HALLAZGOS
        # -------------------------------------------------
        with tabs[9]:
            st.markdown("### Hallazgos clave")

            total_cells = df.size
            missing_cells = int(df.isna().sum().sum())

            missing_pct = (
                missing_cells / total_cells * 100
                if total_cells else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Registros", f"{len(df):,}")
            col2.metric("Variables", f"{df.shape[1]:,}")
            col3.metric("Celdas faltantes", f"{missing_pct:.2f}%")

            findings = []

            # Calidad de datos
            missing = missing_summary(df)

            if missing_cells:
                top_missing = missing.iloc[0]
                findings.append(
                    f"La variable con más faltantes es "
                    f"**{top_missing['Variable']}**, con "
                    f"**{top_missing['% faltantes']:.2f}%**."
                )
            else:
                findings.append("No se detectaron valores faltantes.")

            findings.append(
                f"El dataset contiene **{len(numeric)} variables "
                f"numéricas** y **{len(categorical)} variables "
                "categóricas o de texto**."
            )

            if "id" in df.columns:
                duplicates = int(df["id"].duplicated().sum())
                findings.append(
                    f"Se encontraron **{duplicates:,} identificadores "
                    "repetidos** en `id`; valida si debe ser única."
                )

            # Resumen de renovación
            st.markdown("#### Proporción de renovación")

            if "renewal" not in df.columns:
                st.info("No se encontró la columna `renewal`.")

            else:
                renewal = pd.to_numeric(
                    df["renewal"],
                    errors="coerce"
                )

                valid = renewal.isin([0, 1])
                valid_count = int(valid.sum())

                if valid_count == 0:
                    st.warning(
                        "No hay valores válidos 0/1 para analizar."
                    )

                else:
                    counts = (
                        renewal[valid]
                        .map({1: "Renovó", 0: "No renovó"})
                        .value_counts()
                        .reindex(
                            ["Renovó", "No renovó"],
                            fill_value=0
                        )
                    )

                    percentages = counts / valid_count * 100

                    left, right = st.columns([1.2, 1])

                    with left:
                        fig, ax = plt.subplots(figsize=(6, 3.5))

                        bars = ax.bar(
                            counts.index,
                            percentages.values
                        )

                        ax.set_title("Proporción de renovación")
                        ax.set_ylabel("Clientes (%)")
                        ax.set_ylim(
                            0,
                            max(10, float(percentages.max()) * 1.2)
                        )

                        ax.bar_label(
                            bars,
                            labels=[
                                f"{value:.1f}%"
                                for value in percentages.values
                            ],
                            padding=3
                        )

                        show_plot(fig)

                    with right:
                        st.dataframe(
                            pd.DataFrame({
                                "Resultado": counts.index,
                                "Clientes": counts.values,
                                "Proporción (%)": percentages.round(2).values
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                    findings.append(
                        f"El **{percentages['Renovó']:.2f}%** de los "
                        "registros válidos renovó su póliza, frente al "
                        f"**{percentages['No renovó']:.2f}%** que no renovó."
                    )

                    if len(df) > valid_count:
                        st.caption(
                            f"Se excluyeron {len(df) - valid_count:,} "
                            "registros con valores inválidos de `renewal`."
                        )

            st.markdown("#### Principales hallazgos")

            for i, finding in enumerate(findings, start=1):
                st.markdown(f"{i}. {finding}")

            st.caption(
                "Los hallazgos son descriptivos y no demuestran causalidad."
            )

# =========================================================
# 9. CONCLUSIONES
# =========================================================
elif pagina == "Conclusiones":
    st.subheader("Conclusiones finales")

    if df is None:
        st.warning(
            "Carga un CSV válido para generar conclusiones."
        )

    else:
        numeric, categorical = get_column_types(df)

        missing = int(df.isna().sum().sum())
        cells = df.size

        missing_pct = (
            missing / cells * 100
            if cells else 0
        )

        conclusions = [
            (
                f"**Tamaño:** el dataset contiene {len(df):,} "
                f"registros y {df.shape[1]} variables."
            ),
            (
                f"**Calidad:** hay {missing:,} celdas faltantes "
                f"({missing_pct:.2f}%)."
                if missing else
                "**Calidad:** no se detectaron valores faltantes."
            ),
            (
                f"**Tipos:** se identificaron {len(numeric)} "
                f"variables numéricas y {len(categorical)} "
                "categóricas/texto."
            )
        ]

        if "renewal" in df.columns:
            renewal = pd.to_numeric(
                df["renewal"],
                errors="coerce"
            )

            valid = renewal.isin([0, 1])

            if valid.any():
                proportions = (
                    renewal[valid]
                    .value_counts(normalize=True)
                    .mul(100)
                )

                renewed_pct = proportions.get(1, 0.0)
                not_renewed_pct = proportions.get(0, 0.0)

                conclusions.append(
                    f"**Renovación:** el {renewed_pct:.2f}% "
                    "de los registros válidos corresponde a "
                    "renovaciones y el "
                    f"{not_renewed_pct:.2f}% a no renovaciones."
                )

            else:
                conclusions.append(
                    "**Renovación:** no se encontraron valores "
                    "válidos 0/1 para analizar."
                )

        else:
            conclusions.append(
                "**Renovación:** no existe la columna `renewal`."
            )

        if "id" in df.columns:
            conclusions.append(
                f"**Identificadores:** hay "
                f"{df['id'].duplicated().sum():,} valores repetidos "
                "en `id`; revisa si debe ser única."
            )

        else:
            conclusions.append(
                "**Identificadores:** no existe una columna `id`."
            )

        for i, conclusion in enumerate(conclusions[:5], 1):
            st.markdown(
                f"**Conclusión {i}.** {conclusion}"
            )

        st.divider()

        st.markdown("#### Recomendaciones")

        st.markdown("""
        - Validar la calidad y definición de las variables.
        - Comparar porcentajes y tamaños de grupo.
        - Contrastar los resultados con las áreas de negocio.
        - No inferir causalidad a partir de relaciones descriptivas.
        - Documentar los criterios de limpieza antes de realizar
          análisis posteriores.
        """)
