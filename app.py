
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
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="deep")

MAX_FILE_SIZE = 200 * 1024 * 1024
MAX_PLOT_ROWS = 10_000
MAX_TABLE_ROWS = 500
MAX_CATEGORIES = 20

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


# =========================================================
# 2. ESTILOS
# =========================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.4rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================
def prepare_dataset(df):
    """Limpia encabezados y normaliza cadenas vacías como valores nulos."""
    df = df.copy()
    original_columns = df.columns.tolist()

    df.columns = [str(column).strip() for column in df.columns]

    if any(not column for column in df.columns):
        raise ValueError("Existen columnas sin nombre.")

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Hay nombres de columnas duplicados: {duplicates}"
        )

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    notes = []

    if original_columns != df.columns.tolist():
        notes.append(
            "Se eliminaron espacios al inicio o al final de los encabezados."
        )

    missing = int(df.isna().sum().sum())

    if missing:
        notes.append(
            f"Se detectaron {missing:,} celdas faltantes. "
            "No se imputaron automáticamente."
        )

    if "id" in df.columns:
        duplicates = int(df["id"].duplicated().sum())
        if duplicates:
            notes.append(
                f"Se encontraron {duplicates:,} identificadores repetidos."
            )

    return df, notes


def load_dataset(uploaded_file):
    """Carga y valida un archivo CSV."""
    if uploaded_file.size == 0:
        raise ValueError("El archivo está vacío.")

    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError("El archivo supera el límite de 200 MB.")

    uploaded_file.seek(0)

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            encoding="latin-1",
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "El archivo no contiene datos legibles."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Error en la estructura del CSV: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "El CSV no contiene registros. Verifica el archivo y su separador."
        )

    return prepare_dataset(df)


def number(value):
    """Formatea números y valores ausentes."""
    if pd.isna(value):
        return "N/D"
    return f"{value:,.2f}"


def show_plot(fig):
    """Muestra y cierra una figura para liberar memoria."""
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def download_csv(df, filename, key):
    """Ofrece una descarga en formato CSV."""
    st.download_button(
        label="Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def renewal_labels(series):
    """Transforma los valores 0 y 1 en etiquetas comprensibles."""
    values = pd.to_numeric(series, errors="coerce")

    labels = pd.Series(
        "(Sin dato)",
        index=series.index,
        dtype="object",
    )
    labels.loc[values == 1] = "Renovó"
    labels.loc[values == 0] = "No renovó"

    other = values.notna() & ~values.isin([0, 1])
    labels.loc[other] = series.loc[other].astype(str)

    return labels


# =========================================================
# 4. PROGRAMACIÓN ORIENTADA A OBJETOS
# =========================================================
class DataAnalyzer:
    """
    Encapsula operaciones de análisis exploratorio de datos.

    Atributo:
        df: copia del DataFrame que se analizará.

    Métodos principales:
        classify_variables()
        variable_summary()
        descriptive_statistics()
        missing_summary()
        categorical_summary()
        histogram()
        categorical_plot()
        numeric_by_renewal()
        categorical_by_renewal()
        renewal_summary()
    """

    def __init__(self, df):
        """Inicializa el analizador con una copia del dataset."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El objeto recibido debe ser un DataFrame.")

        self.df = df.copy()

    def classify_variables(self):
        """Clasifica las variables según sus tipos de datos."""
        numeric = self.df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        dates = self.df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()

        categorical = [
            column
            for column in self.df.columns
            if column not in numeric and column not in dates
        ]

        # Los identificadores y la variable objetivo se conservan
        # en la clasificación general, pero se excluyen de algunos gráficos.
        return {
            "numeric": numeric,
            "categorical": categorical,
            "dates": dates,
            "numeric_count": len(numeric),
            "categorical_count": len(categorical),
            "date_count": len(dates),
        }

    def numeric_candidates(self):
        """Devuelve variables numéricas útiles para análisis descriptivo."""
        return [
            column
            for column in self.classify_variables()["numeric"]
            if column not in {"id", "renewal"}
        ]

    def categorical_candidates(self):
        """Devuelve variables categóricas, excluyendo ID y renovación."""
        return [
            column
            for column in self.classify_variables()["categorical"]
            if column not in {"id", "renewal"}
        ]

    def variable_summary(self):
        """Crea un resumen de tipos, nulos y valores únicos."""
        rows = []

        for column in self.df.columns:
            series = self.df[column]

            if pd.api.types.is_numeric_dtype(series):
                classification = "Numérica"
            elif pd.api.types.is_datetime64_any_dtype(series):
                classification = "Fecha"
            else:
                classification = "Categórica / texto"

            rows.append({
                "Variable": column,
                "Tipo de dato": str(series.dtype),
                "Clasificación": classification,
                "Valores no nulos": int(series.notna().sum()),
                "Valores nulos": int(series.isna().sum()),
                "Valores únicos": int(series.nunique(dropna=True)),
            })

        return pd.DataFrame(rows)

    def descriptive_statistics(self):
        """Calcula estadísticas descriptivas para variables numéricas."""
        columns = self.classify_variables()["numeric"]

        if not columns:
            return pd.DataFrame()

        numeric_df = self.df[columns].replace(
            [np.inf, -np.inf], np.nan
        )

        stats = numeric_df.describe().T
        stats["Mediana"] = numeric_df.median()
        stats["Moda"] = numeric_df.mode().iloc[0] if not numeric_df.mode().empty else np.nan

        return stats

    def missing_summary(self):
        """Resume la cantidad y el porcentaje de valores faltantes."""
        counts = self.df.isna().sum()
        total_rows = len(self.df)

        percentages = (
            counts / total_rows * 100
            if total_rows
            else counts * 0
        )

        result = pd.DataFrame({
            "Variable": self.df.columns,
            "Valores faltantes": counts.values,
            "Porcentaje faltante (%)": percentages.values,
        })

        return result.sort_values(
            "Valores faltantes",
            ascending=False,
        ).reset_index(drop=True)

    def categorical_summary(self, column):
        """Calcula las frecuencias y proporciones de una variable categórica."""
        if column not in self.df.columns:
            raise ValueError(f"No existe la variable '{column}'.")

        values = self.df[column].astype("string").fillna("(Sin dato)")

        result = (
            values.value_counts()
            .rename_axis("Categoría")
            .reset_index(name="Frecuencia")
        )

        result["Proporción (%)"] = (
            result["Frecuencia"] / len(self.df) * 100
            if len(self.df)
            else 0
        )

        result["Proporción (%)"] = result["Proporción (%)"].round(2)
        return result

    def histogram(self, column):
        """Genera un histograma de una variable numérica."""
        if column not in self.df.columns:
            raise ValueError(f"No existe la variable '{column}'.")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError("La variable seleccionada no es numérica.")

        values = self.df[column].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if values.empty:
            return None

        if len(values) > MAX_PLOT_ROWS:
            values = values.sample(
                n=MAX_PLOT_ROWS,
                random_state=42,
            )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(values, bins="auto", edgecolor="black", alpha=0.8)
        ax.set_title(f"Distribución de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)

        return fig

    def categorical_plot(self, column, top=10):
        """Genera un gráfico de barras de las categorías más frecuentes."""
        summary = self.categorical_summary(column).head(top)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=summary,
            x="Frecuencia",
            y="Categoría",
            ax=ax,
        )
        ax.set_title(f"Principales categorías: {column}")
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Categoría")

        return fig, summary

    def numeric_by_renewal(self, column):
        """Compara una variable numérica entre clientes que renovaron y no renovaron."""
        if "renewal" not in self.df.columns:
            return None, pd.DataFrame()

        if column not in self.df.columns:
            raise ValueError(f"No existe la variable '{column}'.")

        temp = self.df[[column, "renewal"]].copy()
        temp[column] = pd.to_numeric(temp[column], errors="coerce")
        temp[column] = temp[column].replace([np.inf, -np.inf], np.nan)
        temp["Renovación"] = renewal_labels(temp["renewal"])

        temp = temp.dropna(subset=[column])
        temp = temp[
            temp["Renovación"].isin(["Renovó", "No renovó"])
        ]

        if temp.empty:
            return None, pd.DataFrame()

        plot_df = temp

        if len(plot_df) > MAX_PLOT_ROWS:
            plot_df = plot_df.sample(
                n=MAX_PLOT_ROWS,
                random_state=42,
            )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=plot_df,
            x="Renovación",
            y=column,
            ax=ax,
        )
        ax.set_title(f"{column} según renovación")
        ax.set_xlabel("Estado de renovación")
        ax.set_ylabel(column)

        summary = temp.groupby("Renovación")[column].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )

        return fig, summary

    def categorical_by_renewal(self, column, top=MAX_CATEGORIES):
        """Calcula frecuencias y porcentajes de renovación por categoría."""
        if "renewal" not in self.df.columns:
            return pd.DataFrame(), pd.DataFrame(), None

        if column not in self.df.columns:
            raise ValueError(f"No existe la variable '{column}'.")

        temp = self.df[[column, "renewal"]].copy()
        temp["Renovación"] = renewal_labels(temp["renewal"])
        temp = temp[
            temp["Renovación"].isin(["Renovó", "No renovó"])
        ].copy()

        temp[column] = (
            temp[column].astype("string").fillna("(Sin dato)")
        )

        if temp.empty:
            return pd.DataFrame(), pd.DataFrame(), None

        top_values = temp[column].value_counts().head(top).index
        temp = temp[temp[column].isin(top_values)]

        counts = pd.crosstab(temp[column], temp["Renovación"])
        percentages = pd.crosstab(
            temp[column],
            temp["Renovación"],
            normalize="index",
        ) * 100

        expected_labels = ["Renovó", "No renovó"]

        for label in expected_labels:
            if label not in counts.columns:
                counts[label] = 0
            if label not in percentages.columns:
                percentages[label] = 0.0

        counts = counts[expected_labels]
        percentages = percentages[expected_labels].round(2)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        percentages.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Renovación por {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Porcentaje (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Renovación", bbox_to_anchor=(1.02, 1), loc="upper left")

        return counts, percentages, fig

    def renewal_summary(self):
        """Resume los resultados de la variable renovación."""
        if "renewal" not in self.df.columns:
            return pd.DataFrame()

        values = pd.to_numeric(self.df["renewal"], errors="coerce")
        values = values[values.isin([0, 1])]

        if values.empty:
            return pd.DataFrame()

        counts = (
            values.map({1: "Renovó", 0: "No renovó"})
            .value_counts()
            .reindex(["Renovó", "No renovó"], fill_value=0)
        )

        total = int(counts.sum())

        return pd.DataFrame({
            "Resultado": counts.index,
            "Clientes": counts.values,
            "Proporción (%)": (
                counts.values / total * 100 if total else 0
            ),
        }).round(2)


# =========================================================
# 5. ENCABEZADO Y SIDEBAR
# =========================================================
st.title("📊 InsuranceCompany | EDA")
st.caption("Caso de Estudio N.° 3 · 2026")

with st.sidebar:
    st.subheader("Python for Analytics")

    try:
        st.image("LogoKRMC-rectangular.png", width=225)
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
            "Conclusiones",
        ],
        key="menu_navegacion",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Carga archivo para análisis",
        type=["csv"],
        help="Archivo CSV de hasta 200 MB.",
    )

    st.divider()
    st.caption("EDA · Aplicación académica")


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
# 7. INICIO
# =========================================================
if pagina == "Inicio":
    left, right = st.columns(2)

    with left:
        st.markdown("#### Objetivo")
        st.write(
            "Analizar la estructura, calidad y distribución de los datos "
            "para identificar patrones relacionados con la renovación de "
            "pólizas. No se construyen modelos predictivos."
        )

        st.markdown("#### Dataset")
        st.write(
            "El dataset contiene información histórica de clientes de una "
            "compañía de seguros. Incluye variables demográficas, económicas, "
            "historial de pagos, morosidad, canal de captación, tipo de "
            "residencia, valor de la prima y renovación."
        )

    with right:
        st.markdown("#### Ficha del proyecto")
        st.write("**Autora:**")
        st.write("Katia Roxana Mendez Cortez")
        st.write("**Programa:**")
        st.write("Especialización en Python for Analytics")
        st.write("**Año:** 2026")
        st.write("**Tecnologías:**")
        st.write("Python, Streamlit, Pandas, NumPy, Matplotlib y Seaborn")

    if df is None:
        st.info("Para comenzar, carga un archivo CSV válido desde el panel lateral.")
    else:
        st.success("Dataset cargado correctamente.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Variables", f"{df.shape[1]:,}")
        c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")


# =========================================================
# 8. CARGA DEL DATASET
# =========================================================
elif pagina == "Carga del dataset":
    st.subheader("Carga y validación del dataset")

    if analyzer is None:
        st.warning("Carga un archivo CSV válido para continuar.")
    else:
        for note in notes:
            st.warning(note)

        missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
        if missing_columns:
            st.warning(
                "Faltan columnas del esquema esperado: "
                + ", ".join(missing_columns)
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Variables", f"{df.shape[1]:,}")
        c3.metric("Valores faltantes", f"{int(df.isna().sum().sum()):,}")

        st.markdown("#### Vista previa")
        rows = st.slider(
            "Filas para mostrar",
            min_value=1,
            max_value=min(50, len(df)),
            value=min(10, len(df)),
            key="preview_rows",
        )

        st.dataframe(df.head(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Estructura de columnas")
        st.dataframe(
            analyzer.variable_summary(),
            use_container_width=True,
            hide_index=True,
        )

        download_csv(df, "InsuranceCompany_validado.csv", "download_clean")


# =========================================================
# 9. ANÁLISIS EXPLORATORIO DE DATOS (10 PESTAÑAS)
# =========================================================
elif pagina == "Análisis exploratorio (EDA)":
    st.subheader("Análisis exploratorio de datos")
    st.caption(
        "Selecciona una pestaña para consultar cada uno de los diez análisis."
    )

    if analyzer is None:
        st.warning("Carga un CSV válido para habilitar el análisis.")
    else:
        types = analyzer.classify_variables()
        numeric = types["numeric"]
        categorical = analyzer.categorical_candidates()
        numeric_candidates = analyzer.numeric_candidates()

        tabs = st.tabs([
            "1. Información",
            "2. Variables",
            "3. Descriptiva",
            "4. Faltantes",
            "5. Distribuciones",
            "6. Categóricas",
            "7. Numérica vs renovación",
            "8. Categórica vs renovación",
            "9. Análisis dinámico",
            "10. Hallazgos",
        ])

        # -------------------------------------------------
        # TAB 1. INFORMACIÓN
        # -------------------------------------------------
        with tabs[0]:
            st.markdown("### Información general")
            st.write(
                "Este dataset contiene información histórica de clientes "
                "de una compañía de seguros, incluyendo variables económicas, "
                "historial de pagos y renovación de pólizas."
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Registros", f"{len(df):,}")
            c2.metric("Variables", f"{df.shape[1]:,}")
            c3.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")

            left, right = st.columns(2)

            with left:
                st.markdown("#### Tipos de datos")
                st.dataframe(
                    df.dtypes.astype(str).rename("Tipo").reset_index().rename(
                        columns={"index": "Variable"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with right:
                st.markdown("#### Valores faltantes")
                st.dataframe(
                    analyzer.missing_summary(),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Ver información técnica"):
                buffer = StringIO()
                df.info(buf=buffer)
                st.code(buffer.getvalue())

        # -------------------------------------------------
        # TAB 2. TIPOS DE VARIABLES
        # -------------------------------------------------
        with tabs[1]:
            st.markdown("### Clasificación de variables")
            st.write(
                "La clase `DataAnalyzer` identifica los tipos de datos y "
                "resume los valores nulos y únicos de cada columna."
            )

            summary_types = analyzer.variable_summary()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total de variables", len(summary_types))
            c2.metric(
                "Variables numéricas",
                int((summary_types["Clasificación"] == "Numérica").sum()),
            )
            c3.metric(
                "Variables categóricas",
                int((summary_types["Clasificación"] == "Categórica / texto").sum()),
            )

            st.dataframe(
                summary_types,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Descargar clasificación",
                data=summary_types.to_csv(index=False).encode("utf-8-sig"),
                file_name="clasificacion_variables.csv",
                mime="text/csv",
                key="download_types",
            )

        # -------------------------------------------------
        # TAB 3. ESTADÍSTICA DESCRIPTIVA
        # -------------------------------------------------
        with tabs[2]:
            st.markdown("### Estadísticas descriptivas")
            stats = analyzer.descriptive_statistics()

            if stats.empty:
                st.info("No hay variables numéricas disponibles.")
            else:
                st.dataframe(
                    stats.round(3),
                    use_container_width=True,
                )

                if numeric_candidates:
                    variable = st.selectbox(
                        "Variable para interpretar",
                        numeric_candidates,
                        key="tab3_variable",
                    )

                    values = pd.to_numeric(
                        df[variable], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    if values.empty:
                        st.warning("No hay valores válidos para interpretar.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Media", number(values.mean()))
                        c2.metric("Mediana", number(values.median()))
                        c3.metric("Desviación estándar", number(values.std()))

                        if values.mean() > values.median():
                            st.write(
                                "La media supera la mediana, lo que podría "
                                "ser consistente con asimetría hacia valores altos."
                            )
                        elif values.mean() < values.median():
                            st.write(
                                "La media es inferior a la mediana, lo que podría "
                                "ser consistente con asimetría hacia valores bajos."
                            )
                        else:
                            st.write(
                                "La media y la mediana coinciden o son muy cercanas."
                            )

                        st.caption(
                            "La comparación entre media y mediana es una orientación; "
                            "conviene revisar también la forma del histograma."
                        )
                else:
                    st.info(
                        "No hay variables numéricas distintas de `id` y `renewal` "
                        "para interpretar."
                    )

        # -------------------------------------------------
        # TAB 4. VALORES FALTANTES
        # -------------------------------------------------
        with tabs[3]:
            st.markdown("### Análisis de valores faltantes")
            missing = analyzer.missing_summary()
            total_missing = int(df.isna().sum().sum())
            total_cells = df.size
            missing_pct = total_missing / total_cells * 100 if total_cells else 0
            columns_with_missing = int((df.isna().sum() > 0).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Valores faltantes", f"{total_missing:,}")
            c2.metric("Variables afectadas", f"{columns_with_missing:,}")
            c3.metric("Porcentaje de celdas", f"{missing_pct:.2f}%")

            st.dataframe(
                missing,
                use_container_width=True,
                hide_index=True,
            )

            chart_data = missing[
                missing["Valores faltantes"] > 0
            ].sort_values("Valores faltantes")

            if chart_data.empty:
                st.success("No se encontraron valores faltantes.")
            else:
                fig, ax = plt.subplots(
                    figsize=(9, max(3, len(chart_data) * 0.35))
                )
                ax.barh(
                    chart_data["Variable"],
                    chart_data["Valores faltantes"],
                )
                ax.set_title("Valores faltantes por variable")
                ax.set_xlabel("Cantidad de valores faltantes")
                ax.set_ylabel("Variable")
                show_plot(fig)

                top_missing = missing[missing["Valores faltantes"] > 0].head(3)
                affected_names = ", ".join(top_missing["Variable"].astype(str))

                st.markdown("#### Interpretación")
                st.write(
                    f"El dataset presenta {total_missing:,} celdas faltantes, "
                    f"equivalentes al {missing_pct:.2f}% del total. "
                    f"Las variables más afectadas incluyen: {affected_names}."
                )
                st.write(
                    "Antes de eliminar registros o imputar valores, conviene "
                    "evaluar la proporción de faltantes y su posible impacto."
                )

        # -------------------------------------------------
        # TAB 5. DISTRIBUCIONES NUMÉRICAS
        # -------------------------------------------------
        with tabs[4]:
            st.markdown("### Distribución de variables numéricas")
            st.write(
                "Los histogramas ayudan a identificar concentraciones, "
                "asimetrías y posibles valores extremos."
            )

            if not numeric:
                st.info("No se encontraron variables numéricas.")
            else:
                histogram_candidates = [
                    col for col in numeric if col != "id"
                ]
                if not histogram_candidates:
                    histogram_candidates = numeric

                variable = st.selectbox(
                    "Selecciona una variable numérica",
                    histogram_candidates,
                    key="hist_variable",
                )

                fig = analyzer.histogram(variable)

                if fig is None:
                    st.warning("La variable no contiene valores válidos para graficar.")
                else:
                    show_plot(fig)

                    values = pd.to_numeric(
                        df[variable], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Observaciones válidas", f"{len(values):,}")
                    c2.metric("Media", number(values.mean()))
                    c3.metric("Mediana", number(values.median()))

                    st.markdown("#### Interpretación visual")
                    if len(values) < 2:
                        st.info(
                            "Hay menos de dos observaciones válidas; "
                            "interpreta el histograma con cautela."
                        )
                    elif values.mean() > values.median():
                        st.write(
                            "La media es mayor que la mediana; podría existir "
                            "asimetría hacia valores altos."
                        )
                    elif values.mean() < values.median():
                        st.write(
                            "La media es menor que la mediana; podría existir "
                            "asimetría hacia valores bajos."
                        )
                    else:
                        st.write(
                            "La media y la mediana son similares. Revisa el "
                            "histograma para comprender la forma de la distribución."
                        )

                    st.write(
                        f"Los valores válidos van de {values.min():,.2f} "
                        f"a {values.max():,.2f}. La desviación estándar es "
                        f"{number(values.std())}."
                    )

        # -------------------------------------------------
        # TAB 6. VARIABLES CATEGÓRICAS
        # -------------------------------------------------
        with tabs[5]:
            st.markdown("### Análisis de variables categóricas")

            if not categorical:
                st.info("No hay variables categóricas disponibles.")
            else:
                left, right = st.columns([2, 1])

                with left:
                    variable = st.selectbox(
                        "Variable categórica",
                        categorical,
                        key="tab6_variable",
                    )

                with right:
                    top = st.slider(
                        "Categorías a mostrar",
                        min_value=3,
                        max_value=MAX_CATEGORIES,
                        value=10,
                        key="tab6_top",
                    )

                fig, summary = analyzer.categorical_plot(variable, top=top)
                col1, col2 = st.columns([1, 1.3])

                with col1:
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True,
                    )

                with col2:
                    show_plot(fig)

                st.caption(
                    f"Categorías únicas: {df[variable].nunique(dropna=True):,}. "
                    f"Se muestran las {len(summary)} más frecuentes."
                )

        # -------------------------------------------------
        # TAB 7. VARIABLE NUMÉRICA VS RENOVACIÓN
        # -------------------------------------------------
        with tabs[6]:
            st.markdown("### Variables numéricas según renovación")

            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")
            elif not numeric_candidates:
                st.info("No hay variables numéricas disponibles para comparar.")
            else:
                variable = st.selectbox(
                    "Variable numérica",
                    numeric_candidates,
                    key="tab7_variable",
                )

                fig, summary = analyzer.numeric_by_renewal(variable)

                if fig is None:
                    st.warning("No hay datos válidos para comparar.")
                else:
                    show_plot(fig)
                    st.markdown("**Estadísticas de todos los registros válidos**")
                    st.dataframe(
                        summary.round(2),
                        use_container_width=True,
                    )
                    st.info(
                        "Las diferencias observadas son descriptivas y "
                        "no demuestran causalidad."
                    )

        # -------------------------------------------------
        # TAB 8. VARIABLE CATEGÓRICA VS RENOVACIÓN
        # -------------------------------------------------
        with tabs[7]:
            st.markdown("### Renovación por categoría")

            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")
            elif not categorical:
                st.info("No hay variables categóricas para comparar.")
            else:
                variable = st.selectbox(
                    "Variable categórica",
                    categorical,
                    key="tab8_variable",
                )

                counts, percentages, fig = analyzer.categorical_by_renewal(
                    variable,
                    top=MAX_CATEGORIES,
                )

                if fig is None:
                    st.warning("No hay datos válidos para comparar.")
                else:
                    left, right = st.columns(2)

                    with left:
                        st.markdown("**Frecuencias**")
                        st.dataframe(
                            counts,
                            use_container_width=True,
                        )

                    with right:
                        st.markdown("**Porcentaje dentro de cada categoría**")
                        st.dataframe(
                            percentages,
                            use_container_width=True,
                        )

                    show_plot(fig)
                    st.caption(
                        "Se muestran hasta las 20 categorías más frecuentes. "
                        "Los porcentajes se calculan dentro de cada categoría."
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
                key="tab9_columns",
            )

            if not columns:
                st.warning("Selecciona al menos una columna.")
            else:
                filtered = df[columns].copy()

                apply_filter = st.checkbox(
                    "Aplicar filtro por categoría",
                    key="tab9_apply_filter",
                )

                if apply_filter:
                    filter_options = [
                        col for col in columns
                        if col in categorical
                    ]

                    if not filter_options:
                        st.warning(
                            "Incluye al menos una columna categórica "
                            "para poder filtrar."
                        )
                    else:
                        variable = st.selectbox(
                            "Filtrar por",
                            filter_options,
                            key="tab9_filter_variable",
                        )

                        series = (
                            df[variable].astype("string").fillna("(Sin dato)")
                        )
                        available = series.value_counts().head(100).index.tolist()

                        selected = st.multiselect(
                            "Valores incluidos",
                            options=available,
                            default=available[:min(10, len(available))],
                            key="tab9_filter_values",
                        )

                        filtered = filtered.loc[series.isin(selected)]

                        st.caption(
                            "Se ofrecen hasta 100 categorías, priorizando "
                            "las más frecuentes."
                        )

                c1, c2 = st.columns(2)
                c1.metric("Registros filtrados", f"{len(filtered):,}")
                c2.metric("Columnas seleccionadas", len(columns))

                st.dataframe(
                    filtered.head(MAX_TABLE_ROWS),
                    use_container_width=True,
                    hide_index=True,
                )

                if len(filtered) > MAX_TABLE_ROWS:
                    st.caption(
                        f"Se muestran los primeros {MAX_TABLE_ROWS:,} registros; "
                        "la descarga incluye todos los registros filtrados."
                    )

                download_csv(
                    filtered,
                    "insurancecompany_seleccion.csv",
                    "tab9_download",
                )

                selected_numeric = [
                    col for col in columns if col in numeric
                ]

                if selected_numeric and not filtered.empty:
                    st.markdown("#### Resumen estadístico")
                    st.dataframe(
                        filtered[selected_numeric]
                        .replace([np.inf, -np.inf], np.nan)
                        .describe()
                        .T.round(2),
                        use_container_width=True,
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

            c1, c2, c3 = st.columns(3)
            c1.metric("Registros", f"{len(df):,}")
            c2.metric("Variables", f"{df.shape[1]:,}")
            c3.metric("Celdas faltantes", f"{missing_pct:.2f}%")

            findings = []

            missing = analyzer.missing_summary()
            if missing_cells:
                top_missing = missing.iloc[0]
                findings.append(
                    f"La variable con más faltantes es "
                    f"**{top_missing['Variable']}**, con "
                    f"**{top_missing['Porcentaje faltante (%)']:.2f}%**."
                )
            else:
                findings.append("No se detectaron valores faltantes.")

            findings.append(
                f"El dataset contiene **{len(types['numeric'])} variables "
                f"numéricas** y **{len(types['categorical'])} variables "
                "categóricas o de texto**."
            )

            if "id" in df.columns:
                duplicates = int(df["id"].duplicated().sum())
                findings.append(
                    f"Se encontraron **{duplicates:,} identificadores "
                    "repetidos** en `id`; valida si este campo debe ser único."
                )

            st.markdown("#### Proporción de renovación")

            renewal_table = analyzer.renewal_summary()

            if renewal_table.empty:
                st.info(
                    "No se encontró la columna `renewal` o no hay valores "
                    "válidos 0/1 para analizar."
                )
            else:
                left, right = st.columns([1.2, 1])

                with left:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    bars = ax.bar(
                        renewal_table["Resultado"],
                        renewal_table["Proporción (%)"],
                    )
                    ax.set_title("Proporción de renovación")
                    ax.set_ylabel("Clientes (%)")
                    ax.set_ylim(
                        0,
                        max(
                            10,
                            float(renewal_table["Proporción (%)"].max()) * 1.2,
                        ),
                    )
                    ax.bar_label(
                        bars,
                        labels=[
                            f"{value:.1f}%"
                            for value in renewal_table["Proporción (%)"]
                        ],
                        padding=3,
                    )
                    show_plot(fig)

                with right:
                    st.dataframe(
                        renewal_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                renewed_pct = float(
                    renewal_table.loc[
                        renewal_table["Resultado"] == "Renovó",
                        "Proporción (%)",
                    ].iloc[0]
                )
                not_renewed_pct = float(
                    renewal_table.loc[
                        renewal_table["Resultado"] == "No renovó",
                        "Proporción (%)",
                    ].iloc[0]
                )

                findings.append(
                    f"El **{renewed_pct:.2f}%** de los registros válidos "
                    f"renovó su póliza, frente al **{not_renewed_pct:.2f}%** "
                    "que no renovó."
                )

                valid_count = int(
                    pd.to_numeric(df["renewal"], errors="coerce")
                    .isin([0, 1]).sum()
                )
                if valid_count < len(df):
                    st.caption(
                        f"Se excluyeron {len(df) - valid_count:,} registros "
                        "con valores inválidos o faltantes en `renewal`."
                    )

            st.markdown("#### Principales hallazgos")
            for index, finding in enumerate(findings, start=1):
                st.markdown(f"{index}. {finding}")

            st.caption(
                "Los resultados son descriptivos y no demuestran causalidad."
            )


# =========================================================
# 10. CONCLUSIONES
# =========================================================
elif pagina == "Conclusiones":
    st.subheader("Conclusiones finales")

    if analyzer is None:
        st.warning("Carga un CSV válido para generar conclusiones.")
    else:
        types = analyzer.classify_variables()
        missing = int(df.isna().sum().sum())
        total_cells = df.size
        missing_pct = missing / total_cells * 100 if total_cells else 0

        conclusions = [
            (
                f"**Tamaño:** el dataset contiene {len(df):,} registros "
                f"y {df.shape[1]} variables."
            ),
            (
                f"**Calidad:** hay {missing:,} celdas faltantes "
                f"({missing_pct:.2f}%)."
                if missing
                else "**Calidad:** no se detectaron valores faltantes."
            ),
            (
                f"**Tipos:** se identificaron {types['numeric_count']} "
                f"variables numéricas, {types['categorical_count']} "
                "categóricas/texto y "
                f"{types['date_count']} de fecha."
            ),
        ]

        renewal_table = analyzer.renewal_summary()

        if not renewal_table.empty:
            renewed_pct = float(
                renewal_table.loc[
                    renewal_table["Resultado"] == "Renovó",
                    "Proporción (%)",
                ].iloc[0]
            )
            not_renewed_pct = float(
                renewal_table.loc[
                    renewal_table["Resultado"] == "No renovó",
                    "Proporción (%)",
                ].iloc[0]
            )
            conclusions.append(
                f"**Renovación:** el {renewed_pct:.2f}% de los registros "
                f"válidos corresponde a renovaciones y el {not_renewed_pct:.2f}% "
                "a no renovaciones."
            )
        elif "renewal" not in df.columns:
            conclusions.append(
                "**Renovación:** no existe la columna `renewal`."
            )
        else:
            conclusions.append(
                "**Renovación:** no se encontraron valores válidos 0/1."
            )

        if "id" in df.columns:
            conclusions.append(
                f"**Identificadores:** hay {df['id'].duplicated().sum():,} "
                "valores repetidos en `id`; revisa si debe ser única."
            )

        for index, conclusion in enumerate(conclusions, start=1):
            st.markdown(f"**Conclusión {index}.** {conclusion}")

        st.markdown("#### Consideraciones")
        st.write(
            "Los hallazgos permiten comprender la estructura, calidad y "
            "comportamiento de los datos. Antes de construir modelos "
            "predictivos, sería conveniente revisar los valores faltantes, "
            "validar la consistencia de las variables y analizar las "
            "diferencias observadas entre clientes que renovaron y los que no."
        )
