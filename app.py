"""
InsuranceCompany | Exploración de Datos (EDA)
Caso de Estudio N.° 3 – Python for Analytics

Ejecución:
    pip install -r requirements.txt
    streamlit run app.py
"""

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
    layout="wide"
)

sns.set_theme(style="whitegrid", palette="deep")

MAX_FILE_SIZE = 200 * 1024 * 1024
EXPECTED_COLUMNS = {
    "id", "perc_premium_paid_by_cash_credit", "age_in_days",
    "Income", "Count_3-6_months_late", "Count_6-12_months_late",
    "Count_more_than_12_months_late", "application_underwriting_score",
    "no_of_premiums_paid", "sourcing_channel", "residence_area_type",
    "premium", "renewal"
}


# =========================================================
# 2. FUNCIONES AUXILIARES
# =========================================================
def clean_dataframe(df):
    """Valida encabezados y normaliza valores faltantes."""
    df = df.copy()
    original_columns = df.columns.tolist()
    df.columns = [str(c).strip() for c in df.columns]

    if any(not c for c in df.columns):
        raise ValueError("Existen columnas sin nombre.")

    if df.columns.duplicated().any():
        raise ValueError("Existen nombres de columnas duplicados.")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    notes = []

    if original_columns != df.columns.tolist():
        notes.append("Se eliminaron espacios de los encabezados.")

    if df.isna().any().any():
        notes.append("Hay valores faltantes; no se imputaron automáticamente.")

    if "id" in df.columns:
        duplicates = int(df["id"].duplicated().sum())
        if duplicates:
            notes.append(f"Se encontraron {duplicates:,} identificadores repetidos.")

    return df, notes


def load_dataset(file):
    """Carga y valida el CSV."""
    if file.size == 0:
        raise ValueError("El archivo está vacío.")

    if file.size > MAX_FILE_SIZE:
        raise ValueError("El archivo supera el límite de 200 MB.")

    try:
        file.seek(0)
        df = pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="latin-1", low_memory=False)

    if df.empty:
        raise ValueError("El CSV no contiene registros.")

    return clean_dataframe(df)


def missing_summary(df):
    """Resume los valores faltantes."""
    result = pd.DataFrame({
        "Variable": df.columns,
        "Valores faltantes": df.isna().sum().values
    })

    result["% faltantes"] = (
        result["Valores faltantes"] / len(df) * 100
    ).round(2)

    return result.sort_values(
        "Valores faltantes", ascending=False
    ).reset_index(drop=True)


def categorical_summary(df, column):
    """Calcula frecuencias y porcentajes por categoría."""
    values = df[column].astype("string").fillna("(Sin dato)")
    result = values.value_counts().rename_axis("Categoría").reset_index(
        name="Frecuencia"
    )
    result["Proporción (%)"] = (
        result["Frecuencia"] / len(df) * 100
    ).round(2)

    return result


def renewal_labels(series):
    """Presenta la variable renewal con etiquetas legibles."""
    labels = {1: "Renovó (1)", 0: "No renovó (0)",
              "1": "Renovó (1)", "0": "No renovó (0)"}

    return series.map(labels).fillna(
        series.astype("string").fillna("(Sin dato)")
    )


def number(value):
    """Formatea números y valores ausentes."""
    return "N/D" if pd.isna(value) else f"{value:,.2f}"


def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def download_csv(df, filename, key):
    st.download_button(
        "Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


# =========================================================
# 3. ENCABEZADO Y NAVEGACIÓN
# =========================================================
st.title("📊 InsuranceCompany | Análisis Exploratorio de Datos")
st.caption("Caso de Estudio N.° 3 · Python for Analytics · 2026")

with st.sidebar:
    st.header("Navegación")
    module = st.radio(
        "Selecciona un módulo",
        ["Inicio", "Carga del dataset",
         "Análisis exploratorio (EDA)", "Conclusiones"]
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "Carga InsuranceCompany.csv",
        type=["csv"],
        help="Se admite un archivo CSV de hasta 200 MB."
    )

    st.caption("Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn")


# =========================================================
# 4. CARGA Y VALIDACIÓN
# =========================================================
df = None
load_error = None
notes = []

if uploaded_file is not None:
    try:
        df, notes = load_dataset(uploaded_file)
    except (ValueError, pd.errors.ParserError,
            pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        load_error = str(exc)
    except Exception:
        load_error = "No se pudo procesar el archivo. Verifica su formato."

if load_error:
    st.error(f"Error al cargar el dataset: {load_error}")

if df is not None:
    for note in notes:
        st.warning(note)

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing_columns:
        st.warning(
            "Faltan columnas del esquema esperado: "
            + ", ".join(missing_columns)
        )


# =========================================================
# 5. INICIO
# =========================================================
if module == "Inicio":
    st.subheader("Bienvenido")

    st.write(
        "Explora los datos históricos de clientes de una compañía de "
        "seguros mediante estadísticas descriptivas y visualizaciones."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Objetivo")
        st.write(
            "Analizar la estructura, calidad y distribución de los datos "
            "para identificar patrones relacionados con la renovación "
            "de pólizas. No se construyen modelos predictivos."
        )

        st.markdown("#### Variables")
        st.write(
            "Edad, ingresos, primas, pagos tardíos, canal de captación, "
            "tipo de residencia y renovación."
        )

    with right:
        st.markdown("#### Ficha del proyecto")
        st.write("**Autora:** Katia Roxana Mendez Cortez")
        st.write("**Programa:** Especialización en Python for Analytics")
        st.write("**Año:** 2026")
        st.write("**Tecnologías:** Python, Streamlit, Pandas, NumPy y Seaborn")

    if df is None:
        st.info("Carga un CSV válido desde el panel lateral.")
    else:
        a, b, c = st.columns(3)
        a.metric("Registros", f"{len(df):,}")
        b.metric("Variables", f"{df.shape[1]:,}")
        c.metric("Celdas faltantes", f"{df.isna().sum().sum():,}")


# =========================================================
# 6. CARGA DEL DATASET
# =========================================================
elif module == "Carga del dataset":
    st.subheader("Carga y validación del dataset")

    if df is None:
        st.warning("Carga un archivo CSV válido para continuar.")
    else:
        a, b, c = st.columns(3)
        a.metric("Registros", f"{len(df):,}")
        b.metric("Variables", f"{df.shape[1]:,}")
        c.metric("Valores faltantes", f"{df.isna().sum().sum():,}")

        st.markdown("#### Vista previa")

        rows = st.slider(
            "Filas para mostrar",
            min_value=1,
            max_value=min(50, len(df)),
            value=min(10, len(df))
        )
        st.dataframe(df.head(rows), use_container_width=True)

        schema = pd.DataFrame({
            "Variable": df.columns,
            "Tipo de dato": df.dtypes.astype(str).values,
            "No nulos": df.notna().sum().values,
            "Nulos": df.isna().sum().values,
            "Valores únicos": df.nunique().values
        })

        st.markdown("#### Estructura de columnas")
        st.dataframe(schema, use_container_width=True, hide_index=True)

        download_csv(df, "InsuranceCompany_validado.csv", "download_clean")


# =========================================================
# 7. ANÁLISIS EXPLORATORIO (10 APARTADOS)
# =========================================================
elif module == "Análisis exploratorio (EDA)":
    st.subheader("Análisis exploratorio de datos")

    if df is None:
        st.warning("Carga un CSV válido para habilitar el análisis.")
    else:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            or c in ["id", "renewal"]
        ]

        tabs = st.tabs([
            "1. Información", "2. Tipos", "3. Descriptiva",
            "4. Faltantes", "5. Distribuciones", "6. Categóricas",
            "7. Numérica vs renovación", "8. Categórica vs renovación",
            "9. Análisis dinámico", "10. Hallazgos"
        ])

        # 1. Información general
        with tabs[0]:
            st.write(f"**Filas:** {len(df):,} | **Columnas:** {df.shape[1]:,}")

            a, b = st.columns(2)

            with a:
                st.markdown("**Tipos de datos**")
                st.dataframe(
                    pd.DataFrame({
                        "Variable": df.columns,
                        "Tipo": df.dtypes.astype(str).values
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            with b:
                st.markdown("**Valores faltantes**")
                st.dataframe(
                    missing_summary(df),
                    use_container_width=True,
                    hide_index=True
                )

            buffer = StringIO()
            df.info(buf=buffer)
            st.code(buffer.getvalue())

        # 2. Tipos de variables
        with tabs[1]:
            a, b = st.columns(2)
            a.metric("Variables numéricas", len(numeric))
            b.metric("Variables categóricas/texto", len(categorical))

            a, b = st.columns(2)
            a.write("**Numéricas**")
            a.dataframe(pd.DataFrame({"Variable": numeric}), hide_index=True)

            b.write("**Categóricas / texto**")
            b.dataframe(pd.DataFrame({"Variable": categorical}), hide_index=True)

            if "renewal" in df.columns:
                st.info("`renewal` es un indicador de negocio, aunque esté almacenado como número.")

        # 3. Estadísticas descriptivas
        with tabs[2]:
            if not numeric:
                st.info("No hay variables numéricas.")
            else:
                stats = df[numeric].describe().T
                stats["Mediana"] = df[numeric].median()
                stats["Moda"] = df[numeric].mode().iloc[0]
                st.dataframe(stats.round(3), use_container_width=True)

                variable = st.selectbox("Variable para interpretar", numeric)
                values = pd.to_numeric(df[variable], errors="coerce").dropna()

                if values.empty:
                    st.warning("No hay valores numéricos válidos.")
                else:
                    a, b, c = st.columns(3)
                    a.metric("Media", number(values.mean()))
                    b.metric("Mediana", number(values.median()))
                    c.metric("Desviación estándar", number(values.std()))

                    if values.mean() > values.median():
                        st.write("La media supera la mediana; podría existir asimetría hacia valores altos.")
                    elif values.mean() < values.median():
                        st.write("La media es inferior a la mediana; podría existir asimetría hacia valores bajos.")
                    else:
                        st.write("La media y la mediana coinciden o son muy cercanas.")

        # 4. Valores faltantes
        with tabs[3]:
            missing = missing_summary(df)
            missing = missing[missing["Valores faltantes"] > 0]

            st.metric("Celdas faltantes", f"{df.isna().sum().sum():,}")

            if missing.empty:
                st.success("No se detectaron valores faltantes.")
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=missing, x="% faltantes", y="Variable", ax=ax)
                ax.set_title("Porcentaje de valores faltantes")
                show_plot(fig)

                st.dataframe(missing, use_container_width=True, hide_index=True)
                st.info("Valida la causa de los faltantes antes de imputar o excluir registros.")

        # 5. Distribuciones numéricas
        with tabs[4]:
            if not numeric:
                st.info("No hay variables numéricas.")
            else:
                variable = st.selectbox("Variable numérica", numeric, key="hist")
                values = pd.to_numeric(df[variable], errors="coerce").dropna()

                if values.empty:
                    st.warning("No hay observaciones numéricas válidas.")
                else:
                    bins = st.slider("Número de intervalos", 5, 80, 30)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    sns.histplot(
                        values, bins=bins,
                        kde=values.nunique() > 1, ax=ax
                    )
                    ax.set_title(f"Distribución de {variable}")
                    show_plot(fig)

                    st.write(
                        f"**Observaciones:** {len(values):,} | "
                        f"**Media:** {number(values.mean())} | "
                        f"**Mediana:** {number(values.median())} | "
                        f"**Mínimo:** {number(values.min())} | "
                        f"**Máximo:** {number(values.max())}"
                    )

        # 6. Variables categóricas
        with tabs[5]:
            if not categorical:
                st.info("No hay variables categóricas.")
            else:
                variable = st.selectbox("Variable categórica", categorical)
                top = st.slider("Categorías a mostrar", 3, 30, 10)
                summary = categorical_summary(df, variable).head(top)

                st.dataframe(summary, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(9, 4))
                sns.barplot(data=summary, x="Frecuencia", y="Categoría", ax=ax)
                ax.set_title(f"Frecuencia por categoría: {variable}")
                show_plot(fig)

        # 7. Numérica vs renovación
        with tabs[6]:
            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")
            else:
                candidates = [c for c in numeric if c not in ["id", "renewal"]]

                if not candidates:
                    st.info("No hay otras variables numéricas disponibles.")
                else:
                    variable = st.selectbox("Variable numérica", candidates, key="num_ren")
                    temp = df[[variable, "renewal"]].copy()
                    temp[variable] = pd.to_numeric(temp[variable], errors="coerce")
                    temp = temp.dropna()
                    temp["Renovación"] = renewal_labels(temp["renewal"])

                    if temp.empty:
                        st.warning("No hay datos válidos para comparar.")
                    else:
                        fig, ax = plt.subplots(figsize=(9, 4))
                        sns.boxplot(data=temp, x="Renovación", y=variable, ax=ax)
                        ax.set_title(f"{variable} según renovación")
                        show_plot(fig)

                        summary = temp.groupby("Renovación")[variable].agg(
                            ["count", "mean", "median", "std", "min", "max"]
                        )
                        st.dataframe(summary.round(2), use_container_width=True)

                        st.caption("Las diferencias descriptivas no demuestran causalidad.")

        # 8. Categórica vs renovación
        with tabs[7]:
            if "renewal" not in df.columns:
                st.info("Se requiere la columna `renewal`.")
            else:
                candidates = [c for c in categorical if c != "renewal"]

                if not candidates:
                    st.info("No hay variables categóricas para comparar.")
                else:
                    variable = st.selectbox("Variable categórica", candidates, key="cat_ren")
                    temp = df[[variable, "renewal"]].copy()
                    temp[variable] = temp[variable].astype("string").fillna("(Sin dato)")
                    temp["Renovación"] = renewal_labels(temp["renewal"])

                    counts = pd.crosstab(temp[variable], temp["Renovación"])
                    percentages = pd.crosstab(
                        temp[variable], temp["Renovación"], normalize="index"
                    ).mul(100).round(2)

                    st.write("**Frecuencias**")
                    st.dataframe(counts, use_container_width=True)

                    st.write("**Porcentaje dentro de cada categoría**")
                    st.dataframe(percentages, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    percentages.plot(kind="bar", stacked=True, ax=ax)
                    ax.set_title(f"Renovación por {variable}")
                    ax.set_ylabel("Porcentaje (%)")
                    ax.tick_params(axis="x", rotation=30)
                    ax.legend(title="Renovación", bbox_to_anchor=(1.02, 1))
                    fig.tight_layout()
                    show_plot(fig)

        # 9. Análisis dinámico
        with tabs[8]:
            columns = st.multiselect(
                "Columnas a incluir",
                df.columns.tolist(),
                default=df.columns.tolist()[:min(5, len(df.columns))]
            )

            if not columns:
                st.warning("Selecciona al menos una columna.")
            else:
                filtered = df[columns].copy()

                if st.checkbox("Aplicar filtro por categoría"):
                    options = [c for c in columns if c in categorical]

                    if not options:
                        st.warning("Incluye una columna categórica para filtrar.")
                    else:
                        variable = st.selectbox("Filtrar por", options)
                        values = sorted(df[variable].dropna().astype(str).unique())

                        selected = st.multiselect(
                            "Valores incluidos", values,
                            default=values[:min(10, len(values))]
                        )

                        if selected:
                            filtered = filtered[
                                df[variable].astype(str).isin(selected)
                            ]
                        else:
                            filtered = filtered.iloc[0:0]

                st.metric("Registros filtrados", f"{len(filtered):,}")
                st.dataframe(filtered.head(500), use_container_width=True)

                if len(filtered) > 500:
                    st.caption("Se muestran hasta 500 filas; la descarga incluye todos los registros filtrados.")

                download_csv(filtered, "insurancecompany_seleccion.csv", "dynamic")

                selected_numeric = [c for c in columns if c in numeric]
                if selected_numeric and not filtered.empty:
                    st.write("**Resumen estadístico**")
                    st.dataframe(
                        filtered[selected_numeric].describe().T.round(2),
                        use_container_width=True
                    )

        # 10. Hallazgos
        with tabs[9]:
            total_missing = int(df.isna().sum().sum())
            total_cells = df.size

            a, b, c = st.columns(3)
            a.metric("Filas", f"{len(df):,}")
            b.metric("Columnas", f"{df.shape[1]:,}")
            c.metric("% celdas faltantes", f"{total_missing / total_cells * 100:.2f}%")

            missing = missing_summary(df)
            if total_missing:
                top = missing.iloc[0]
                st.write(
                    f"**Mayor cantidad de faltantes:** {top['Variable']} "
                    f"({top['Valores faltantes']:,}; {top['% faltantes']:.2f}%)."
                )
            else:
                st.success("No hay valores faltantes.")

            st.write(f"Variables numéricas: **{len(numeric)}**.")
            st.write(f"Variables categóricas/texto: **{len(categorical)}**.")

            if "id" in df.columns:
                st.write(f"Identificadores repetidos: **{df['id'].duplicated().sum():,}**.")

            if "renewal" in df.columns:
                renewal = df["renewal"].value_counts(dropna=False)
                summary = pd.DataFrame({
                    "Valor": renewal.index.astype(str),
                    "Frecuencia": renewal.values,
                    "Proporción (%)": (
                        renewal.values / len(df) * 100
                    ).round(2)
                })

                st.write("**Distribución de renovación**")
                st.dataframe(summary, use_container_width=True, hide_index=True)


# =========================================================
# 8. CONCLUSIONES
# =========================================================
elif module == "Conclusiones":
    st.subheader("Conclusiones finales")

    if df is None:
        st.warning("Carga un CSV válido para generar conclusiones.")
    else:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            or c in ["id", "renewal"]
        ]

        missing = int(df.isna().sum().sum())
        cells = df.size

        conclusions = [
            f"**Tamaño:** el dataset contiene {len(df):,} registros y {df.shape[1]} variables.",
            (
                f"**Calidad:** hay {missing:,} celdas faltantes "
                f"({missing / cells * 100:.2f}%)."
                if missing else
                "**Calidad:** no se detectaron valores faltantes."
            ),
            f"**Tipos:** se identificaron {len(numeric)} variables numéricas y {len(categorical)} categóricas/texto."
        ]

        if "renewal" in df.columns:
            proportions = df["renewal"].value_counts(normalize=True, dropna=False) * 100
            conclusions.append(
                f"**Renovación:** el valor más frecuente es "
                f"`{proportions.idxmax()}` ({proportions.max():.2f}%). "
                "Confirma la codificación con el diccionario de datos."
            )
        else:
            conclusions.append(
                "**Renovación:** no existe la columna `renewal`; "
                "no se puede evaluar este indicador."
            )

        if "id" in df.columns:
            conclusions.append(
                f"**Identificadores:** hay {df['id'].duplicated().sum():,} "
                "valores repetidos en `id`; revisa si debe ser único."
            )
        else:
            conclusions.append(
                "**Identificadores:** no existe una columna `id` para verificar unicidad."
            )

        for i, conclusion in enumerate(conclusions[:5], 1):
            st.markdown(f"**Conclusión {i}.** {conclusion}")

        st.divider()
        st.markdown("#### Recomendaciones")

        st.markdown("""
        - Validar la calidad y definición de las variables.
        - Comparar porcentajes y tamaños de grupo.
        - Contrastar los resultados con las áreas de negocio.
        - No inferir causalidad a partir de relaciones descriptivas.
        - Documentar los criterios de limpieza antes de realizar análisis posteriores.
        """)


# =========================================================
# 9. PIE DE PÁGINA
# =========================================================
st.sidebar.divider()
st.sidebar.caption("InsuranceCompany EDA · Aplicación académica")
```
