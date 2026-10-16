import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Insurance Company - EDA",
    page_icon="📊",
    layout="wide"
)


class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def dataset_info(self):
        return pd.DataFrame({
            "Tipo": self.df.dtypes.astype(str),
            "Nulos": self.df.isnull().sum(),
            "% Nulos": round(
                self.df.isnull().sum()*100/len(self.df),
                2
            )
        })

    def variables_numericas(self):
        return self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

    def variables_categoricas(self):
        return self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

    def descriptivos(self):
        return self.df.describe()

    def correlacion(self):
        return self.df.corr(numeric_only=True)


st.title("📊 Insurance Company - Exploratory Data Analysis")

st.markdown("""
Aplicación desarrollada para el análisis exploratorio de datos del dataset Insurance Company.
""")

archivo = st.file_uploader(
    "Seleccione el archivo CSV",
    type=["csv"]
)

if archivo is not None:

    df = pd.read_csv(archivo)

    analyzer = DataAnalyzer(df)

    st.success("Archivo cargado correctamente")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Datos",
        "EDA General",
        "Variables",
        "Relaciones",
        "Hallazgos"
    ])

    with tab1:

        st.subheader("Vista previa")

        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Filas",
            df.shape[0]
        )

        col2.metric(
            "Columnas",
            df.shape[1]
        )

        col3.metric(
            "Variables Numéricas",
            len(analyzer.variables_numericas())
        )

    with tab2:

        st.subheader("Información General")

        st.dataframe(
            analyzer.dataset_info()
        )

        st.subheader("Estadísticas Descriptivas")

        st.dataframe(
            analyzer.descriptivos()
        )

        st.subheader("Valores Faltantes")

        missing = df.isnull().sum()

        fig, ax = plt.subplots(figsize=(10, 4))
        missing[missing > 0].plot(
            kind="bar",
            ax=ax
        )
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Matriz de Correlación")

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            analyzer.correlacion(),
            cmap="RdBu_r",
            ax=ax
        )

        st.pyplot(fig)

    with tab3:

        st.subheader("Análisis de Variables")

        variable = st.selectbox(
            "Seleccione una variable",
            analyzer.variables_numericas()
        )

        fig, ax = plt.subplots()

        sns.histplot(
            df[variable],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

        st.subheader("Boxplot")

        fig, ax = plt.subplots()

        sns.boxplot(
            x=df[variable],
            ax=ax
        )

        st.pyplot(fig)

        st.subheader("Variables categóricas")

        cat_cols = analyzer.variables_categoricas()

        if len(cat_cols) > 0:

            cat_var = st.selectbox(
                "Variable categórica",
                cat_cols
            )

            fig, ax = plt.subplots(
                figsize=(8,4)
            )

            df[cat_var].value_counts().plot(
                kind="bar",
                ax=ax
            )

            st.pyplot(fig)

    with tab4:

        st.subheader("Análisis Bivariado")

        num_cols = analyzer.variables_numericas()

        x_var = st.selectbox(
            "Variable X",
            num_cols,
            key="x"
        )

        y_var = st.selectbox(
            "Variable Y",
            num_cols,
            key="y"
        )

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df,
            x=x_var,
            y=y_var,
            hue="renewal",
            ax=ax
        )

        st.pyplot(fig)

        st.subheader("Renewal por categorías")

        if "sourcing_channel" in df.columns:

            fig, ax = plt.subplots()

            sns.countplot(
                data=df,
                x="sourcing_channel",
                hue="renewal",
                ax=ax
            )

            st.pyplot(fig)

    with tab5:

        st.subheader("Análisis Interactivo")

        primas = st.slider(
            "Prima máxima",
            int(df["premium"].min()),
            int(df["premium"].max()),
            int(df["premium"].max())
        )

        data_filtrada = df[
            df["premium"] <= primas
        ]

        st.write(
            "Registros:",
            len(data_filtrada)
        )

        canales = st.multiselect(
            "Canales",
            df["sourcing_channel"].unique(),
            default=list(df["sourcing_channel"].unique())
        )

        data_filtrada = data_filtrada[
            data_filtrada["sourcing_channel"].isin(canales)
        ]

        mostrar = st.checkbox(
            "Mostrar datos filtrados"
        )

        if mostrar:
            st.dataframe(data_filtrada)

        st.subheader("Hallazgos sugeridos")

        st.markdown("""
        1. Identificar variables con mayor correlación con renewal.
        2. Evaluar el impacto de los atrasos de pago.
        3. Comparar zonas Urban y Rural.
        4. Analizar diferencias por canal de captación.
        5. Revisar comportamiento de clientes con primas altas.
        6. Investigar registros con underwriting score faltante.
        7. Analizar distribución de ingresos.
        8. Identificar outliers.
        9. Comparar edades de clientes.
        10. Evaluar variables más importantes para renovación.
        """)

else:
    st.info("Cargue el archivo CSV para iniciar el análisis.")
