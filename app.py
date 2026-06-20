import streamlit as st
import joblib
from Bio import pairwise2

# Cargar modelo y vectorizador
model = joblib.load("modelo_adn.pkl")
vectorizer = joblib.load("vectorizer.pkl")
INFO_MICROORGANISMOS = {
    "Ecoli": {
        "nombre": "Escherichia coli",
        "tipo": "Bacteria",
        "clasificacion": "Intestinal, ambiental/alimentaria y hospitalaria oportunista",
        "ambiente": "Se encuentra en el intestino de personas y animales, y también puede estar en alimentos, agua y ambiente contaminado.",
        "enfermedades": [
            "Diarrea",
            "Infecciones urinarias",
            "Neumonía",
            "Sepsis u otras infecciones graves"
        ],
        "sintomas": [
            "Diarrea",
            "Dolor o cólicos abdominales",
            "Vómitos",
            "Fiebre en algunos casos",
            "Sangre en heces u orina en casos graves"
        ],
        "nota": "No todas las cepas de E. coli son dañinas. Muchas viven normalmente en el intestino."
    },

    "Bacillus": {
        "nombre": "Bacillus spp.",
        "tipo": "Bacteria formadora de esporas",
        "clasificacion": "Principalmente ambiental; algunas especies pueden ser alimentarias u oportunistas",
        "ambiente": "Se encuentra comúnmente en la naturaleza, especialmente en suelo, polvo, agua y alimentos.",
        "enfermedades": [
            "Intoxicación alimentaria, especialmente por Bacillus cereus",
            "Infecciones oportunistas en casos especiales",
            "Algunas especies pueden estar relacionadas con contaminación ambiental o de laboratorio"
        ],
        "sintomas": [
            "Náuseas",
            "Vómitos",
            "Diarrea",
            "Dolor abdominal",
            "Cólicos"
        ],
        "nota": "Bacillus es un género amplio. No todas sus especies causan enfermedad; muchas son ambientales e inofensivas."
    }
}
# Función k-mers
def get_kmers(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# Función de mutaciones
def detectar_mutaciones(ref_seq, query_seq):

    alineamiento = pairwise2.align.globalms(
        ref_seq,
        query_seq,
        2,
        -1,
        -2,
        -0.5
    )[0]

    ref_alineada = alineamiento.seqA
    query_alineada = alineamiento.seqB

    mutaciones = []

    pos_ref = 0

    for i in range(len(ref_alineada)):

        if ref_alineada[i] != "-":
            pos_ref += 1

        if ref_alineada[i] == query_alineada[i]:
            continue

        if ref_alineada[i] == "-":

            mutaciones.append(
                f"Inserción en posición {pos_ref}: {query_alineada[i]}"
            )

        elif query_alineada[i] == "-":

            mutaciones.append(
                f"Deleción en posición {pos_ref}: {ref_alineada[i]}"
            )

        else:

            mutaciones.append(
                f"Sustitución en posición {pos_ref}: "
                f"{ref_alineada[i]} → {query_alineada[i]}"
            )

    return mutaciones


st.title("Análisis de ADN con Inteligencia Artificial")

tab1, tab2 = st.tabs([
    "Clasificación",
    "Mutaciones"
])

# ------------------------------------------------
# CLASIFICACIÓN
# ------------------------------------------------

with tab1:

    st.header("Identificación de microorganismos")

    secuencia = st.text_area(
        "Ingrese la secuencia ADN"
    )

    if st.button("Clasificar"):

        kmers = get_kmers(secuencia)

        texto = " ".join(kmers)

        X = vectorizer.transform([texto])

        pred = model.predict(X)[0]

        prob = model.predict_proba(X).max() * 100

        st.success(
            f"Microorganismo identificado: {pred}"
        )

        st.info(
            f"Confianza del modelo: {prob:.2f}%"
        )

# ------------------------------------------------
# MUTACIONES
# ------------------------------------------------

with tab2:

    st.header("Detección de mutaciones")

    referencia = st.text_area(
        "Secuencia de referencia"
    )

    muestra = st.text_area(
        "Secuencia a analizar"
    )

    if st.button("Buscar mutaciones"):

        mutaciones = detectar_mutaciones(
            referencia,
            muestra
        )

        if len(mutaciones) == 0:

            st.success(
                "No se encontraron mutaciones."
            )

        else:

            st.warning(
                f"Se detectaron {len(mutaciones)} mutaciones"
            )

            for m in mutaciones:
                st.write(m)