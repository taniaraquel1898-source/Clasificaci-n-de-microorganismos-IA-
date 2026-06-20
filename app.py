import streamlit as st
import joblib
from Bio import pairwise2

# Cargar modelo y vectorizador
model = joblib.load("modelo_adn.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Información biomédica de los microorganismos
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
            "Dolor abdominal",
            "Vómitos",
            "Fiebre en algunos casos"
        ],
        "nota": "No todas las cepas de E. coli son dañinas. Muchas viven normalmente en el intestino."
    },

    "Bacillus": {
        "nombre": "Bacillus spp.",
        "tipo": "Bacteria formadora de esporas",
        "clasificacion": "Principalmente ambiental; algunas especies pueden ser alimentarias u oportunistas",
        "ambiente": "Se encuentra comúnmente en suelo, polvo, agua y alimentos.",
        "enfermedades": [
            "Intoxicación alimentaria",
            "Infecciones oportunistas",
            "Contaminación ambiental o de laboratorio"
        ],
        "sintomas": [
            "Náuseas",
            "Vómitos",
            "Diarrea",
            "Dolor abdominal",
            "Cólicos"
        ],
        "nota": "Bacillus es un género amplio. No todas sus especies causan enfermedad."
    }
}


# Función para mostrar información biomédica
def mostrar_info_microorganismo(prediccion):
    info = INFO_MICROORGANISMOS.get(prediccion)

    if info is None:
        st.warning("No hay información biomédica registrada para este microorganismo.")
        return

    st.subheader("Información biomédica del microorganismo")

    st.write(f"**Nombre científico:** {info['nombre']}")
    st.write(f"**Tipo:** {info['tipo']}")
    st.write(f"**Clasificación:** {info['clasificacion']}")
    st.write(f"**Ambiente o fuente frecuente:** {info['ambiente']}")

    st.write("**Enfermedades asociadas:**")
    for enfermedad in info["enfermedades"]:
        st.write(f"- {enfermedad}")

    st.write("**Síntomas que puede causar:**")
    for sintoma in info["sintomas"]:
        st.write(f"- {sintoma}")

    st.info(info["nota"])


# Función para generar k-mers
def get_kmers(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


# Función para detectar mutaciones
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
                f"Sustitución en posición {pos_ref}: {ref_alineada[i]} → {query_alineada[i]}"
            )

    return mutaciones


# Interfaz de Streamlit
st.title("Análisis de ADN con Inteligencia Artificial")

tab1, tab2 = st.tabs(["Clasificación", "Mutaciones"])


# Pestaña 1: Clasificación
with tab1:

    st.header("Identificación de microorganismos")

    secuencia = st.text_area("Ingrese la secuencia ADN")

    if st.button("Clasificar"):

        secuencia = secuencia.upper().replace("\n", "").replace(" ", "")

        if secuencia == "":
            st.warning("Debe ingresar una secuencia de ADN.")
        else:
            kmers = get_kmers(secuencia)
            texto = " ".join(kmers)
            X = vectorizer.transform([texto])

            pred = model.predict(X)[0]
            prob = model.predict_proba(X).max() * 100

            st.success(f"Microorganismo identificado: {pred}")
            st.info(f"Confianza del modelo: {prob:.2f}%")

            mostrar_info_microorganismo(pred)


# Pestaña 2: Mutaciones
with tab2:

    st.header("Detección de mutaciones genéticas")

    referencia = st.text_area("Secuencia de referencia")
    muestra = st.text_area("Secuencia a analizar")

    if st.button("Buscar mutaciones"):

        referencia = referencia.upper().replace("\n", "").replace(" ", "")
        muestra = muestra.upper().replace("\n", "").replace(" ", "")

        if referencia == "" or muestra == "":
            st.warning("Debe ingresar ambas secuencias.")
        else:
            mutaciones = detectar_mutaciones(referencia, muestra)

            if len(mutaciones) == 0:
                st.success("No se encontraron mutaciones.")
            else:
                st.warning(f"Se detectaron {len(mutaciones)} mutaciones")

                for m in mutaciones:
                    st.write(m)