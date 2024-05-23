# Chat_Bot_medical
Resumen del Flujo de Trabajo
1. Conversión de XML a JSON
Función: xml_to_json(xml_file, json_file)
Descripción: Esta función toma un archivo XML, lo analiza, extrae información relevante y lo guarda en un archivo JSON.
Pasos:
Parsear el archivo XML.
Extraer preguntas y sub-preguntas con sus respectivos atributos.
Guardar los datos estructurados en un archivo JSON.

import xml.etree.ElementTree as ET
import json

def xml_to_json(xml_file, json_file):
    try:
        # Parsear el archivo XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Crear una lista para almacenar las preguntas
        questions = []

        # Iterar sobre cada pregunta en el archivo XML
        for question in root.findall('NLM-QUESTION'):
            qid = question.get('qid')
            subject = question.find('SUBJECT').text if question.find('SUBJECT') is not None else ''
            message = question.find('MESSAGE').text if question.find('MESSAGE') is not None else ''

            sub_questions = []
            for sub_question in question.findall('SUB-QUESTIONS/SUB-QUESTION'):
                focus = sub_question.find('ANNOTATIONS/FOCUS').text if sub_question.find('ANNOTATIONS/FOCUS') is not None else ''
                qtype = sub_question.find('ANNOTATIONS/TYPE').text if sub_question.find('ANNOTATIONS/TYPE') is not None else ''
                answer = sub_question.find('ANSWERS/ANSWER').text if sub_question.find('ANSWERS/ANSWER') is not None else ''

                sub_questions.append({
                    "focus": focus,
                    "type": qtype,
                    "answer": answer
                })

            questions.append({
                "qid": qid,
                "subject": subject,
                "message": message,
                "sub_questions": sub_questions
            })

        # Convertir la lista de preguntas a JSON
        with open(json_file, 'w') as f:
            json.dump(questions, f, indent=4)

        print("Archivo JSON guardado en:", json_file)

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Ruta al archivo XML y JSON
xml_file_path = r"C:\Users\danny\OneDrive\Escritorio\FinBotcamp4\TrainingDatasets\TREC-2017-LiveQA-Medical-Train-1.xml"
json_file_path = r"C:\Users\danny\OneDrive\Escritorio\FinBotcamp4\TrainingDatasets.json"

# Llamar a la función para convertir XML a JSON
xml_to_json(xml_file_path, json_file_path)

import json
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar los datos desde un archivo JSON
with open('TrainingDatasets.json', 'r') as f:
    data = json.load(f)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Función para limpiar el texto
def clean_text(text):
    if text is None:
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Quitar signos especiales
    text = re.sub(r'\s+', ' ', text)  # Quitar espacios extras
    return text.strip()

# Aplicar la función de limpieza al 'message'
df['message'] = df['message'].apply(clean_text)

# Función para limpiar cada sub-pregunta en la lista
def clean_sub_questions(sub_questions):
    cleaned_sub_questions = []
    for sub_question in sub_questions:
        cleaned_sub_question = sub_question.copy()
        if 'annotations' in sub_question:
            if 'FOCUS' in sub_question['annotations']:
                cleaned_sub_question['annotations']['FOCUS'] = clean_text(sub_question['annotations']['FOCUS'])
            if 'TYPE' in sub_question['annotations']:
                cleaned_sub_question['annotations']['TYPE'] = clean_text(sub_question['annotations']['TYPE'])
        if 'answers' in sub_question:
            cleaned_sub_question['answers'] = [clean_text(answer) for answer in sub_question['answers']]
        cleaned_sub_questions.append(cleaned_sub_question)
    return cleaned_sub_questions

# Aplicar la función de limpieza a 'sub_questions'
df['sub_questions'] = df['sub_questions'].apply(clean_sub_questions)

# Imprimir el DataFrame limpio para verificar
print(df.to_string(index=False))

# Guardar los conjuntos de datos en archivos CSV
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

import json
import pandas as pd

# Cargar los datos desde el JSON
with open('TrainingDatasets.json', 'r') as f:
    data = json.load(f)

# Crear una lista para almacenar los datos extraídos
focus_list = []
type_list = []
answer_list = []

# Iterar sobre cada entrada en los datos
for entry in data:
    # Obtener los valores de focus, type y answer de sub_questions
    sub_question = entry.get('sub_questions', [{}])[0]  # Obtener el primer elemento de sub_questions
    focus = sub_question.get('focus', '')
    _type = sub_question.get('type', '')
    answer = sub_question.get('answer', '')
    
    # Agregar los valores a las listas respectivas
    focus_list.append(focus)
    type_list.append(_type)
    answer_list.append(answer)

# Crear un DataFrame con los datos extraídos
extracted_data = pd.DataFrame({
    'focus': focus_list,
    'type': type_list,
    'answer': answer_list
})

# Imprimir el DataFrame
print(extracted_data)

# Concatenar el DataFrame original con el DataFrame de los datos extraídos
new_df = pd.concat([df.drop(columns=['sub_questions'])] * len(extracted_data), ignore_index=True)

# Añadir las columnas 'focus', 'type' y 'answer' al DataFrame nuevo
new_df['focus'] = extracted_data['focus']
new_df['type'] = extracted_data['type']
new_df['answer'] = extracted_data['answer']

# Imprimir las primeras filas del DataFrame nuevo para verificar
print(new_df.head())

# Reordenar las columnas
new_df = new_df[['qid', 'subject', 'message', 'focus', 'type', 'answer']]

# Imprimir el DataFrame con un formato más legible
print(new_df.to_string(index=False))

2. Preprocesamiento de Datos
Descripción: Limpieza y transformación de los datos para prepararlos para el entrenamiento del modelo.
Pasos:
Cargar los datos desde el archivo JSON.
Limpiar el texto de las preguntas y sub-preguntas (eliminar caracteres especiales, convertir a minúsculas, etc.).
Extraer sub-preguntas en columnas separadas para foco, tipo y respuesta.
Guardar los datos preprocesados en un nuevo archivo JSON.
# Entrenamiento de modelo

!pip install nltk
!pip install spacy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

!python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text.lower())
    # Eliminar palabras vacías (stop words)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

#Elimnar valores NaN
new_df['message'].fillna('', inplace=True)
new_df['type'].fillna('', inplace=True)
new_df ['subject'].fillna('', inplace=True)
new_df ['focus'].fillna('', inplace=True)
new_df ['answer'].fillna('', inplace=True)

new_df['message'] = new_df['message'].apply(preprocess_text)

# Dividir los datos en características (X) y etiquetas (y)
X = new_df['message']
y = new_df['type']
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Convertir el texto a vectores TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

Entrenar un modelo de clasificación
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

Predecir los tipos de preguntas en el conjunto de prueba
y_pred = model.predict(X_test_tfidf)

Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Inicializar el lematizador y el stemmer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

Definir una función para preprocesar el texto
def preprocess_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text.lower())

    # Eliminar puntuaciones y palabras vacías
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]

    # Lematizar o derivar las palabras
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

Ejemplo de uso
text = "12 years ago I was bitten by tick while deer hunting..."
preprocessed_text = preprocess_text(text)
print(preprocessed_text)

Vectorizar textos
from sklearn.feature_extraction.text import TfidfVectorizer

# Definir un vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Entrenar el vectorizador y transformar los textos de entrenamiento
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transformar los textos de prueba
X_test_tfidf = vectorizer.transform(X_test)

# Imprimir la forma de las matrices resultantes
print("Forma de la matriz TF-IDF de entrenamiento:", X_train_tfidf.shape)
print("Forma de la matriz TF-IDF de prueba:", X_test_tfidf.shape)

Entrenar modelo
from sklearn.naive_bayes import MultinomialNB

# Inicializar el clasificador Naive Bayes
classifier = MultinomialNB()

# Entrenar el clasificador utilizando los datos de entrenamiento
classifier.fit(X_train_tfidf, y_train)

# Predecir las etiquetas de los datos de prueba
y_pred = classifier.predict(X_test_tfidf)

# Evaluar el rendimiento del clasificador
from sklearn.metrics import accuracy_score, classification_report

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del clasificador Naive Bayes:", accuracy)

# Mostrar un reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

Evaluación de modelo

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Mostrar un reporte de clasificación detallado
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Mostrar un reporte de clasificación detallado
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)
# Crear un mapa de calor para la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)

# Añadir etiquetas y título
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')

# Mostrar el mapa de calor
plt.show()


# Guardar el DataFrame en un archivo JSON
new_df = new_df.drop('qid', axis = 1)
new_df.to_json('new_df.json', orient='records')

3. Manejo de Desbalanceo de Clases
Descripción: Balancear las clases en el conjunto de datos utilizando SMOTE para evitar sesgos en el modelo.
Pasos:
Identificar y manejar clases con pocas muestras.
Aplicar SMOTE para generar muestras sintéticas y equilibrar las clases.
Guardar los datos balanceados en un archivo CSV.

#MEJORA DE RENDIMIENTO:
import pandas as pd

# Cargar el DataFrame desde el archivo JSON
df_imported = pd.read_json('new_df.json', orient='records')

# Mostrar los primeros registros para verificar
print(df_imported.head())

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Cargar el DataFrame desde el archivo JSON
df = pd.read_json('new_df.json', orient='records')

#columna de etiquetas y las demás son características
X = df.drop('type', axis=1)
y = df['type']

# Identificar las columnas categóricas
columnas_categoricas = X.select_dtypes(include=['object']).columns


# Crear un preprocesador que convierte las características categóricas en numéricas
preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ],
    remainder='passthrough'  # Dejar las columnas numéricas sin cambios
)

# Crear un pipeline que incluye la imputación de valores faltantes y la codificación de variables categóricas
pipeline = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('imputador', SimpleImputer(strategy='most_frequent'))  # Imputar valores faltantes con la moda
])

# Aplicar el preprocesador a los datos
X_preprocesado = pipeline.fit_transform(X)

# Convertir la matriz dispersa a una matriz densa si es necesario
if hasattr(X_preprocesado, 'toarray'):
    X_preprocesado = X_preprocesado.toarray()

# Calcular el número mínimo de muestras en una clase
minimo_muestras_clase = y.value_counts().min()

# Si hay alguna clase con menos de 2 muestras, duplicar esas muestras manualmente
if minimo_muestras_clase < 2:
    clases_con_una_muestra = y.value_counts()[y.value_counts() < 2].index
    for clase in clases_con_una_muestra:
        X_clase = X_preprocesado[y == clase]
        y_clase = y[y == clase]
        # Duplicar las muestras de esa clase
        X_preprocesado = np.vstack([X_preprocesado, X_clase])
        y = pd.concat([y, y_clase], ignore_index=True)
    # Recalcular el número mínimo de muestras en una clase
    minimo_muestras_clase = y.value_counts().min()

# Ajustar el parámetro k_neighbors de SMOTE para asegurar que sea válido
k_neighbors = min(minimo_muestras_clase - 1, 5)

# Aplicar SMOTE para balancear las clases con k_neighbors ajustado
smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
X_res, y_res = smote.fit_resample(X_preprocesado, y)

# Obtener los nombres de las características después de la codificación one-hot
nombres_caracteristicas = pipeline.named_steps['preprocesador'].get_feature_names_out()

# Crear un nuevo DataFrame con los datos balanceados
nuevo_df_res = pd.DataFrame(X_res, columns=nombres_caracteristicas)
nuevo_df_res['type'] = y_res

# Verificar el tamaño del DataFrame antes de guardarlo
print(f"El tamaño del DataFrame es: {nuevo_df_res.shape}")

# Guardar el DataFrame balanceado en un archivo CSV en lugar de JSON
nuevo_df_res.to_csv('new_df_res.csv', index=False)

print("\nNuevo DataFrame guardado como 'new_df_res.csv'.")

5. Entrenamiento del Modelo
Descripción: Entrenar un modelo de aprendizaje automático para clasificar las preguntas médicas.
Pasos:
Dividir los datos en conjuntos de entrenamiento y prueba.
Entrenar un modelo de bosque aleatorio (Random Forest).
Evaluar el modelo utilizando precisión y reporte de clasificación.
Guardar el modelo entrenado y el imputador.
mport pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Leer el archivo CSV
df_res_sin_nan = pd.read_csv('new_df_res.csv')

# Verificar si hay valores NaN en el DataFrame original
print("Valores NaN en el DataFrame original:")
print(df_res_sin_nan.isnull().sum())

# Eliminar filas con valores NaN en la columna 'type' si los hay
df_res_sin_nan = df_res_sin_nan.dropna(subset=['type'])

# Separar las características y la etiqueta
X = df_res_sin_nan.drop('type', axis=1)
y = df_res_sin_nan['type']

# Imputar los valores NaN en X con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Verificar si hay valores NaN en los datos imputados
print("Valores NaN en los datos imputados:")
print(np.isnan(X_imputed).any())

# Dividir los datos imputados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar un modelo de clasificación (Random Forest en este caso)
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
precision = accuracy_score(y_test, y_pred)
reporte_clasificacion = classification_report(y_test, y_pred)

print(f"Precisión del modelo: {precision}")
print("Reporte de clasificación:")
print(reporte_clasificacion)

import joblib

# Guardar el modelo entrenado y el imputador en un archivo
joblib.dump(modelo, 'modelo_entrenado.pkl')
joblib.dump(imputer, 'imputador.pkl')

print("Modelo y imputador guardados exitosamente.")

6. Inferencia
Descripción: Utilizar el modelo entrenado para realizar predicciones sobre nuevos datos.
Pasos:
Cargar el modelo y el imputador.
Preprocesar los nuevos datos.
Realizar predicciones.
Guardar las predicciones en un archivo CSV.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Mostrar un reporte de clasificación detallado
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)
# Crear un mapa de calor para la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)

# Añadir etiquetas y título
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')

# Mostrar el mapa de calor
plt.show()

import joblib

# Guardar el modelo entrenado y el imputador en un archivo
joblib.dump(modelo, 'modelo_entrenado.pkl')
joblib.dump(imputer, 'imputador.pkl')

print("Modelo y imputador guardados exitosamente.")

import pandas as pd

# Cargar el DataFrame desde el archivo JSON
df = pd.read_json("new_df.json", orient="records")

# Definir una función para buscar respuestas en el DataFrame
def buscar_respuesta(pregunta):
    # Buscar en el DataFrame una fila que contenga la pregunta
    resultado = df[df['message'].str.contains(pregunta, case=False)]
    
    # Si se encuentra una respuesta, devolverla
    if len(resultado) > 0:
        return resultado.iloc[0]['answer']
    else:
        return "Lo siento, no encontré una respuesta para esa pregunta."

# Bucle principal para que el chatbot interactúe con el usuario
while True:
    # Solicitar al usuario que ingrese una pregunta
    pregunta = input("Tú: ")
    
    # Salir del bucle si el usuario ingresa "salir"
    if pregunta.lower() == "salir":
        print("¡Hasta luego!")
        break
    
    # Buscar una respuesta basada en la pregunta del usuario
    respuesta = buscar_respuesta(pregunta)
    
    # Mostrar la respuesta encontrada o un mensaje de error si no se encuentra una respuesta
    print("Chatbot: " + respuesta)

INFERENCIA:
import pandas as pd

# Leer los datos desde el archivo CSV
df_res_sin_nan = pd.read_csv('new_df_res.csv')

# Verificar los primeros registros para asegurarse de que los datos se han cargado correctamente
print(df_res_sin_nan.head())

import pandas as pd
import joblib

# Leer los datos desde el archivo CSV
df_res_sin_nan = pd.read_csv('new_df_res.csv')

# Verificar los primeros registros para asegurarse de que los datos se han cargado correctamente
print(df_res_sin_nan.head())

# Cargar el pipeline de preprocesamiento y el modelo entrenado
pipeline = joblib.load('imputador.pkl')
modelo = joblib.load('modelo_entrenado.pkl')

# Separar características y etiquetas si es necesario
if 'type' in df_res_sin_nan.columns:
    X_nuevos = df_res_sin_nan.drop('type', axis=1)
else:
    X_nuevos = df_res_sin_nan

# Preprocesar los nuevos datos
X_nuevos_preprocesado = pipeline.transform(X_nuevos)

# Realizar predicciones
predicciones = modelo.predict(X_nuevos_preprocesado)

# Crear un DataFrame con los resultados
resultados = df_res_sin_nan.copy()
resultados['prediccion'] = predicciones

# Guardar los resultados en un archivo CSV
resultados.to_csv('resultados_predicciones.csv', index=False)

print("Predicciones realizadas y guardadas en 'resultados_predicciones.csv'.")

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración de MLflow
mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_experiment('modelo_entrenado.pkl')

# Cargar datos de ejemplo
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Inicia una nueva ejecución en MLflow
with mlflow.start_run():
    # Parámetros del modelo
    n_estimators = 100
    max_depth = 5
    
    # Registrar parámetros
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    
    # Entrena el modelo
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Registrar métricas
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)

    # Registrar el modelo
    mlflow.sklearn.log_model(model, 'random_forest_model')

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Definir la función de preprocesamiento
def preprocess_data(data):
    # Manejo de valores faltantes
    data.fillna(data.mean(), inplace=True)
    
    # Normalización de datos
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Codificación de variables categóricas.
    data_encoded = pd.get_dummies(data, drop_first=True)
    
    # Ajustar escalador sobre los datos codificados
    data_encoded = scaler.fit_transform(data_encoded)
    
    return data_encoded

# Leer los datos desde el archivo CSV
df_res_sin_nan = pd.read_csv('new_df_res.csv')

# Verificar los primeros registros para asegurarse de que los datos se han cargado correctamente
print(df_res_sin_nan.head())

# Cargar el modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

# Separar características y etiquetas 
if 'type' in df_res_sin_nan.columns:
    X_nuevos = df_res_sin_nan.drop('type', axis=1)
else:
    X_nuevos = df_res_sin_nan

# Preprocesar los nuevos datos utilizando la función definida
X_nuevos_preprocesado = preprocess_data(X_nuevos)

# Realizar predicciones
predicciones = modelo.predict(X_nuevos_preprocesado)

# Crear un DataFrame con los resultados
resultados = df_res_sin_nan.copy()
resultados['prediccion'] = predicciones

# Guardar los resultados en un archivo CSV
resultados.to_csv('resultados_predicciones.csv', index=False)

print("Predicciones realizadas y guardadas en 'resultados_predicciones.csv'.")

import csv
import json

def csv_to_json(csv_file, json_file):
    # Abre el archivo CSV en modo lectura
    with open(csv_file, 'r', newline='') as csvfile:
        # Lee los datos del archivo CSV
        csv_reader = csv.DictReader(csvfile)
        # Convierte los datos a una lista de diccionarios
        data = list(csv_reader)

    # Abre el archivo JSON en modo escritura
    with open(json_file, 'w') as jsonfile:
        # Escribe los datos en el archivo JSON
        json.dump(data, jsonfile, indent=4)

# Archivo CSV de entrada
csv_file = 'resultados_predicciones.csv'
# Archivo JSON de salida
json_file = 'resultados_predicciones.json'

# Convierte el archivo CSV a JSON
csv_to_json(csv_file, json_file)

print("Se ha convertido el archivo CSV a JSON con éxito.")

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configura la URI de seguimiento de MLflow (local o remoto)
mlflow.set_tracking_uri("http://localhost:8080")  # Ejemplo con servidor local
mlflow.set_experiment("modelo_entrenado.pkl")

def log_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    return accuracy, precision, recall, f1
