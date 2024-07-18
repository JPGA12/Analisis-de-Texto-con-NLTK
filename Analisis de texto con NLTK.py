import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd

# Descargar los recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Texto de ejemplo
text = """
El procesamiento de lenguaje natural (PLN) es una subdisciplina de la inteligencia artificial (IA) que se ocupa de la interacción entre las computadoras y los lenguajes humanos. El objetivo del PLN es permitir que las computadoras comprendan, interpreten y generen el lenguaje humano de una manera que sea valiosa. Las aplicaciones de PLN son variadas y se utilizan en sistemas de traducción automática, asistentes virtuales como Siri y Alexa, análisis de sentimientos en redes sociales, entre otros.

Una de las primeras fases del PLN es el análisis morfológico, que descompone las palabras en sus morfemas, las unidades más pequeñas de significado. Esta fase es fundamental para comprender cómo se forman y varían las palabras en diferentes contextos gramaticales. Por ejemplo, en español, la palabra "niños" se puede descomponer en la raíz "niño" y el sufijo plural "s".

El análisis sintáctico es la siguiente fase y se encarga de organizar las palabras en una oración de acuerdo con las reglas gramaticales del lenguaje. Esto se realiza a través de la identificación de partes del discurso, como sustantivos, verbos y adjetivos, y la creación de estructuras jerárquicas que representan la relación entre estas partes. Por ejemplo, en la oración "El gato duerme en la alfombra", el análisis sintáctico identificaría "El gato" como el sujeto y "duerme en la alfombra" como el predicado.

El análisis semántico busca comprender el significado de las palabras y las oraciones. Esto incluye la resolución de ambigüedades léxicas, donde una palabra puede tener múltiples significados, y la representación del significado a través de estructuras semánticas. Por ejemplo, la palabra "banco" puede referirse a una entidad financiera o a un asiento, dependiendo del contexto en el que se use.

La integración del discurso es una fase que va más allá de las oraciones individuales y considera el contexto más amplio del texto. Esto incluye la resolución de referencias y la coherencia temática a lo largo de un párrafo o documento. Por ejemplo, si un texto menciona "María fue al mercado" y más adelante "ella compró frutas", el análisis del discurso debe identificar que "ella" se refiere a "María".

Finalmente, el análisis pragmático considera la intención del hablante y cómo el contexto influye en la interpretación del significado. Esto incluye la comprensión de implicaturas y actos de habla, como solicitudes, promesas y órdenes. Por ejemplo, la frase "¿Puedes cerrar la puerta?" se interpreta pragmáticamente como una solicitud y no como una pregunta sobre la capacidad física del interlocutor para cerrar la puerta.

El PLN utiliza herramientas y librerías especializadas para llevar a cabo estos análisis. Una de las más conocidas es NLTK (Natural Language Toolkit), una librería de Python que proporciona herramientas para la tokenización, análisis sintáctico, etiquetado de partes del discurso y más. NLTK es ampliamente utilizada en la investigación y el desarrollo de aplicaciones de PLN.
"""

# Tokenización
tokens = word_tokenize(text)

# Etiquetado de Partes del Discurso
tags = pos_tag(tokens)

# Crear un DataFrame con los tokens y sus etiquetas
df = pd.DataFrame(tags, columns=["Token", "Etiqueta"])

# Mostrar el DataFrame en formato de tabla
print(df)
