# Evaluamos usando Bertscore, esta biblioteca ademas devuelve los valores de Recall, Precision y F1 Score

from evaluate import load

bertscore = load("bertscore")

predicciones = []
referencias = []

# Cargamos en una lista las preguntas y sus respuestas asociadas.
# Además, predecimos una respuesta usando la pregunta como argumento.

for pregunta, respuesta in dataset:
  prediccion = model(pregunta)
  predicciones.append(prediccion)
  referencias.append(respuesta)

 

results = bertscore.compute(predictions=predicciones, references=referencias, model_type="distilbert-base-uncased")
print(results)

