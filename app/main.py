from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import recommend
from app.model.model import __version__ as model_version
from app.functions.functions import movies_per_month, \
                                    movies_per_day, \
                                    title_score,\
                                    title_votes,\
                                    get_actor_info, get_director_info

# -----------------------------------------------------------------------------------
#                            API Aspects
# -----------------------------------------------------------------------------------
# Create an instance of App object
app = FastAPI()


# Base classes to set input and output data type
class TextIn(BaseModel):
    text: str

class RecommendationOut(BaseModel):
    recommendations: list

class ConsultOut(BaseModel):
    answer: dict



# -----------------------------------------------------------------------------------
#                            Health check
# -----------------------------------------------------------------------------------
@app.get("/")
def home():
    return{"Health_check": "OK", "model version": model_version}


# -----------------------------------------------------------------------------------
#                       Consult Movies DataFrame
# -----------------------------------------------------------------------------------

@app.get('/cantidad_filmaciones_mes/{mes}', response_model=ConsultOut)
def cantidad_filmaciones_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    answer = movies_per_month(mes)
    return answer

@app.get('/cantidad_filmaciones_dia{dia}', response_model=ConsultOut)
def cantidad_filmaciones_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    answer = movies_per_day(dia)
    return answer

@app.get('/score_titulo/{titulo}', response_model=ConsultOut)
def score_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''
    answer = title_score(titulo)
    return answer

@app.get('/votos_titulo/{titulo}', response_model=ConsultOut)
def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
    La misma variable deberá de contar con al menos 2000 valoraciones, 
    caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''
    answer = title_votes(titulo)
    return answer

@app.get('/get_actor/{nombre_actor}', response_model=ConsultOut)
def get_actor(nombre_actor:str):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, la cantidad de películas que en las que ha participado y el promedio de retorno'''
    answer = get_actor_info(nombre_actor)
    return answer

@app.get('/get_director/{nombre_director}', response_model=ConsultOut)
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
    answer = get_director_info(nombre_director)
    return answer

print(get_director('Steven Spielberg'))

# -----------------------------------------------------------------------------------
#                   ML - Recommendation System
# -----------------------------------------------------------------------------------

@app.get('/recomendacion/{titulo}', response_model=RecommendationOut)
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    recommendations = recommend(titulo)
    return {'lista recomendada': recommendations}
