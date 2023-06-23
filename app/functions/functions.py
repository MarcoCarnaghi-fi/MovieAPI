import pickle
import re
from pathlib import Path
import pandas as pd


#Get base directory path
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve(strict = True).parent



# Load pickle files
# -------------------------------------------------------------
'''

with open(f"{BASE_DIR}/df_movies_dict.pkl","rb") as f:
    temp_dict = pickle.load(f)

df_movies = pd.DataFrame(temp_dict)


with open(f"{BASE_DIR}/df_cast_dict.pkl","rb") as f:
    temp_dict = pickle.load(f)

df_cast = pd.DataFrame(temp_dict)


with open(f"{BASE_DIR}/df_crew_dict.pkl","rb") as f:
    temp_dict = pickle.load(f)

df_crew = pd.DataFrame(temp_dict)
'''
df_movies = pd.read_csv(f'{BASE_DIR}/datasets/df_movies_clean.csv')
df_cast = pd.read_csv(f'{BASE_DIR}/datasets/df_cast.csv')
df_crew = pd.read_csv(f'{BASE_DIR}/datasets/df_crew.csv')


# Define Functions/Methods
# -------------------------------------------------------------

# Movies per month
def movies_per_month( mes:str ):
  '''
  Se ingresa un mes en idioma Español.
  Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.

  Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X
  '''
  month_map = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
  }

  month_number = month_map.get(mes.lower())

  # Set the release_date column to a datetime format
  df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%m-%d')
  # Filter the movies that were released in the specified month and count them
  month_movies = df_movies[df_movies['release_date'].dt.month == month_number]
  month_count = len(month_movies)

  answer = {'mes':mes.lower(), 'cantidad':month_count}
  return answer

# Movies per day
def movies_per_day( dia:str ):
  '''
  Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.
                    Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X
  '''
  day_map = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'sabado': 'Saturday',
    'domingo': 'Sunday',
  }

  day = day_map.get(dia.lower())
  # Set the release_date column to a datetime format
  df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%m-%d')

  # Count the number of movies that were released on Mondays
  day_count = len(df_movies[df_movies['release_date'].dt.day_name() == day])

  answer = {'dia':dia.lower(), 'cantidad':day_count}
  return answer

# TITLE SCORE
def title_score( titulo:str ):
  '''
  Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
  Ejemplo de retorno: La película X fue estrenada en el año X con un score/popularidad de X
  '''

  # Filter the DataFrame to include only the row(s) that match the title
  movie = df_movies[df_movies['title'].str.lower() == titulo.lower()]

  # Extract the values for the title, release_year, and popularity columns
  movie_title = movie['title'].values[0]
  release_year = int(movie['release_year'].values[0])
  popularity = float(movie['popularity'].values[0])

  # Return the movie title, release year, and popularity
  answer = {'titulo':movie_title, 'anio':release_year, 'popularidad':popularity}
  return answer


# TITLE VOTES
def title_votes(titulo:str): 
  '''
  Se ingresa el título de una filmación esperando como respuesta:
  * el título,
  * la cantidad de votos
  * el valor promedio de las votaciones

  La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.

  Ejemplo de retorno: La película X fue estrenada en el año X. La misma cuenta con un total de X valoraciones, con un promedio de X
  '''

  # Filter the DataFrame to include only the row(s) that match the title
  movie = df_movies[df_movies['title'].str.lower() == titulo.lower()]

  # Extract the values for the title, release_year, and popularity columns
  movie_title = movie['title'].values[0]
  release_year = int(movie['release_year'].values[0])
  vote_average = float(movie['vote_average'].values[0])
  vote_count = float(movie['vote_count'].values[0])

  if vote_count >= 2000.0:
    # Return the movie title, release year, and votes
    answer = {'titulo':movie_title,
              'anio':release_year,
              'voto_total':vote_count,
              'voto_promedio':vote_average}
    
  else:
      answer = {'mensaje': f'La película "{movie_title}" no cumple con la condición de tener al menos 2000 valoraciones.'}
    
    
  return answer



# GET_ACTOR
#Note, I should make more robust this code to consider that the input may not be in th df
def get_actor_info( nombre_actor:str ):

  # Filter the cast DataFrame to include only rows for the given actor
  actor_df = df_cast[df_cast['name'].str.lower() == nombre_actor.lower()]

  # Join the actor_df with the movies_df on the 'id' column

  joined_df = actor_df.merge(df_movies,
                            left_on='movie_id',
                            right_on='id')

  # Calculate the sum and average of the 'return' column
  return_sum = float(joined_df['return'].sum())
  movies_count = int(joined_df['return'].count())
  average_return = float(return_sum / movies_count)

  actor_name = actor_df['name'].iloc[0]

  answer =  {'actor':actor_name,
             'cantidad_filmaciones':movies_count,
             'retorno_total':return_sum,
             'retorno_promedio':average_return}
  return answer


# GET_DIRECTOR
def get_director_info( nombre_director:str ):
  # Filter the crew DataFrame to include only rows for the given director
  director_df = df_crew[df_crew['name'].str.lower() == nombre_director.lower()]

  director_df = director_df[director_df['job'] == 'Director']

  # Join the director_df with the movies_df on the 'id' column
  joined_df = director_df.merge(df_movies,
                          left_on='movie_id',
                          right_on='id')

  # Calculate the sum of the 'return' column
  return_sum = joined_df['return'].sum()

  # Calculate the number of movies directed by the director
  movie_count = director_df['movie_id'].count()

  # Calculate the average return per movie
  average_return = return_sum / movie_count

  # Return a DataFrame with the movie name, release date, return, budget, and revenue
  director_name = director_df['name'].iloc[0]
  movie_data_df = joined_df[['title', 'release_year', 'return', 'budget', 'revenue']]
  
  # Rename the columns
  movie_data_df.columns = ['titulo', 'anio', 'retorno_pelicula', 'budget_pelicula', 'revenue_pelicula']
  
  # Convert each row to a dictionary
  dict_list = movie_data_df.to_dict(orient='index')

  answer = {'director':director_name,
          'retorno_total_director':return_sum,
          'peliculas': dict_list}
  return answer