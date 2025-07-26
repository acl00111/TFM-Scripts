import numpy as np
import os, json, cv2, random
import yaml
import itertools
import pandas as pd
import pathlib
import hashlib
import random
import time
from trainDetectron import run_training_pipeline
from individuo import Individuo

def readConfs(yamlPath):
    with open(yamlPath) as f:
        param_space = yaml.safe_load(f)["param_space"]

    keys = ["modalidad", "modelo", "flip", "batch_size", "gamma",
            "base_lr", "weight_decay", "maxiter_steps"]

    combinations = list(itertools.product(*(param_space[k] for k in keys)))

    list_of_dicts = []

    for combination in combinations:
        params = dict(zip(keys, combination))
        maxiter_steps = params.pop("maxiter_steps")
        params.update(maxiter_steps)
        list_of_dicts.append(params)

    return list_of_dicts

def get_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

def inicializar_poblacion(list_of_dicts, size=10):
    """
    Inicializa una población de individuos a partir de una lista de diccionarios.
    Cada diccionario representa un conjunto de hiperparámetros.
    Asegurarse de que al menos haya un individuo por cada modelo y modalidad.
    """
    modelos_vistos = set()
    modalidades_vistas = set()
    poblacion = []
    usados = {}

    print(f"--- Inicializando población de tamaño {size}... ---")

    random.shuffle(list_of_dicts)

    for config in list_of_dicts:
        if len(poblacion) >= size:
            break
        key = json.dumps(config, sort_keys=True)
        if key in usados:
            continue

        modelo = config["modelo"]
        modalidad = config["modalidad"]

        if (modelo not in modelos_vistos) or (modalidad not in modalidades_vistas):
            modelos_vistos.add(modelo)
            modalidades_vistas.add(modalidad)
            poblacion.append(Individuo(config))
            usados[get_hash(config)] = None

    # Si faltan individuos hasta completar el tamaño deseado
    while len(poblacion) < size:
        config = random.choice(list_of_dicts)
        key = json.dumps(config, sort_keys=True)
        if key in usados:
            continue
        poblacion.append(Individuo(config))
        usados[get_hash(config)] = None

    return poblacion, usados

def evaluar_poblacion(poblacion, usados):
    """
    Evalua cada individuo de la población.
    """
    for individuo in poblacion:
        print(f"Evaluando {individuo}")
        individuo.fitness = run_training_pipeline(individuo.as_dict())
        usados[get_hash(individuo.as_dict())] = individuo.fitness
        print(f"Fitness de {individuo}: {individuo.fitness}")

def seleccionar_padres(poblacion, p=2, k=3):
    """
    Selecciona p padres de la población por torneo.
    """
    print(f"Seleccionando {p} padres de la población...")
    padres = []
    for _ in range(p):
        torneo = random.sample(poblacion, k)
        ganador = max(torneo, key=lambda ind: ind.fitness)
        padres.append(ganador)
    return padres

def cruzar_padres(padres):
    """
    Cruza dos padres para crear un nuevo individuo.
    """
    print("Cruzando padres...")
    padre1, padre2 = padres[0], padres[1]
    hijo_params = {k: random.choice([padre1[k], padre2[k]]) for k in padre1.params.keys()}
    hijo = Individuo(hijo_params)
    return hijo

def mutar_hijo(hijo, mutation_rate=0.1):
    """
    Aplica mutaciones al hijo con una probabilidad de mutation_rate.
    """
    for key in hijo.params.keys():
        if random.random() < mutation_rate:
            if key == 'flip':
                print(f"Mutando {key} del hijo: {hijo[key]}")
                hijo[key] = random.choice(['horizontal', 'vertical', 'none'])
            elif key == 'modalidad':
                print(f"Mutando {key} del hijo: {hijo[key]}")
                if hijo[key] == 'FLAIR':
                    hijo[key] = 'RGB'
                else:
                    hijo[key] = 'FLAIR'
            elif key == 'modelo':
                print(f"Mutando {key} del hijo: {hijo[key]}")
                hijo[key] = random.choice(['mask_rcnn_R_50_FPN_3x.yaml',
                        'mask_rcnn_R_101_FPN_3x.yaml',
                        'mask_rcnn_R_50_C4_3x.yaml', 
                        'mask_rcnn_R_101_C4_3x.yaml',
                        'mask_rcnn_R_50_DC5_3x.yaml',
                        'mask_rcnn_R_101_DC5_3x.yaml',
                        'mask_rcnn_1_101_32x8d_FPN_3x.yaml'])
            elif key == 'batch_size':
                print(f"Mutando {key} del hijo: {hijo[key]}")
                hijo[key] = random.choice([2, 4])
            else:
                print(f"Mutando {key} del hijo: {hijo[key]}")
                hijo[key] *= random.uniform(0.9, 1.1)  # Pequeña variación
    print(f"Hijo mutado: {hijo}")

def reemplazar_individuo(poblacion, hijo, num_reemplazables=5):
    """
    Añade el hijo, ordena, y elimina uno aleatorio de los peores `num_reemplazables`.
    """
    print("Reemplazando un individuo de la población...")
    poblacion.append(hijo)
    poblacion.sort(key=lambda ind: ind.fitness, reverse=True)  # De mejor a peor

    peores = poblacion[-num_reemplazables:]
    eliminar = random.choice(peores)
    poblacion.remove(eliminar)

    print(f"Reemplazado un individuo con fitness bajo: {eliminar.fitness}")

def guardar_poblacion(poblacion, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([ind.as_dict() for ind in poblacion], f, indent=2)

def guardar_usados(usados, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(usados, f, indent=2)


def main():
    """ ind_params = {
        'modalidad': 'FLAIR',
        'modelo': 'mask_rcnn_R_101_FPN_3x.yaml',
        'flip': 'vertical',
        'batch_size': 2,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter': 5000,
        'steps': [2500]
    } """

    tiempo_inicio = time.time()
    print(f"Inicio del script: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tiempo_inicio))}")
    #run_training_pipeline(ind_params)
    yaml_path = '/home/albacano/TFM-Scripts/posiblesConfs.yaml'
    list_of_dicts = readConfs(yaml_path)

    # Inicialización Población
    poblacion, usados = inicializar_poblacion(list_of_dicts, size=10)
    iteraciones = 10
    evaluar_poblacion(poblacion, usados)
    for i in range(iteraciones):
        print(f"Iteración {i+1} de {iteraciones}")
        padres = seleccionar_padres(poblacion, p=2)
        hijo = cruzar_padres(padres)
        mutar_hijo(hijo)
        hash = get_hash(hijo.as_dict())
        if hash in usados:
            print(f"Hijo ya evaluado: {hash}, saltando...")
            hijo.fitness = usados[hash]
        else:
            hijo.fitness = run_training_pipeline(hijo.as_dict())
            usados[hash] = hijo.fitness
        reemplazar_individuo(poblacion, hijo, 3)
        guardar_poblacion(poblacion, f"/home/albacano/TFM-Scripts/AE/poblacion_iter_{i+1}.json")
        guardar_usados(usados, f"/home/albacano/TFM-Scripts/AE/usados_iter_{i+1}.json")

    mejor_individuo = max(poblacion, key=lambda ind: ind.fitness)
    tiempo_fin = time.time()
    print(f"Mejor individuo: {mejor_individuo} con fitness {mejor_individuo.fitness}")
    print(f"Fin del script: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tiempo_fin))}")
    print(f"Tiempo total de ejecución: {tiempo_fin - tiempo_inicio:.2f} segundos")
    

if __name__ == "__main__":
    main()