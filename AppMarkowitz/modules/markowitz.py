#!/usr/bin/.venv python
# -*- coding: utf-8 -*-

"""Modelo Markovitz"""

from pandera.typing import DataFrame, Series
from nptyping import NDArray
from typing import Tuple, Union

import numpy as np
import pandas as pd

import cvxopt as opt
from cvxopt import solvers, blas
import scipy.optimize._minimize as scopt

from ..utils.constants import Constants

CTE: Constants = Constants()

# Se silencia el solver (es opcional)
solvers.options['show_progress'] = False


def rendimiento_logaritmo(precios_cierre_activos: DataFrame)->DataFrame:
    """
    Convertir los precio de cierre en
    rendimiento de logaritmo

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos
    
    Returns
    -------
    DataFrame
        Contiene los rendimientos en logartimos
        de los activos

    Nota
    -------
    Podria emplearse la siguiente expresion
    np.log(1+precios_cierre_activos.pct_change()[1:])
    Al respecto hay un post donde se detalla por que
    usar rendimiento logartimicos:
    https://quantdare.com/por-que-usar-rendimientos-logaritmicos/
    """
    rendimientos_activos = precios_cierre_activos / precios_cierre_activos.shift(1) 
    return np.log(rendimientos_activos[1:])


def metricas_historicas_portafolio(
        precios_cierre_activos: DataFrame,
        dias_anual: int=252
    )->Tuple[NDArray, NDArray, int]:
    """
    Da un pequenio reporte sobre las observaciones de
    cada activo contenido en el portafolio, para 
    devolver finalmente dos matrices


    Parameters
    ----------
    `portafolio`: DataFrame
        es un dataframe de observaciones (filas)
        por activos (columnas) historico

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    Returns
    ----------
    Tuple[NDArray, NDArray, int]
        devuelve las matrices o arrays de rendimientos
        esperados y la de covarianza del portafolio historico,
        asi como el numero de activos

        `r`: NDArray 
            matriz con rendimiento esperados
            para cada activo del portafolio

        `C`: NDArray
            matriz con la covarianza historico del
            portafolio
        
        `total_activos`: int
            activos que se incluyen en el portafolio
    """

    # Metricas Historicas
    rendimientos_activos_historicos = rendimiento_logaritmo(precios_cierre_activos)
    total_activos = rendimientos_activos_historicos.shape[1]
    rendimiento_anual = rendimientos_activos_historicos.mean() ** (dias_anual-1)
    covarianza_anual = rendimientos_activos_historicos.cov() ** (dias_anual-1)

    # Matrices con las Estadisticas Hitoricas de Interes
    # la variable `C`es mayuscula 

    r = rendimiento_anual.to_numpy()
    C = covarianza_anual.to_numpy()

    return (r, C, total_activos)


def calcular_ratio_sharpe(
        rendimiento: Union[NDArray, float], 
        volatilidad: Union[NDArray, float], 
        libre_riesgo:float=0
    )->Union[NDArray, float]:
    """
    Ratio de Sharpe
 
    Parameters
    ----------
    `redimiento` (mu): NDArray | float
        esperado del portafolio

    `volatilidad` (sigma o desviacion estandar): NDArray | float
        del portafolio
    
    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1     

    Returns
    ----------   
    NDArray | float :
        El ratio de sharpe

    Referencias
    ----------
    [1] https://en.wikipedia.org/wiki/Sharpe_ratio
    """
    retorno = rendimiento - libre_riesgo
    if isinstance(rendimiento, float) and isinstance(volatilidad, float):
        if volatilidad == 0:
            return 0
        return retorno / volatilidad
    else:
        return np.divide(retorno, volatilidad, out=np.zeros_like(retorno), where=volatilidad!=0)


def resultados_portafolio(
        r: NDArray,
        C: NDArray,
        w: NDArray=np.array([]),
        libre_riesgo:float=0
    )->Tuple[float, float, float]:
    """
    Dados unos pesos de colocacion para un
    portafolio y teniendose los rendimientos y
    covarianzas historicas, se obtiene el
    rendimiento y volatilidad del portafolio

    Parameters
    ----------
    `r`: NDArray 
        matriz con rendimientos esperados
       del portafolio

    `C`: NDArray
        matriz con la covarianza historico del
        portafolio
    
    `w`: NDArray 
        peso que se empleara para colocar los
        los fondos en los activos correspondientes
        del portafolio
        Si no se coloca nada (inversionista ingenuo)
        se le asigna el mismo peso a todos sumando 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1

    Returns
    ----------   
    Tuple[NDArray, NDArray]: 
        el redimiento `mu` 
        y desviacion estandar (volatilidad) `sigma`
        del portafolio
    """
    w = np.asarray(w)
    if w.shape == (0,):
        n: int = r.shape[0]
        w = np.repeat(1./n, n)

    mu = w.T @ r                       # Rendimiento o Retorno Esperado
    sigma = np.sqrt(w.T @ C @ w)        # Volatilidad

    sharpe_ratio = calcular_ratio_sharpe(mu, sigma, libre_riesgo)
    return (mu, sigma, sharpe_ratio)


def simular_pesos_portafolio(
        numero_de_activos: int, 
        interval: Tuple[int, int]=(0,1)
    )->NDArray:
    """ 
    Generar pesos aleatorios para cada
    activo en el portafolio

    Parameters
    ----------
    `numero_de_activos`: int
        es entero y en la otras functiones recibe
        el nombre de `n`
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1, se recomienda el siguiente
        rango -1 <= interval <= 1

    Returns
    ----------  
    NDArray:
        El peso de cada uno de los 
        activos en el portafolio cuya suma es 1
        como matriz
    """
    pesos = np.array([])
    while pesos.sum() != 1.0:
        pesos = np.random.uniform(
            low=interval[0],
            high=interval[1],
            size=numero_de_activos
        )
        if np.isnan(pesos).sum() == 0 and pesos.sum()!=0:
            pesos /= pesos.sum()

    return pesos


def simular_portafolio(
        precios_cierre_activos: DataFrame, 
        dias_anual: int=252,
        interval: Tuple[int, int]=(0,1),
        libre_riesgo: float=0.,
        limite_volatilidad: float=1.
    )->Tuple[NDArray, float, float, float]:
    """
    Genera los rendimientos y volatidades para un conjunto
    de portafolios

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1

    `limite_volatilidad`: int (opcional)
        es para mantener la volatilidad hasta un tope durante la
        simulacion
    
    Returns
    ----------  
    Tuple[NDArray, NDArray, NDArray, NDArray:
        los pesos, rendimientos esperados así
        como las volatidades `desviacion estandar`
        para cada uno de los portafolios simulados
    """

    r, C, n = metricas_historicas_portafolio(precios_cierre_activos, dias_anual)
    w = simular_pesos_portafolio(n, interval)
    mu, sigma, sharpe_ratio = resultados_portafolio(r, C, w, libre_riesgo)
 
    # Esta recursividad reduce los valores atípicos
    # para mantener el portafolio de interés
    # esto se puede desarrollar tambien con un centinela 
    # dentro de un ciclo `while`
    if sigma > limite_volatilidad:
        return simular_portafolio(
        precios_cierre_activos, 
        dias_anual,
        interval,
        libre_riesgo,
        limite_volatilidad
    )
    
    return (w, mu, sigma, sharpe_ratio)


def simulacion_de_portafolios(
        precios_cierre_activos: DataFrame, 
        dias_anual: int=252,
        interval: Tuple[int, int]=(0,1),
        libre_riesgo: float=0.,
        limite_volatilidad: float=1.,
        numero_de_portafolios: int = 1000
    )->DataFrame:
    """
    Genera los rendimientos y volatidades para un conjunto
    de portafolios

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1

    `limite_volatilidad`: int (opcional)
        es para mantener la volatilidad hasta un tope durante la
        simulacion

    `numero_de_portafolios`: int (opcional)
        entero que indica la cantidad de replicas
        o simulaciones a efectuarse

    Returns
    ----------  
    DataFrame:
        los rendimientos esperados así
        como las volatidades `desviacion estandar`
        para cada uno de los portafolios simulados,
        su ratio de sharpe y los pesos empleados.
    """

    pesos_simu, rendimientos_simu, volatilidades_simu, ratio_sharpe_simu = zip(*[
        simular_portafolio(
            precios_cierre_activos,
            dias_anual,
            interval,
            libre_riesgo,
            limite_volatilidad)
        for _ in range(numero_de_portafolios)
    ])

    portafolios_simulados: DataFrame = pd.DataFrame({
        'rendimiento': rendimientos_simu, 
        'volatilidad': volatilidades_simu, 
        'ratio_sharpe': ratio_sharpe_simu, 
        'pesos': pesos_simu
    })
    
    return portafolios_simulados


def portafolios_frontera(
        precios_cierre_activos: DataFrame, 
        dias_anual: int=252,
        interval: Tuple[int, int]=(0,1),
        libre_riesgo: float=0.
    )->Union[DataFrame, DataFrame]:
    """
    Genera los puntos para la Frontera Eficiente

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1

    Returns
    ---------- 
    DataFrame | DataFrame
        Devuelve dos dataframes

        `frontera eficiente`: 
            rendimientos esperados,
            volatidades `desviacion estandar`
            ratio de sharpe
            pesos de los portafolios
        `portafolio maximo sharpe frontera`: 
            rendimientos esperados,
            volatidades `desviacion estandar`
            ratio de sharpe
            pesos del portafolio con maximo sharpe
            de la frontera de eficiencia
        `portafolio maximo rendimiento frontera`: 
            rendimientos esperados,
            volatidades `desviacion estandar`
            ratio de sharpe
            pesos del portafolio con maximo rendimiento
            de la frontera de eficiencia

    Referencias
    -------
    Resolviendo el modelo cuadratico
    [1] http://cvxopt.org/userguide/coneprog.html
    [2] http://cvxopt.org/examples/book/portfolio.html
    [3] http://cvxopt.org/examples/tutorial/qp.html
    [4] https://druce.ai/2020/12/portfolio-opimization
    [5] https://towardsdatascience.com/quadratic-optimization-with-constraints-in-python-using-cvxopt-fc924054a9fc
    [6] https://stackoverflow.com/questions/48420021/cvxopt-for-markowitz-portfolio-optimization-finding-point-of-optimal-sharpe-ra
    """

    # Se establece saltos discretos para hallar la
    # la frontera eficiente estos seran los 
    # `targets` u objetivos que se fijan para optimizar
    N: int = 200
    r, C, n = metricas_historicas_portafolio(precios_cierre_activos, dias_anual)
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    
    # convertir el p y C a matrices del tipo cvxopt
    # en el caso de p se trabaja con su transpuesta
    pbar = opt.matrix(r.T)
    S = opt.matrix(C)

    # Crear las matrices de restricciones
    # Gx <= h
    G = -opt.matrix(np.eye(n))          # matriz identidad negativa n x n 
    h = opt.matrix(0.0, (n ,1))

    # Ax = b
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)       # La suma de los pesos es 1

    # Calcular los pesos de la frontera eficiente
    # Empleando Programacion Cuadratica
    pesos_frontera = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    # Calcular los rendimientos y volatilidades o riesgos
    # para la frontera eficiente
    rendimientos_frontera = [blas.dot(pbar, x) for x in pesos_frontera]
    volatilidades_frontera = [np.sqrt(blas.dot(x, S*x)) for x in pesos_frontera]
    ratios_sharpe_frontera = calcular_ratio_sharpe(
        np.asarray(rendimientos_frontera), np.asarray(volatilidades_frontera)
    )

    ## Calcular el polinomio de 2do grado de la frontera de eficiencia
    m1 = np.polyfit(x=rendimientos_frontera, y=volatilidades_frontera, deg=2)
    x1 = np.sqrt(m1[2] / m1[0])

    # Calcular el portafolio con el rendimiento maximo
    peso_max = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    peso_max = np.asarray(peso_max).reshape(1,-1)[0] # convert matrix to numpy

    rendimiento_max, volatilidad_max, ratio_sharpe_max = \
        resultados_portafolio(p, C, peso_max, libre_riesgo)

    portafolios_frontera: DataFrame = pd.DataFrame({
        'rendimiento': rendimientos_frontera, 
        'volatilidad': volatilidades_frontera, 
        'ratio_sharpe': ratios_sharpe_frontera, 
        'pesos': pesos_frontera
    })

    portafolio_max: DataFrame = pd.DataFrame({
        'rendimiento': rendimiento_max, 
        'volatilidad': volatilidad_max, 
        'ratio_sharpe': ratio_sharpe_max, 
        'pesos': [peso_max]
    })

    mask_sharpe_max: Series = portafolios_frontera.ratio_sharpe.max() == portafolios_frontera.ratio_sharpe
    portafolio_sharpe_max: DataFrame = portafolios_frontera.loc[mask_sharpe_max, :]

    return (portafolios_frontera, portafolio_sharpe_max, portafolio_max)


def sharpe_negativo(
        w: NDArray,
        r: NDArray,
        C: NDArray,
        libre_riesgo: float = 0
    )->Union[NDArray, float]:
    """"""
    _, _, sharpe = resultados_portafolio(r,C,w,libre_riesgo)
    return -1*sharpe


def ratio_sharpe_optimo(
        precios_cierre_activos: DataFrame, 
        dias_anual: int=252,
        interval: Tuple[int, int]=(0,1),
        libre_riesgo: float=0.
    )->scopt.OptimizeResult:
    """
    Genera los puntos para la Frontera Eficiente

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1
    """
    r, C, n = metricas_historicas_portafolio(precios_cierre_activos, dias_anual)

    args = (r, C, libre_riesgo)
    restricciones = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    rangos_pesos = tuple(interval for _ in range(n))

    solve: scopt.OptimizeResult = scopt.minimize(
        sharpe_negativo,
        simular_pesos_portafolio(n),
        args=args,
        method='SLSQP',
        bounds=rangos_pesos,
        constraints=restricciones
    )

    return solve


def varianza_portafolio(
        w: NDArray,
        r: NDArray,
        C: NDArray,
        libre_riesgo: float = 0
    )->Union[NDArray, float]:
    """"""
    _, sigma, _ = resultados_portafolio(r,C,w,libre_riesgo)
    return sigma


def minimizar_varianza(
        precios_cierre_activos: DataFrame, 
        dias_anual: int=252,
        interval: Tuple[int, int]=(0,1),
        libre_riesgo: float=0.
    )->scopt.OptimizeResult:
    """
    Genera los puntos para la Frontera Eficiente

    Parameters
    ----------
    `precios_cierre_activos` : DataFrame
        Contiene los precio de cierre por día para cada uno
        de los activos

    `dias_anual`: int (opcional)
        por defecto es 252 dias.
        es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias
    
    `interval`: Tuple[int, int] (opcional)
        rango en el que se moveran los pesos,
        pero estos serán ajustados luego
        para que sumen 1

    `libre de riesgo`: float (opcional)
        flotante que va de 0 a 1
    """
    r, C, n = metricas_historicas_portafolio(precios_cierre_activos, dias_anual)
    args = (r, C, libre_riesgo)
    restricciones = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    rangos_pesos = tuple(interval for _ in range(n))

    solve: scopt.OptimizeResult = scopt.minimize(
        varianza_portafolio,
        simular_pesos_portafolio(n),
        args=args,
        method='SLSQP',
        bounds=rangos_pesos,
        constraints=restricciones
    )

    return solve


if __name__ == '__main__':
    df_2 = pd.read_csv(CTE.DATA_STOCK_PRICE, parse_dates=True, index_col="date")
    df_frontera, df_sharpe_max, _ = portafolios_frontera(precios_cierre_activos=df_2)
    print(df_sharpe_max)
