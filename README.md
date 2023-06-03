[![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/dev/peps/pep-0537/#schedule-first-bugfix-release)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white)
![GitHub last commit](https://img.shields.io/github/last-commit/hubertronald/TemplateDockerDjango?style=flat-square)
[![MIT](https://img.shields.io/github/license/hubertronald/TemplateDockerDjango?style=flat-square)](LICENSE)

# Optimización de Portafolio en Python - Modelo de Markowitz

El desarollo contenida en este repositorio está orientado hacia la [Teoría del portafolio moderna - MPT](https://es.wikipedia.org/wiki/Teor%C3%ADa_del_portafolio_moderna)  la cual si bien en un inicio implica sólo la minimización de la varianza de un protafolio seleccionado (Modelo de Markovitz), acá se incluye además la maximización del rendimiento del mismo por medio del [Ratio de Sharpe](https://es.wikipedia.org/wiki/Ratio_de_Sharpe).


## El Problema del Portafolio Media-Varianza
---
Lo que se menciona a continuación son algunas consideraciones que se deben tener en cuenta al momento de aplicar el MTP, las cuales han sido tomadas de la referencia [[9]](https://arxiv.org/abs/2208.07158).

EL MPT hace varias llaves supuestos que se deben tener en cuenta antes de utilizar la optimización de media-varianza:

- El riesgo del portafolio está basado en la volatilidad de sus retornos como por ejemplo la fluctuación de precios.

- El análisis es conducido a un simple periodo de inversión $^{a}$.

- Los inversores son racionales, adversos al riesgo y deseosos de aumentar la rentabilidad. En consecuencia, la utilidad La función es cóncava y creciente.

- Los inversores buscan maximizar el rendimiento de su cartera para un determinado nivel de riesgo o minimizar su riesgo para un rendimiento dado.

En tal sentido desde una perpesctiva matemática, dado un portafolio $p$ con $n$ activos, se puede calcular la desviación estándar como:


$$\sigma_p=\sqrt{\sigma_{p}^{2}}$$

Siendo la variancia:

$$\sigma_{p}^{2}=\sum_{i=1}^{n}\sum_{j=1}^{n}w_iw_jCov(r_i,r_j)=w^T\Sigma w$$

Y el retorno esperado:

$$\mu_p = E(r_p)=\sum_{i=1}^{n}w_iE(r_i)=w^T \bar{r}$$

La variable $w$ denota los pesos individuales de cada activo, mientras que la variable $\mu_p$ el retorno del portafolio. Por otro lado, $\Sigma$ es la matriz de covarianza para los retornos `diarios` correspondiente a los $n$ activos que conforman el portafolio.

> [a]: A veces se le suele combinar con el [modelo de Black–Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) para predecir los precios de algunos tipos de [Derivados Financieros](https://es.wikipedia.org/wiki/Derivado_financiero) y con ello obtener los rendimientos esperados futuros.

### Problema Media-Varianza sin Activo Libre de Riesgo
---
El punto donde la frontera eficiente cambia de convexa a cóncava es donde la mínima varianza cae. Esta asignación de cartera tiene una solución única que se puede encontrar resolviendo un problema de optimización cuadrática simple a través de métodos multiplicadores de Lagrange estándar. la optimización del problema se puede formular como:

$$
\begin{aligned}
\min\limits_{w}     &\textbf{ }\textbf{ }\textbf{ }\textbf{ } \dfrac{1}{2}w^T\Sigma w \\
s.t.                &\textbf{ }\textbf{ }\textbf{ }\textbf{ } w^r=\mu_p \\
                    &\textbf{ }\textbf{ }\textbf{ }\textbf{ } \textbf{1}^Tw = \textbf{1} \\
\end{aligned}
$$

### Problema Portafolio Tangente
---
El portafolio tangente es la asignación de activos que maximiza el índice de Sharpe. Este mide el exceso de rendimiento ganado sobre la tasa libre de riesgo por unidad de volatilidad o riesgo total, lo cual ayuda a los inversores a comprender mejor el retorno de su inversión. Se puede formular como:

$$\text{Sharpe ratio}=\dfrac{E(r_p)-r_f}{\sigma_p}$$

Siendo $r_f$ es la tasa libre de riesgo, como por ejemplo los bonos del tesoro de Estados Unidos. Por tanto, la optimización del problema se puede formular como:

$$
\begin{aligned}
\max\limits_{w}     &\textbf{ }\textbf{ }\textbf{ }\textbf{ } \dfrac{w^T\bar{r}-r_f}{w^T\Sigma w} \\
s.t.                 &\textbf{ }\textbf{ }\textbf{ }\textbf{ } \textbf{1}^Tw = \textbf{1} \\
\end{aligned}
$$

Gráficamente, es el punto donde una línea recta a través de la $r_f$ es tangente a la frontera eficiente, en el Espacio modelo de Markowitz.

## Datos
---
Las fuentes fueron dos `DataHub.io` y `Alpaca Markets`, las cuales no son las únicas existiendo alternativas como [yfinance](https://pypi.org/project/yfinance/), [pandas-datareader](https://pypi.org/project/pandas-datareader/) entre otras en `python`
o en wikipedia se ofrece también un [listado de las compañias del S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

### S&P 500

Del siguiente sitio web [DataHub.io](https://datahub.io/core/s-and-p-500-companies), se obtuvo el listado de compañias que conforma el índice bursátil de referencia mundial S&P 500

```bash
curl -L https://datahub.io/core/s-and-p-500-companies/r/0.csv > ./data/s_and_p_500_companies.csv
```

### Alpaca Markets

Los `precios de cierre` para las compañías que conforma el índice bursátil S&P 500 se han obtenido (u obtienen) a través del API para el comercio de acciones y criptomonedas que ofrece [Alpaca Markets](https://alpaca.markets/)

## Instalación
---
Clonar o descargar el repo e instalar

```bash
pip install -r requirements.txt
```

## Autores
---
* **Hubert Ronald** - *Trabajo Inicial* - [HubertRonald / ModeloMarkowitz](https://github.com/HubertRonald/ModeloMarkowitz)

Ve también la lista de [contribuyentes](https://github.com/HubertRonald/ModeloMarkowitz/contributors) que participaron en este proyecto.


## Referencias
---
[1] [Derivados Financieros](https://es.wikipedia.org/wiki/Derivado_financiero)

[2] [Markowitz Model](https://en.wikipedia.org/wiki/Markowitz_model)

[3] [Efficient Frontier](https://en.wikipedia.org/wiki/Efficient_frontier)

[4] [Capital Market Line](https://en.wikipedia.org/wiki/Capital_market_line)

[5] [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)

[6] [Risk Aversion](https://en.wikipedia.org/wiki/Risk_aversion)

[7] [Demystifying Portfolio Optimization with Python and CVXOPT](https://druce.ai/2020/12/portfolio-opimization)

[8] [Portfolio Optimisation](https://quantpy.com.au/python-for-finance/portfolio-optimisation/)

[9] [Asset Allocation: From Markowitz to Deep Reinforcement Learning](https://arxiv.org/abs/2208.07158) $^{b}$

[10] [Dynamic Portfolio Optimization with Real Datasets Using Quantum Processors and Quantum-Inspired Tensor Networks](https://arxiv.org/abs/2007.00017) $^{b}$

[11] [Black–Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

> [b] Se menciona como futura referencia

## Licencia
---
Este proyecto está bajo licencia MIT - ver la [LICENCIA](LICENSE) archivo (en inglés) con más detalles
