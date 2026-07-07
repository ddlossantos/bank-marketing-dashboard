# Dashboard de Resultados

App: Dash + Plotly.

Archivo principal:

```text
dashboard/app.py
```

## Ejecutar localmente

Desde la carpeta principal `bank_marketing_project`:

```powershell
pip install -r requirements.txt
python -m dashboard.app
```

Luego abrir:

```text
http://127.0.0.1:8050
```

## Cumplimiento del entregable

- Mas de 4 graficas de analisis:
  - Tasa de aceptacion por ocupacion.
  - Distribucion de edad por aceptacion.
  - Duracion de llamada segun aceptacion.
  - Correlacion de variables numericas.
  - Mapa coropletico de poblacion por distrito.
  - Top de distritos por poblacion.
- Mas de 2 controladores interactivos:
  - Filtro por ocupacion.
  - Rango de edad.
  - Mes.
  - Hipoteca.
  - Prestamo.
  - Slider del top de distritos.
- Control interactivo de regresion:
  - Usa Random Forest Regressor para estimar `balance`.
- Mapa interactivo:
  - Nivel geografico: distritos de Panama.
  - Variables seleccionables del INEC MAPI:
    - Poblacion total.
    - Porcentaje de poblacion menor de 15 anos.
    - Porcentaje de poblacion de 15 a 64 anos.
    - Porcentaje de poblacion de 65 y mas anos.
    - Porcentaje sin seguro social.
    - Porcentaje de analfabetas.
    - Porcentaje de desocupados.
    - Mediana del ingreso mensual del hogar.
    - Promedio de anos aprobados.
  - Fuente de datos: INEC MAPI.
  - Se utiliza GeoPandas para cargar y unir la geometria con los valores.
  - Al hacer clic en un distrito se actualiza una tarjeta con el detalle del distrito seleccionado.
- Grafica resumen del mapa:
  - Barra por provincia usando la misma variable mostrada en el mapa.
  - Para poblacion total se usa suma provincial.
  - Para porcentajes, medianas y promedios se usa promedio distrital por provincia.

## Fuentes

- Dataset principal: UCI Machine Learning Repository, Bank Marketing.
- Datos sociodemograficos: INEC MAPI, variable `Poblacion total`, id `179`, nivel `Distrito`, anio 2023.
- Geografia documentada: geoBoundaries PAN ADM2.
