# Proyecto Final - Bank Marketing

Tema: Analisis predictivo de aceptacion de depositos bancarios a plazo mediante campanas de telemarketing.

Dataset principal: Bank Marketing Dataset, UCI Machine Learning Repository.

Link: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

Archivo local usado en el notebook:

```text
data/bank/bank-full.csv
```

## Como abrirlo en Visual Studio Code

1. Abre la carpeta `bank_marketing_project` en Visual Studio Code.
2. Instala las extensiones de Python y Jupyter si VS Code las solicita.
3. Crea un entorno virtual o usa tu Python local.
4. Instala dependencias:

```powershell
pip install -r requirements.txt
```

5. Abre el notebook:

```text
notebooks/01_bank_marketing_analisis.ipynb
```

## Estructura

- `data/`: datasets descargados o procesados.
- `notebooks/`: notebook principal del proyecto.
- `dashboard/`: codigo del dashboard en Dash/Plotly.
- `reports/`: articulo, imagenes y material del informe.

## Ejecutar dashboard

Desde esta carpeta:

```powershell
pip install -r requirements.txt
python -m dashboard.app
```

URL local:

```text
http://127.0.0.1:8050
```

Para publicarlo en Render puede usarse `render.yaml`, o configurar manualmente:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn dashboard.app:server`

## Instalar en PythonAnywhere

PythonAnywhere puede quedarse sin cuota si se instalan librerias pesadas del notebook. Para publicar solo el dashboard, usa:

```bash
pip install -r requirements-pythonanywhere.txt
```

El dashboard usa archivos geograficos preparados en `data/map/`, por lo que no necesita instalar GeoPandas en el servidor. GeoPandas se uso durante la preparacion local de los datos del mapa.

## Nota sobre datos faltantes

El dataset no trae valores faltantes explicitos segun UCI. Sin embargo, contiene categorias `unknown` en algunas columnas. Para cumplir el analisis solicitado, estas categorias se trataran como datos no informados y se convertiran a `NaN` durante la limpieza.

Columnas con `unknown` en `bank-full.csv`:

- `poutcome`: 36959 registros.
- `contact`: 13020 registros.
- `education`: 1857 registros.
- `job`: 288 registros.

## Nota importante

El archivo `adult.zip` descargado aparte no se usa en este proyecto porque corresponde a Adult/Census Income, dataset que ya fue elegido por otro grupo.
