# Bank Marketing Dataset

Fuente: UCI Machine Learning Repository

Link: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

Archivo usado:

```text
data/bank/bank-full.csv
```

Dimensiones:

- Filas: 45211
- Columnas: 17

Variable objetivo para clasificacion:

- `y`: indica si el cliente acepto (`yes`) o no acepto (`no`) el deposito bancario a plazo.

Variable objetivo sugerida para regresion:

- `balance`: balance anual promedio del cliente.

Columnas del dataset:

- `age`
- `job`
- `marital`
- `education`
- `default`
- `balance`
- `housing`
- `loan`
- `contact`
- `day`
- `month`
- `duration`
- `campaign`
- `pdays`
- `previous`
- `poutcome`
- `y`

Valores no informados:

El dataset no incluye valores faltantes explicitos, pero usa la categoria `unknown` en algunas columnas. En el notebook, esos valores se convierten a `NaN` para poder realizar el analisis de valores faltantes solicitado.

Conteo de `unknown`:

- `poutcome`: 36959
- `contact`: 13020
- `education`: 1857
- `job`: 288

Decision de limpieza inicial:

- `poutcome` se elimina para modelado por tener una proporcion muy alta de datos no informados.
- `contact`, `education` y `job` se conservan y se imputan dentro de los pipelines.

Nota:

El archivo `adult.zip` no se utiliza en este proyecto porque corresponde a Adult/Census Income, dataset que ya fue seleccionado por otro grupo.
