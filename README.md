# aurora-pm-station-corrector

Proyecto base para un nuevo estudio de PM10/PM2.5 centrado en:

- corrección local de predicciones Aurora-AQ a nivel estación
- comparación contra persistencia y baseline local
- pipeline reproducible
- separación clara entre smoke test, baseline y corrector

## Estructura

- `configs/`: configuración del experimento y modelos
- `data/`: datos raw y processed
- `docs/`: pregunta científica y límites de novelty
- `scripts/`: puntos de entrada ejecutables
- `src/`: módulos reutilizables
- `results/`: salidas de ejecución
- `tests/`: comprobaciones mínimas

## Estado

Scaffold inicial creado.  
Sin datasets cargados todavía.  
Sin integración Aurora todavía.
