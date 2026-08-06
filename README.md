# aurora-pm-station-corrector

Repositorio experimental para auditar el valor predictivo downstream de **Aurora Air Pollution** en PM10 y PM2.5 a escala de estación.

## Pregunta central

¿Aporta Aurora información adicional a la historia local y a CAMS en cada horizonte, y se mantiene ese valor durante episodios de alta contaminación?

## Diseño registrado

El experimento compara, bajo las mismas ventanas temporales y el mismo corrector:

- persistencia;
- Aurora crudo;
- CAMS crudo;
- historia local;
- historia local + Aurora;
- historia local + CAMS;
- historia local + Aurora + CAMS.

Las comparaciones decisivas son `local + Aurora` frente a `local-only` y `local + Aurora + CAMS` frente a `local + CAMS`. Mejorar únicamente Aurora crudo no demuestra valor incremental de Aurora.

## Validación

- rolling-origin por tiempo de emisión;
- entrenamiento permitido solo cuando el `valid_time` del target ya ha ocurrido;
- preprocesamiento y umbrales ajustados solo con train;
- resultados separados para 12–120 h;
- MAE, RMSE, sesgo, skill, ratio de varianza, amplitud, sesgo de cola y métricas de eventos.

## Archivos principales

- `configs/incremental_skill.yaml`: registro del experimento y contrastes.
- `docs/experiment_protocol.md`: protocolo científico y reglas de decisión.
- `docs/data_contract.md`: esquema del dataset alineado estación–Aurora–CAMS.
- `src/validation/rolling_origin.py`: particiones temporales con protección contra leakage multihorizonte.
- `src/metrics/basic.py`: métricas continuas, amplitud y eventos.

## Estado

La infraestructura metodológica está preparada. La integración real de Aurora y CAMS sigue pendiente de disponer de forecasts alineados con las estaciones y de documentar su disponibilidad operacional.
