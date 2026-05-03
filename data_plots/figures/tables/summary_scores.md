# Mean success rate per memory system

Bold = best score in each (dataset, column).

| Dataset | Memory | gpt-4.1-mini k=5 | gpt-4.1-mini k=10 | gpt-4.1-mini k=15 | haiku-4.5 k=5 | haiku-4.5 k=10 | haiku-4.5 k=15 |
|---|---|---|---|---|---|---|---|
| Coexisting facts | mem0 | **0.350** | **0.610** | **0.690** | **0.370** | **0.640** | **0.760** |
|  | simplemem | 0.070 | 0.200 | 0.340 | 0.100 | 0.220 | -- |
|  | amem | -- | 0.510 | 0.610 | 0.230 | 0.420 | -- |
| Conditional | mem0 | 0.580 | 0.680 | -- | -- | -- | -- |
|  | simplemem | 0.850 | **0.930** | -- | -- | -- | -- |
|  | amem | **0.950** | 0.930 | -- | -- | -- | -- |
| Conditional (hard) | mem0 | 0.130 | 0.110 | 0.110 | -- | -- | -- |
|  | simplemem | 0.270 | **0.310** | 0.260 | -- | -- | -- |
|  | amem | **0.350** | 0.290 | **0.360** | -- | -- | -- |
| Persona retrieval | mem0 | 0.490 | 0.503 | 0.493 | -- | -- | -- |
|  | simplemem | 0.483 | 0.630 | 0.617 | -- | -- | -- |
|  | amem | **0.703** | **0.673** | **0.630** | -- | -- | -- |
