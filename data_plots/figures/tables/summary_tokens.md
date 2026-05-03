# Average tokens per graded question

Sum of `eval_input_tokens + eval_output_tokens`. Bold = lowest in each (dataset, column).

| Dataset | Memory | gpt-4.1-mini k=5 | gpt-4.1-mini k=10 | gpt-4.1-mini k=15 | haiku-4.5 k=5 | haiku-4.5 k=10 | haiku-4.5 k=15 |
|---|---|---|---|---|---|---|---|
| Coexisting facts | mem0 | **1381** | **1630** | **1830** | **1429** | **1670** | **1879** |
|  | simplemem | 1760 | 2119 | 2402 | 1652 | 1867 | -- |
|  | amem | -- | 3635 | 4614 | 2642 | 3682 | -- |
| Conditional | mem0 | **763** | **881** | -- | -- | -- | -- |
|  | simplemem | 806 | 987 | -- | -- | -- | -- |
|  | amem | 2228 | 3797 | -- | -- | -- | -- |
| Conditional (hard) | mem0 | **953** | **1085** | **1205** | -- | -- | -- |
|  | simplemem | 979 | 1196 | 1400 | -- | -- | -- |
|  | amem | 3060 | 5361 | 7544 | -- | -- | -- |
| Persona retrieval | mem0 | 1145 | **1285** | **1465** | -- | -- | -- |
|  | simplemem | **1098** | 1371 | 1583 | -- | -- | -- |
|  | amem | 4303 | 7252 | 10037 | -- | -- | -- |
