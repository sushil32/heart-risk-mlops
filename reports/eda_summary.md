# EDA summary

- Raw rows: 303, cleaned rows: 303
- Missing cells in raw data: 6 (ca=4, thal=2)
- Class balance: {0: 164, 1: 139} (positive rate 45.9%)

## Strongest Spearman correlations with target

|          |    rho |
|:---------|-------:|
| thal     |  0.522 |
| ca       |  0.489 |
| cp       |  0.472 |
| exang    |  0.432 |
| thalach  | -0.423 |
| oldpeak  |  0.413 |
| slope    |  0.364 |
| sex      |  0.277 |
| age      |  0.237 |
| restecg  |  0.169 |
| trestbps |  0.128 |
| chol     |  0.121 |
| fbs      |  0.025 |

## Numeric summary

|          |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:---------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| age      |     303 |  54.44 |  9.04 |    29 |  48   |  56   |  61   |  77   |
| trestbps |     303 | 131.69 | 17.6  |    94 | 120   | 130   | 140   | 200   |
| chol     |     303 | 246.69 | 51.78 |   126 | 211   | 241   | 275   | 564   |
| thalach  |     303 | 149.61 | 22.88 |    71 | 133.5 | 153   | 166   | 202   |
| oldpeak  |     303 |   1.04 |  1.16 |     0 |   0   |   0.8 |   1.6 |   6.2 |
| ca       |     299 |   0.67 |  0.94 |     0 |   0   |   0   |   1   |   3   |
