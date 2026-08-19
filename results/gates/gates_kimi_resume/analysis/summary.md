# Gates run summary

## Boundary calibration quality (selected items)

| provider          | gate   |   n_items |   mean_p_text |   min_p_text |   max_p_text |   n_in_band |
|:------------------|:-------|----------:|--------------:|-------------:|-------------:|------------:|
| openai_compatible | resume |        20 |       0.48125 |            0 |            1 |           4 |


## Modality effect at the boundary (reference fonts only)

| provider          | gate   |   n_items |   mean_delta_p |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:------------------|:-------|----------:|---------------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| openai_compatible | resume |        40 |      -0.152083 |        -0.152083 |   -0.239583 |      -0.075 |    0.00418091 |           0.175 |            -7 |


## Variant effects pooled over gates

| provider          | variant_id      |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:------------------|:----------------|----------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| openai_compatible | A1_opendyslexic |        20 |        -0.18125  |   -0.3      |  -0.075     |     0.0078125 |            0.25 |            -5 |
| openai_compatible | A2_large_sans   |        20 |        -0.15625  |   -0.270885 |  -0.05625   |     0.0078125 |            0.15 |            -3 |
| openai_compatible | C1_red          |        20 |        -0.164583 |   -0.291667 |  -0.0520833 |     0.0703125 |            0.2  |            -4 |
| openai_compatible | C2_gray         |        20 |        -0.172917 |   -0.291667 |  -0.0708333 |     0.0078125 |            0.15 |            -3 |
| openai_compatible | H1_highlight    |        20 |        -0.15625  |   -0.28125  |  -0.0499479 |     0.015625  |            0.15 |            -3 |
| openai_compatible | L1_cramped      |        20 |        -0.114583 |   -0.233333 |  -0.0166667 |     0.21875   |            0.15 |            -3 |
| openai_compatible | T1_serif        |        20 |        -0.164583 |   -0.2875   |  -0.0541667 |     0.0703125 |            0.2  |            -4 |
| openai_compatible | T3_sans         |        20 |        -0.139583 |   -0.262552 |  -0.0375    |     0.0703125 |            0.15 |            -3 |
| openai_compatible | T7_comic        |        20 |        -0.147917 |   -0.266667 |  -0.0479167 |     0.015625  |            0.15 |            -3 |


## OpenDyslexic vs sans reference (within-image)

| provider          | gate    |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |
|:------------------|:--------|----------:|-----------------:|------------:|------------:|--------------:|
| openai_compatible | resume  |        20 |       -0.0416667 |   -0.108333 |  0.00833333 |         0.625 |
| openai_compatible | __all__ |        20 |       -0.0416667 |   -0.108333 |  0.00833333 |         0.625 |

