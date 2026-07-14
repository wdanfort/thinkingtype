# Gates run summary

## Boundary calibration quality (selected items)

| provider   | gate       |   n_items |   mean_p_text |   min_p_text |   max_p_text |   n_in_band |
|:-----------|:-----------|----------:|--------------:|-------------:|-------------:|------------:|
| anthropic  | moderation |        60 |      0.4875   |            0 |            1 |           1 |
| google     | moderation |        60 |      0.504167 |            0 |            1 |           3 |
| openai     | moderation |        60 |      0.497917 |            0 |            1 |           5 |


## Modality effect at the boundary (reference fonts only)

| provider   | gate       |   n_items |   mean_delta_p |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:-----------|----------:|---------------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | moderation |        60 |     -0.159722  |        0.159722  |  0.077066   |   0.251389  |   0.000488281 |       0.166667  |            10 |
| google     | moderation |        60 |     -0.0319444 |        0.0319444 |  0.00416667 |   0.0652778 |   0.21875     |       0.0166667 |             1 |
| openai     | moderation |        60 |     -0.0673611 |        0.0673611 |  0.0152778  |   0.125     |   0.000518799 |       0.0833333 |             3 |


## Variant effects pooled over gates

| provider   | variant_id      |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:----------------|----------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | A1_opendyslexic |        60 |       0.120833   |  0.0416667  |   0.208333  |   0.109375    |       0.133333  |             8 |
| anthropic  | T3_sans         |        60 |       0.159722   |  0.077066   |   0.251389  |   0.000488281 |       0.166667  |            10 |
| google     | A1_opendyslexic |        60 |      -0.00694444 | -0.0500174  |   0.0256944 |   1           |       0.0166667 |            -1 |
| google     | T3_sans         |        60 |       0.0319444  |  0.00416667 |   0.0652778 |   0.21875     |       0.0166667 |             1 |
| openai     | A1_opendyslexic |        60 |       0.00625    | -0.0541667  |   0.0659722 |   0.454498    |       0.1       |             0 |
| openai     | T3_sans         |        60 |       0.0673611  |  0.0152778  |   0.125     |   0.000518799 |       0.0833333 |             3 |


## OpenDyslexic vs sans reference (within-image)

| provider   | gate       |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |
|:-----------|:-----------|----------:|-----------------:|------------:|------------:|--------------:|
| anthropic  | moderation |        60 |       -0.0388889 |  -0.108333  |  0.0277778  |      0.21875  |
| google     | moderation |        60 |       -0.0388889 |  -0.0888889 | -0.00277778 |      0.125    |
| openai     | moderation |        60 |       -0.0611111 |  -0.122222  | -0.00833333 |      0.226562 |
| anthropic  | __all__    |        60 |       -0.0388889 |  -0.108333  |  0.0277778  |      0.21875  |
| google     | __all__    |        60 |       -0.0388889 |  -0.0888889 | -0.00277778 |      0.125    |
| openai     | __all__    |        60 |       -0.0611111 |  -0.122222  | -0.00833333 |      0.226562 |

