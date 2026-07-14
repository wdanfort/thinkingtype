# Gates run summary

## Boundary calibration quality (selected items)

| provider   | gate       |   n_items |   mean_p_text |   min_p_text |   max_p_text |   n_in_band |
|:-----------|:-----------|----------:|--------------:|-------------:|-------------:|------------:|
| anthropic  | moderation |        20 |       0.48125 |            0 |            1 |           2 |
| anthropic  | resume     |        20 |       0.5     |            0 |            1 |           0 |
| google     | moderation |        20 |       0.49375 |            0 |            1 |           0 |
| google     | resume     |        20 |       0.5     |            0 |            1 |           0 |
| openai     | moderation |        20 |       0.4625  |            0 |            1 |           4 |
| openai     | resume     |        20 |       0.49375 |            0 |            1 |           2 |


## Modality effect at the boundary (reference fonts only)

| provider   | gate       |   n_items |   mean_delta_p |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:-----------|----------:|---------------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | moderation |        40 |     -0.0645833 |        0.0645833 |   0.0104167 |   0.133333  |     0.34375   |           0.1   |             4 |
| anthropic  | resume     |        40 |      0         |        0         |   0         |   0         |     1         |           0     |             0 |
| google     | moderation |        40 |      0.0270833 |       -0.0270833 |  -0.0875    |   0.0114583 |     0.726562  |           0.025 |            -1 |
| google     | resume     |        40 |      0         |        0         |   0         |   0         |     1         |           0     |             0 |
| openai     | moderation |        40 |     -0.0791667 |        0.0791667 |   0.0291667 |   0.135417  |     0.0385742 |           0.05  |             2 |
| openai     | resume     |        40 |     -0.0479167 |       -0.0479167 |  -0.101042  |  -0.003125  |     0.21875   |           0.05  |            -2 |


## Variant effects pooled over gates

| provider   | variant_id   |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:-------------|----------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | T1_serif     |        40 |        0.0364583 | -0.00208333 |   0.0916667 |      0.375    |           0.05  |             2 |
| anthropic  | T3_sans      |        40 |        0.028125  | -0.00625    |   0.0770833 |      1        |           0.05  |             2 |
| google     | T1_serif     |        40 |       -0.003125  | -0.0125     |   0.00625   |      1        |           0     |             0 |
| google     | T3_sans      |        40 |       -0.0239583 | -0.08125    |   0.0114583 |      1        |           0.025 |            -1 |
| openai     | T1_serif     |        40 |        0.009375  | -0.05       |   0.0677083 |      1        |           0.05  |             0 |
| openai     | T3_sans      |        40 |        0.021875  | -0.0354167  |   0.075026  |      0.507812 |           0.05  |             0 |

