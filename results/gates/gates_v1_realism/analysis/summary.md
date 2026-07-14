# Gates run summary

## Boundary calibration quality (selected items)

| provider   | gate       |   n_items |   mean_p_text |   min_p_text |   max_p_text |   n_in_band |
|:-----------|:-----------|----------:|--------------:|-------------:|-------------:|------------:|
| anthropic  | appeal     |        20 |       0.5     |            0 |            1 |           0 |
| anthropic  | moderation |        20 |       0.48125 |            0 |            1 |           1 |
| anthropic  | resume     |        20 |       0.51875 |            0 |            1 |           1 |
| google     | appeal     |        20 |       0.5     |            0 |            1 |           0 |
| google     | moderation |        20 |       0.48125 |            0 |            1 |           1 |
| google     | resume     |        20 |       0.48125 |            0 |            1 |           4 |
| openai     | appeal     |        20 |       0.5     |            0 |            1 |           0 |
| openai     | moderation |        20 |       0.525   |            0 |            1 |           2 |
| openai     | resume     |        20 |       0.475   |            0 |            1 |           1 |


## Modality effect at the boundary (reference fonts only)

| provider   | gate       |   n_items |   mean_delta_p |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:-----------|----------:|---------------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | moderation |        20 |     -0.18125   |        0.18125   |     0.03125 |    0.34375  |        0.125  |            0.2  |             4 |
| google     | moderation |        20 |     -0.0645833 |        0.0645833 |     0       |    0.1625   |        0.5    |            0.1  |             2 |
| openai     | moderation |        20 |     -0.0833333 |        0.0833333 |     0.01875 |    0.170833 |        0.0625 |            0.05 |             1 |


## Variant effects pooled over gates

| provider   | variant_id     |   n_items |   mean_delta_fav |   ci_lo_fav |   ci_hi_fav |   sign_test_p |   maj_flip_rate |   maj_net_fav |
|:-----------|:---------------|----------:|-----------------:|------------:|------------:|--------------:|----------------:|--------------:|
| anthropic  | N1_forum_light |        20 |      -0.00208333 |  -0.11875   |    0.131302 |        1      |            0.1  |             0 |
| anthropic  | N2_forum_dark  |        20 |       0.0645833  |  -0.05625   |    0.216667 |        1      |            0.1  |             2 |
| anthropic  | N3_phone       |        20 |       0.0229167  |  -0.145833  |    0.2      |        1      |            0.15 |             1 |
| anthropic  | T3_sans        |        20 |       0.18125    |   0.03125   |    0.34375  |        0.125  |            0.2  |             4 |
| google     | N1_forum_light |        20 |       0.0895833  |   0         |    0.197917 |        0.25   |            0.1  |             2 |
| google     | N2_forum_dark  |        20 |       0.0895833  |   0         |    0.2125   |        0.25   |            0.1  |             2 |
| google     | N3_phone       |        20 |       0.05625    |   0         |    0.129167 |        0.25   |            0.05 |             1 |
| google     | T3_sans        |        20 |       0.0645833  |   0         |    0.1625   |        0.5    |            0.1  |             2 |
| openai     | N1_forum_light |        20 |       0.141667   |   0.0375    |    0.264583 |        0.0625 |            0.15 |             3 |
| openai     | N2_forum_dark  |        20 |       0.108333   |   0.0208333 |    0.216667 |        0.0625 |            0.1  |             2 |
| openai     | N3_phone       |        20 |       0.108333   |   0.00625   |    0.233333 |        0.375  |            0.1  |             2 |
| openai     | T3_sans        |        20 |       0.0833333  |   0.01875   |    0.170833 |        0.0625 |            0.05 |             1 |

