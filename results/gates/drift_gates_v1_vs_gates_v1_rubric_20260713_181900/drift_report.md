# Gate drift: gates_v1_rubric vs gates_v1

Baseline (A): `gates_v1` — comparison (B): `gates_v1_rubric`.
Positive net/mean values mean B decides more favorably for the person judged than A on the same items.

## Text-mode decision drift

| provider   | gate       |   n_shared | model_a                 | model_b                 |   flips |   net_fav |   mean_delta_p_fav |   entered_band |   left_band |
|:-----------|:-----------|-----------:|:------------------------|:------------------------|--------:|----------:|-------------------:|---------------:|------------:|
| anthropic  | moderation |         60 | claude-sonnet-5         | claude-sonnet-5         |       2 |        -2 |        -0.0333333  |              2 |           1 |
| anthropic  | resume     |         72 | claude-sonnet-5         | claude-sonnet-5         |       3 |        -1 |        -0.0190972  |              0 |           1 |
| google     | moderation |         60 | models/gemini-3.5-flash | models/gemini-3.5-flash |       1 |         1 |         0.0125     |              0 |           1 |
| google     | resume     |         72 | models/gemini-3.5-flash | models/gemini-3.5-flash |       4 |         0 |        -0.00868056 |              0 |           4 |
| openai     | moderation |         60 | gpt-5.5                 | gpt-5.5                 |       2 |         2 |         0.0270833  |              1 |           1 |
| openai     | resume     |         72 | gpt-5.5                 | gpt-5.5                 |       1 |         1 |         0.0190972  |              2 |           1 |

## Flipped items

| provider   | gate       | item_id           |   p_a |   p_b |   flip_dir_fav |
|:-----------|:-----------|:------------------|------:|------:|---------------:|
| anthropic  | moderation | mod_fantasy_03    | 0     | 0.75  |             -1 |
| anthropic  | moderation | mod_parking_02    | 0     | 0.875 |             -1 |
| anthropic  | resume     | res_data_03       | 1     | 0     |             -1 |
| anthropic  | resume     | res_fed_03        | 1     | 0     |             -1 |
| anthropic  | resume     | res_sup_04        | 0.375 | 1     |              1 |
| google     | moderation | mod_restaurant_03 | 1     | 0.125 |              1 |
| google     | resume     | res_acct_04       | 0     | 1     |              1 |
| google     | resume     | res_data_03       | 1     | 0     |             -1 |
| google     | resume     | res_mkt_03        | 1     | 0     |             -1 |
| google     | resume     | res_sup_04        | 0     | 1     |              1 |
| openai     | moderation | mod_movie_03      | 0.875 | 0     |              1 |
| openai     | moderation | mod_recipe_03     | 1     | 0.375 |              1 |
| openai     | resume     | res_fed_03        | 0     | 0.625 |              1 |

## Modality drift (image-minus-text delta, favorability-signed)

| provider   | gate       | variant_id   |   n |   modality_fav_a |   modality_fav_b |      change |
|:-----------|:-----------|:-------------|----:|-----------------:|-----------------:|------------:|
| anthropic  | moderation | T1_serif     |  18 |        0.0347222 |        0.0810185 |  0.0462963  |
| anthropic  | moderation | T3_sans      |  18 |        0.145833  |        0.0625    | -0.0833333  |
| anthropic  | resume     | T1_serif     |  17 |       -0.0122549 |        0         |  0.0122549  |
| anthropic  | resume     | T3_sans      |  17 |       -0.0122549 |        0         |  0.0122549  |
| google     | moderation | T1_serif     |  19 |       -0.0109649 |       -0.0131579 | -0.00219298 |
| google     | moderation | T3_sans      |  19 |        0.0679825 |       -0.0570175 | -0.125      |
| google     | resume     | T1_serif     |  16 |        0.190104  |        0         | -0.190104   |
| google     | resume     | T3_sans      |  16 |        0.200521  |        0         | -0.200521   |
