## WantWords Baseline Architecture
https://github.com/thunlp/MultiRD

To run: 
- Unzip `data.zip` in the `data` directory
- From the `code` directory, run `python main.py -e <epochs> -m rsl`
- Execute `python evaluate_result.py -m rsl` to see test statistics

Achieved the following results with `batch_size=128`, `epochs=20`:

| Test | Accuracy @ 1/10/100 | Median Rank | Rank Variance |
| -- | -- | -- | -- |
| 500 seen | 19.20 / 43.20 / 72.60 | 16.00 | 298.83 |
| 500 unseen | 8.80 / 29.40 / 59.20 | 49.00 | 351.45 |
|200 | 31.00 / 65.00 / 88.00 | 4.00 | 219.22 |
