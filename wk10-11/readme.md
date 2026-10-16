# Weeks 10 and 11: Expanded output size and neural model

Recursive forecasting does have issues with accuracy and error buildup over long horizons, so instead, we can test extending XGBoost output size. Writing manual dataframe manipulations is time-consuming, so instead we switched to a machine learning framework: Darts.

Darts requires uninterrupted continuous data, and the current dataset has a bunch of gaps during nighttime. Currently, just interpolate linearly for small gaps, and create seperate time-series for larger ones.

Interpolation is easier:

```python
data["kWh"] = data["kWh"].interpolate(
    method="linear", limit=MAX_FILL_HOURS
)

data["is_nan"] = data["kWh"].isna()
data["segment_id"] = (data["is_nan"] != data["is_nan"].shift()).cumsum()
```

Splitting over gaps is harder:

```python
targets_list = []
covariates_list = []
filler = MissingValuesFiller()

is_nan = data["kWh"].isna()
blocks = (is_nan != is_nan.shift()).cumsum()
gap_sizes = data.groupby(blocks)["kWh"].transform("size")

data.loc[is_nan & (gap_sizes < 24), "kWh"] = 0.0


data["is_long_gap"] = data["kWh"].isna()
data["segment_id"] = (data["is_long_gap"] != data["is_long_gap"].shift()).cumsum()

targets_list = []
covariates_list = []
filler = MissingValuesFiller()

for _, group in data.groupby("segment_id"):
    if not group["is_long_gap"].iloc[0] and len(group) >= MIN_LEN:
        target = TimeSeries.from_dataframe(
            group,
            value_cols="kWh",
            fill_missing_dates=True,
            freq="h"
        )

        targets_list.append(target)

        covariates = TimeSeries.from_dataframe(
          group, value_cols = [
        'elevation', 'temperature_2m', 'cloud_cover', 'direct_normal_irradiance',
        'diffuse_radiation', 'relative_humidity_2m','hour_sin', 'hour_cos',
        'month_sin', 'month_cos', 'DOY_sin', 'DOY_cos'],
          fill_missing_dates=True, freq="1h"
        )

        covariates_list.append(covariates)

print(f"Created {len(targets_list)} continuous TimeSeries segments.")
```

However, model training is easier:

```python
model = XGBModel(
    lags=24,                         # Use past 24 hours of target (kWh)
    lags_future_covariates=(0, 24),  # Use future covariates from step t=0 to t=24
    output_chunk_length=24,          # Forecast 24 steps ahead
    n_estimators=274, random_state=69420, learning_rate=0.016402891919412303, max_depth=8, subsample=0.6358289534884547, colsample_bytree=0.921776337426511,
    tree_method = "hist",
    device = 'cuda',
    kwargs={
        "tree_method": "hist",
        "device": "cuda"
    }
)
```

Copy the tuned hyperparameters from our prevous model because they are (probably) not very different.

Switching to recursive forecasting is much easier than the previous implementation

```python
\model = XGBModel(
    lags=24,                         # Use past 24 hours of target (kWh)
    lags_future_covariates=(0, 24),  # Use future covariates from step t=0 to t=24
    output_chunk_length=1,          # Forecast 24 steps ahead
    n_estimators=274, random_state=69420, learning_rate=0.016402891919412303, max_depth=8, subsample=0.6358289534884547, colsample_bytree=0.921776337426511,
    tree_method = "hist",
    device = 'cuda',
    kwargs={
        "tree_method": "hist",
        "device": "cuda"
    }
)
```

And it is even possible to replace the model entirely with a neural one.

```python
early_stopper = EarlyStopping(
    monitor="val_loss", patience=7, min_delta=1e-4, mode="min"
)

model = TiDEModel(
    input_chunk_length=24,
    output_chunk_length=24,
    num_encoder_layers=2,  # Default is 1
    num_decoder_layers=2,  # Default is 1
    hidden_size=128,  # Default is 128
    decoder_output_dim=32,  # Default is 16
    #temporal_decoder_hidden_size=64,  # Default is 32
    dropout=0.3,  # Increase slightly to prevent overfitting with larger capacity
    n_epochs=64,
    batch_size=64,
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1,
        "callbacks": [early_stopper],
    },
    optimizer_kwargs={
        "lr": 3e-4,  # Lower learning rate (default is 1e-3)
        "weight_decay": 1e-4,  # L2 regularization
    },
)
```

Here are the model's performances:

|          | XGBoost 24 hour output size | XGBoost 1 hour output size (recursive) | TiDE               |
| -------- | --------------------------- | -------------------------------------- | ------------------ |
| RMSE     | 24.02674651701756           | 24.203611033244634                     | 25.752345617003467 |
| MAE      | 11.02817711265039           | 11.516941963107577                     | 12.749587952517954 |
| R2 Score | 0.8985314541966268          | 0.8971331505981426                     | 0.8834331427239953 |

TiDE did not receive an actual hyperparameter tuning pass, but is useful as a performance comparison reguardless.~~~~
