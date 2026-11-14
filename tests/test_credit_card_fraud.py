import pandas as pd
import pytest

from credit_card_fraud import (
    build_model,
    load_data,
    prepare_data,
)


def make_dataset(rows: int = 40) -> pd.DataFrame:
    fraud_count = rows // 4
    return pd.DataFrame(
        {
            "Time": range(rows),
            "Amount": [float(index + 1) for index in range(rows)],
            "V1": [index % 5 for index in range(rows)],
            "Class": [1 if index < fraud_count else 0 for index in range(rows)],
        }
    )


def test_load_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "creditcard.csv")


def test_load_data_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "creditcard.csv"
    pd.DataFrame({"Amount": [10.0], "Class": [0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Time"):
        load_data(path)


def test_prepare_data_keeps_stratified_target_split():
    dataframe = make_dataset()
    x_train, x_test, y_train, y_test = prepare_data(dataframe)

    assert len(x_train) + len(x_test) == len(dataframe)
    assert set(x_train.columns) == {"Amount", "V1"}
    assert y_train.value_counts().to_dict() == {0: 21, 1: 7}
    assert y_test.value_counts().to_dict() == {0: 9, 1: 3}


def test_smote_pipeline_produces_probabilities():
    dataframe = make_dataset()
    x_train, x_test, y_train, y_test = prepare_data(dataframe)
    model = build_model("smote")

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    assert len(probabilities) == len(y_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
