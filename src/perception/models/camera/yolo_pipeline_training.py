from src.perception.models.camera import train_in_pipeline as train
from src.perception.models.camera import detect


def forward(data_yaml, data_path, weights, true_result):
    prediction = detect.run(weights=weights, source=data_path, data=data_yaml)

    # TODO: Calc loss
    return 1


def backward(loss, data, weights):
    train.run(loss=loss, data=data, weights=weights)

