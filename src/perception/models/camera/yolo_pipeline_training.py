from src.perception.models.camera import train_backward
from src.perception.models.camera import train_forward
from src.perception.models.camera import detect


def forward(data_yaml, data_path, weights, true_result):
    prediction, loss = train_forward.run(weights=weights, source=data_path, data=data_yaml)

    # TODO: Calc loss
    return 1


def backward(loss, data, weights):
    train_backward.run(loss=loss, data=data, weights=weights)

