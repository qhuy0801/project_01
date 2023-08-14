def reset_parameters(_ema_model, _model):
    """
    Reset parameters from new model
    :param _ema_model:
    :param _model:
    :return:
    """
    _ema_model.load_state_dict(_model.state_dict())


class EMA:
    """
    The Exponential Moving Average (EMA) is a method employed to enhance and stabilize training outcomes.
    It operates by retaining a version of the model's weights from the prior iteration and adjusting
    the weights of the current iteration by a factor of (1-beta).
    """
    def __init__(self, _beta) -> None:
        super().__init__()
        self.beta = _beta
        self.step = 0

    def update_ema(self, _ema_model, _new_model):
        """
        Update model average step including:
        - Get previous weight
        - Put previous model weight and new obtained weight into updating method
        :param _ema_model:
        :param _new_model:
        :return:
        """
        for ema_params, new_params in zip(_ema_model.parameters(), _new_model.parameters()):
            old_weight, new_weight = ema_params.data, new_params.data
            ema_params.data = self.get_weights(old_weight, new_weight)

    def get_weights(self, _old_weight, _new_weight):
        """
        Calculate the weights of the model based on beta
        :param _old_weight:
        :param _new_weight:
        :return:
        """
        if _old_weight is None:
            return _new_weight
        return _old_weight * self.beta + (1 - self.beta) * _new_weight

    def take_step(self, _ema_model, _new_model, _step_start_ema=2000):
        """
        Take a step:
        - Update weights and EMA
        - Increase step record by 1
        :param _ema_model:
        :param _new_model:
        :param _step_start_ema:
        :return:
        """
        if self.step < _step_start_ema:
            reset_parameters(_ema_model, _new_model)
            self.step += 1
            return
        self.update_ema(_ema_model, _new_model)
        self.step += 1
