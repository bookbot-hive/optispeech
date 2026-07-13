import torch

from optispeech.model.base_lightning_module import BaseLightningModule


def test_gradient_logging_emits_one_total_per_optimizer(monkeypatch):
    module = BaseLightningModule()
    module.generator = torch.nn.Linear(2, 2)
    module.discriminator = torch.nn.Linear(2, 1)
    module.train_args = {"gradient_log_interval": 100}
    optimizer = torch.optim.SGD(module.generator.parameters(), lr=0.1)
    for parameter in module.generator.parameters():
        parameter.grad = torch.ones_like(parameter)

    logged = []
    monkeypatch.setattr(type(module), "global_step", property(lambda _self: 100))
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: logged.append((args, kwargs)))

    module.on_before_optimizer_step(optimizer)

    assert len(logged) == 1
    assert logged[0][0][0] == "grad_norm/generator_total"
    assert logged[0][1]["on_epoch"] is False


def test_gradient_logging_skips_non_interval_steps(monkeypatch):
    module = BaseLightningModule()
    module.generator = torch.nn.Linear(2, 2)
    module.discriminator = torch.nn.Linear(2, 1)
    module.train_args = {"gradient_log_interval": 100}
    optimizer = torch.optim.SGD(module.generator.parameters(), lr=0.1)
    logged = []
    monkeypatch.setattr(type(module), "global_step", property(lambda _self: 99))
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: logged.append((args, kwargs)))

    module.on_before_optimizer_step(optimizer)

    assert logged == []
