import pytest

from optispeech.model.base_lightning_module import scheduler_steps_per_optimizer


def test_scheduler_uses_half_of_trainer_steps_for_two_optimizers():
    assert scheduler_steps_per_optimizer(5_000_000) == 2_500_000


def test_scheduler_horizon_is_not_shortened_for_gradient_accumulation():
    # Accumulation is handled by whether optimizer.step()/scheduler.step() are
    # called; the scheduler horizon remains in actual optimizer steps.
    accumulation = 2
    incorrectly_divided_horizon = 5_000_000 // 2 // accumulation

    assert scheduler_steps_per_optimizer(5_000_000) != incorrectly_divided_horizon


@pytest.mark.parametrize("max_steps", [0, -1])
def test_scheduler_requires_finite_positive_max_steps(max_steps):
    with pytest.raises(ValueError, match="max_steps"):
        scheduler_steps_per_optimizer(max_steps)
