import pytest

from optispeech.model.base_lightning_module import gradient_accumulation_state


def test_no_accumulation_steps_every_batch():
    assert gradient_accumulation_state(0, None, False) == (1.0, True)


def test_full_accumulation_group_steps_at_requested_size():
    assert gradient_accumulation_state(0, 2, False) == (2.0, False)
    assert gradient_accumulation_state(1, 2, False) == (2.0, True)


def test_incomplete_group_steps_and_rescales_at_epoch_end():
    # An odd 9,905-batch epoch leaves one microbatch in its final group.
    assert gradient_accumulation_state(9_904, 2, True) == (1.0, True)


def test_complete_group_at_epoch_end_keeps_full_divisor():
    assert gradient_accumulation_state(9_903, 2, True) == (2.0, True)


def test_invalid_accumulation_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        gradient_accumulation_state(0, 0, False)
