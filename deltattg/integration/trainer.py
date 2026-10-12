"""Minimal bridge between the pinned verl-agent trainer and DeltaTTG."""

from verl.trainer.ppo.core_algos import AdvantageEstimator

def apply_to_trainer_batch(trainer, batch):
    credit_config = trainer.config.get("credit_assignment", {})
    if not bool(credit_config.get("enabled", False)):
        return batch, {}
    if trainer.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
        raise ValueError("DeltaTTG currently requires algorithm.adv_estimator=grpo")
    meter_config = trainer.config.get("meter", {})
    if not meter_config.get("model_path"):
        raise ValueError("meter.model_path is required when DeltaTTG is enabled")
    if trainer._continuation_meter_client is None:
        from deltattg.meter.client import create_meter_client

        trainer._continuation_meter_client = create_meter_client(
            meter_config
        )
    from deltattg.integration.verl_agent import apply_verl_agent_credit_assignment

    # Use the configured failure cost, falling back to the data response limit.
    configured_failure_cost = credit_config.get("failure_cost")
    failure_cost = (
        int(configured_failure_cost)
        if configured_failure_cost is not None
        else int(trainer.config.data.max_response_length)
    )
    if failure_cost <= 0:
        raise ValueError("credit_assignment.failure_cost must be positive or null")
    return apply_verl_agent_credit_assignment(
        batch,
        actor_tokenizer=trainer.tokenizer,
        credit_config=credit_config,
        meter_config=meter_config,
        meter_client=trainer._continuation_meter_client,
        failure_cost=failure_cost,
    )

