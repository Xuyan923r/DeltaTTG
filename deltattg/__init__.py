"""DeltaTTG: reusable task progress and outcome-credit redistribution."""


def __getattr__(name):
    if name == "apply_continuation_credit_assignment":
        from .integration.trajectory import apply_continuation_credit_assignment
        return apply_continuation_credit_assignment
    if name == "create_meter_client":
        from .meter.client import create_meter_client
        return create_meter_client
    raise AttributeError(name)


__all__ = ["apply_continuation_credit_assignment", "create_meter_client"]
