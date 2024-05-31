def relative_error(estimated, truth):
  """Calculate relative error.

  The relative error is defined as (estimated - truth) / truth.

  Args:
    estimated: the estimated value.
    truth: the true value.

  Returns:
    The relative error.
  """
  return (estimated - truth) / truth
