class SamplingCallback:
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn

    def on_epoch_end(self, epoch: int, model=None):
        if self.sample_fn is not None:
            return self.sample_fn(epoch=epoch, model=model)
        return None
