class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise ValueError(attr)
