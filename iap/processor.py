import inspect

__all__ = ["func_processor", "select_processor", "if_processor"]


class func_processor():
    '''Convenient processor for arbitrary process function.
    '''
    def __init__(self, process_func, params):
        self.process_func = process_func
        self.params = params

    def dump_config(self):
        config = {"__name__": "func_processor", "source": inspect.getsource(self.process_func)}
        config.update(self.params)
        return config

    def __call__(self, iap_data):
        return self.process_func(iap_data, **self.params)


class select_processor():
    def __init__(self, cleaner, params):
        self.params = params
        self.cleaner = cleaner(**params)
        def process_func(iap_data):
            if self.cleaner.check_data(iap_data):
                return iap_data
            else:
                return None
        self.process_func = process_func

    def dump_config(self):
        config = {"__name__": "select_processor", "cleaner": self.cleaner.__class__.__name__}
        config.update(self.params)
        return config

    def __call__(self, iap_data):
        return self.process_func(iap_data)


class if_processor():
    def __init__(self, if_expressions):
        self.params = if_expressions
        def process_func(iap_data):
            # print([if_expression(iap_data) for if_expression in if_expressions])
            if all([if_expression(iap_data) for if_expression in if_expressions]):
                return iap_data
            else:
                return None
        self.process_func = process_func

    def dump_config(self):
        config = {"__name__": "if_processor"}
        config.update({"lambda": [inspect.getsource(if_expression) for if_expression in self.params]})
        return config

    def __call__(self, iap_data):
        return self.process_func(iap_data)

