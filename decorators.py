events = {}


def event(event_name):
    def decorator(func):
        events[event_name] = func
        return func

    return decorator
