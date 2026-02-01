def to_int_list(arg):
    # to simplify parsing of list inputs when using 3rd party code
    if isinstance(arg, str):
        return [int(x) for x in arg.split()]
    elif isinstance(arg, int):
        return [arg]
    else:
        return arg
