is_simple_core = True

if is_simple_core:
    from myke.core_simple import Variable
    from myke.core_simple import Function
    from myke.core_simple import using_config
    from myke.core_simple import no_grad
    from myke.core_simple import as_array
    from myke.core_simple import as_variable
    from myke.core_simple import setup_variable

setup_variable()