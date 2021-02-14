is_simple_core = False

if is_simple_core:
    from myke.core_simple import Variable
    from myke.core_simple import Function
    from myke.core_simple import using_config
    from myke.core_simple import no_grad
    from myke.core_simple import as_array
    from myke.core_simple import as_variable
    from myke.core_simple import setup_variable
else:
    from myke.core import Variable
    from myke.core import Function
    from myke.core import Config
    from myke.core import using_config
    from myke.core import no_grad
    from myke.core import test_mode
    from myke.core import as_array
    from myke.core import as_variable
    from myke.core import setup_variable
    from myke.core import Parameter
    from myke.layers import Layer
    from myke.models import Model
    from myke.dataloaders import DataLoader

setup_variable()