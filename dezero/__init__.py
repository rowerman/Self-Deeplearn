# step32之前is_simple_core = True
is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable, Function, as_variable, as_array
    from dezero.core_simple import using_config, no_grad
    from dezero.core_simple import Config
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable, Function, as_variable, as_array, as_variable
    from dezero.core import using_config, no_grad
    from dezero.core import Config
    from dezero.core import setup_variable

setup_variable()