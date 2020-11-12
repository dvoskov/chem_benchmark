"""
Delft Advanced Research Terra Simulator (DARTS)
 Collection of evaluators of physical properties and operators
"""

class property_evaluator_iface():
    def evaluate(self, *args, **kwargs):
        """
        evaluate(*args, **kwargs)
        Overloaded function.
        
        1. evaluate(self: darts.physics.property_evaluator_iface, state: darts.physics.value_vector) -> float
        
        Evaluate property value for a given state
        
        2. evaluate(self: darts.physics.property_evaluator_iface, states: darts.physics.value_vector, n_blocks: int, values: darts.physics.value_vector) -> int
        
        Evaluate property values for a vector of states
        """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: darts.physics.property_evaluator_iface) -> None """
        pass
