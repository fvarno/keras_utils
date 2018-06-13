class ScaleAll(Layer):

    def __init__(self, **kwargs):
        self.axis = -1
        super(ScaleAll, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('Axis ' +  + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes=dict(list(enumerate(input_shape[1:], start=1))))
        
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(ScaleAll, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return np.multiply(x,self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape)
        
        
