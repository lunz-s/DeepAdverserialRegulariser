import numpy as np
from abc import ABC, abstractmethod
import odl
from odl.contrib.tensorflow import as_tensorflow_layer


class ForwardModel(ABC):
    # Defining the forward operators used. For customization, create a subclass of forward_model, implementing
    # the abstract classes.
    name = 'abstract'

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def get_image_size(self):
        # Returns the image size in the format (width, height)
        pass

    @abstractmethod
    def get_measurement_size(self):
        # Returns the measurement size in the format (width, height)
        pass

    # All inputs to the evaluation methods have the format [width, height, channels]
    @abstractmethod
    def forward_operator(self, image):
        # The forward operator
        pass

    @abstractmethod
    def forward_operator_adjoint(self, measurement):
        # Needed for implementation of  RED only.
        # Returns the adjoint of the forward operator of measurements
        pass

    @abstractmethod
    def inverse(self, measurement):
        # An approximate (possibly regularized) inverse of the forward operator.
        # Used as starting point and for training
        pass

    @abstractmethod
    def get_odl_operator(self):
        # The forward operator as odl operator. Needed for total variation only.
        pass

    # Input in the form [batch, width, height, channels]
    @abstractmethod
    def tensorflow_operator(self, tensor):
        # The forward operator as tensorflow layer. Needed for evaluation during training
        pass


class CT(ForwardModel):
    # a model for computed tomography on image of size 64x64. Allows for one color channel only.

    name = 'Computed_Tomography'
    def __init__(self, size):
        super(CT, self).__init__(size)
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')

        geometry = odl.tomo.parallel_beam_geometry(self.space, num_angles=30)
        op = odl.tomo.RayTransform(self.space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(op)
        self.operator = (1 / opnorm) * op
        self.fbp = opnorm * odl.tomo.fbp_op(op)
        self.adjoint_operator = (1 / opnorm)*op.adjoint

        # Create tensorflow layer from odl operator
        self.ray_transform = as_tensorflow_layer(self.operator, 'RayTransform')

    def get_image_size(self):
        return self.space.shape

    def get_measurement_size(self):
        return self.operator.range.shape

    def forward_operator(self, image):
        assert len(image.shape) == 3
        assert image.shape[-1] == 1
        ip = self.space.element(image[..., 0])
        result = np.expand_dims(self.operator(ip), axis=-1)
        return result

    def forward_operator_adjoint(self, measurement):
        assert len(measurement.shape) == 3
        assert measurement.shape[-1] == 1
        ip = self.operator.range.element(measurement[..., 0])
        result = np.expand_dims(self.adjoint_operator(ip), axis=-1)
        return result

    def inverse(self, measurement):
        assert len(measurement.shape) == 3
        assert measurement.shape[-1] == 1
        m = self.operator.range.element(measurement[..., 0])
        return np.expand_dims(self.fbp(m), axis=-1)

    def tensorflow_operator(self, tensor):
        return self.ray_transform(tensor)

    def get_odl_operator(self):
        return self.operator


class Denoising(ForwardModel):
    name = 'Denoising'

    def __init__(self, size):
        super(Denoising, self).__init__(size)
        self.size = size
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')
        self.operator = odl.IdentityOperator(self.space)

    def get_image_size(self):
        return self.size

    def get_measurement_size(self):
        return self.size

    def forward_operator(self, image):
        return image

    def forward_operator_adjoint(self, measurement):
        return measurement

    def inverse(self, measurement):
        return measurement

    def tensorflow_operator(self, tensor):
        return tensor

    def get_odl_operator(self):
        return self.operator
