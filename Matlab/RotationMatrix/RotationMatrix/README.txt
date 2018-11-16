The class RotationMatrix allows to handle a 3D rotation matrix with different parametrizations: 
- a [3x3] rotation matrix 
- Euler angles 
- exponential map 
- quaternions 
Once a RotationMatrix instance has been created from one of the parametrizations above, 
all the parametrizations can be obtained interchangeably.

Additional static methods allow to convert a parametrization into another one without creating instances.

Examples:

% create a RotationMatrix from different parametrizations 

r = RotationMatrix(rand([3,1]), 'exponentialMap');
r = RotationMatrix(eye(3), 'rotationMatrix');
r = RotationMatrix(rand([4,1]), 'quaternion');
r = RotationMatrix(rand([3,1]), 'eulerAngles');


% get a particular representation of a RotationMatrix

aMatrix = r.GetRotationMatrix(); 
aQuaternion = r.GetQuaternion();
eulerAngles = r.GetEulerAngles();
anExponentialMap = r.GetExponentialMap();


% use static methods for creating 3x3 rotation matrices:

rotationMatrix = RotationMatrix.ComputeRotationMatrixFromQuaternion(rand(4,1))
rotationMatrix = RotationMatrix.ComputeRotationMatrixFromEulerAngles(rand(), rand(), rand());
rotationMatrix = RotationMatrix.ComputeRotationMatrixFromExponentialMap(rand(3,1))

% See RotationMatrixTest.m for further examples.
