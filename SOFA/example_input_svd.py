# to be able to add sofa objects you need to first load the plugins that implement them.
# For simplicity you can load the plugin "SofaComponentAll" that will load all most
# common sofa objects.
import SofaRuntime
# SofaRuntime.importPlugin("SofaComponentAll")

# to add elements like Node or objects
import Sofa.Core
root = Sofa.Core.Node()
#import SofaViscoElastic
import math 
import numpy as np
from scipy import signal
import pickle
import os
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import DeLaN_model_svd as delan_svd



# load svd. Make sure to put models directory in the same path as this file, or change the path accordingly.
with open(f"./models/one_segment_spatial_soft_robot_delan_no_svd.jax", 'rb') as f:
    data = pickle.load(f)


hyper = data["hyper"]
params = data["params"]


activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

lagrangian_fn = hk.transform(partial(
    delan_svd.structured_lagrangian_fn_svd,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan_svd.dissipative_matrix_svd,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

# potential matrix
potential_fn = hk.transform(partial(
    delan_svd.potential_energy_fn_svd,
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
))

V = lambda q: potential_fn.apply(params["lagrangian"], None, q).squeeze()
# dV_dq = jax.grad(V)(q)                 # shape (n_dof,)


#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan_svd.input_transform_matrix_svd,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
# dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply


# --- JAX helpers (outside Controller, so they're compiled once) ---
def AL(q):
    # A(q) : (n_dof, actuator_dof)
    return input_mat(params['input_transform'], None, q)

def GL(q):
    # G(q) = dV/dq : (n_dof,)
    return jax.grad(V)(q)

# jit for speed
AL_jit = jax.jit(AL)
GL_jit = jax.jit(GL)
scale_factor = 1e4  # to scale the pressure inputs appropriately

def feedforward_control(qd):
    Al = AL_jit(qd)
    Gl = GL_jit(qd)
    u_ff = np.linalg.pinv(np.array(Al)) @ np.array(Gl)/scale_factor
    return float(np.array(u_ff).squeeze())

def feedback_control(q, qd, dq, Kp=0.5, Kd=0.01):
    Al = AL_jit(qd)
    u_fb = Al*(Kp * (qd-q) - Kd * dq)/scale_factor
    return u_fb


def calculate_angle(origin, tip, angle0=0.0, dt=0.01):
    dx = tip[0] - origin[0]
    dz = tip[2] - origin[2]
    angle = -math.atan2(dx, dz)          # radians
    dangle = (angle - angle0) / dt if dt > 1e-9 else 0.0
    return angle, dangle


class CylinderController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = kwargs['node']
        self.effectorMO = kwargs['effectorMO']
        self.originMO = kwargs['originMO']

        self.q1PointMO = kwargs['q1PointMO']
        self.q2PointMO = kwargs['q2PointMO']
        self.q3PointMO = kwargs['q3PointMO']
        self.pressureValue = kwargs['pressure']

        self.x_prev = np.zeros(4, dtype=float)

        self.file = open("input_test.txt", "w")
        self.file.write(
            "# time[s] u[MPa] "
            "x0[deg] x1[deg] x2[deg] x3[deg] "
            "dx0[deg/s] dx1[deg/s] dx2[deg/s] dx3[deg/s]\n"
        )
    def onAnimateEndEvent(self, event):
        time = float(self.node.time.value)

        # current state
        tip = self.effectorMO.position.value[0]
        origin = self.originMO.position.value[0]

        q1_point = self.q1PointMO.position.value[0]

        q2_point = self.q2PointMO.position.value[0]

        q3_point = self.q3PointMO.position.value[0]

        # get the state of the robot 
        rad2deg = 180/np.pi
        dt = float(self.node.dt.value)

        # --- compute angles in radians ---
        x0, dx0 = calculate_angle(origin, q1_point, angle0=self.x_prev[0], dt=dt)
        x1, dx1 = calculate_angle(origin, q2_point, angle0=self.x_prev[1], dt=dt)
        x2, dx2 = calculate_angle(origin, q3_point, angle0=self.x_prev[2], dt=dt)
        x3, dx3 = calculate_angle(origin, tip,      angle0=self.x_prev[3], dt=dt)

        if time <= dt:
            dx0 = dx1 = dx2 = dx3 = 0.0

        self.x_prev[:] = [x0, x1, x2, x3]
        # --- compute commanded input ---
        # Example: use the tip angle as the 1-DOF state            
        q = np.array([x3], dtype=float)     # (1,)
        dq = np.array([dx3], dtype=float)   # (1,)

        # Desired angle trajectory (example)
        q_ref = np.array([25.0/rad2deg], dtype=float)   # rad, choose what you want

        u_ff = feedforward_control(q_ref)
        pressure_cmd = u_ff 
        pressure_cmd = float(np.asarray(pressure_cmd).squeeze())  # converts np/jax scalars to Python float

        # --- apply input (handle scalar vs vector data) ---
        try:
            # if it's a 1-element vector-like Data
            self.pressureValue.value[0] = pressure_cmd
        except TypeError:
            # if it's a scalar Data
            self.pressureValue.value = pressure_cmd


        print(
            f"Time: {time:.3f} s | "
            f"Pressure cmd: {pressure_cmd:.6f} MPa | "
            f"Angles: [{x0*rad2deg:.2f}, {x1*rad2deg:.2f}, {x2*rad2deg:.2f}, {x3*rad2deg:.2f}] deg | "
            f"dAngles: [{dx0*rad2deg:.2f}, {dx1*rad2deg:.2f}, {dx2*rad2deg:.2f}, {dx3*rad2deg:.2f}] deg/s"
        )

        self.file.write(
            f"{time:.6f} {pressure_cmd:.6e} "
            f"{x0*rad2deg:.6f} {x1*rad2deg:.6f} {x2*rad2deg:.6f} {x3*rad2deg:.6f} "
            f"{dx0*rad2deg:.6f} {dx1*rad2deg:.6f} {dx2*rad2deg:.6f} {dx3*rad2deg:.6f}\n"
        )

        self.file.flush()
    




def createScene(rootNode):

    ## UNIT OF MEASURE : LENGHT [mm] MASS [tonne] TIME [s] PRESSURE/STRESS [MPa]

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields hideInteractionForceFields hideWireframe')
    
    rootNode.addObject('RequiredPlugin', name='SofaPython3')
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Engine.Select")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Forward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Setting")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.HyperElastic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.Elastic")	
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.Spring")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Visual")
    rootNode.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.AnimationLoop")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Solver")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Direct")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Correction")
    rootNode.addObject("RequiredPlugin", name = "Sofa.Component.Constraint.Projective")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Response.Contact")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Constant")
    rootNode.addObject("RequiredPlugin", name="SoftRobots")  
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
    



    rootNode.gravity=[9810,0,0]
    rootNode.dt = (0.01)
    rootNode.name = 'rootNode'
    rootNode.addObject('FreeMotionAnimationLoop')
    # rootNode.addObject('GenericConstraintSolver', tolerance=1e-12, maxIterations=10000, computeConstraintForces = True)
    rootNode.addObject(
    "BlockGaussSeidelConstraintSolver",
    maxIterations=10000,
    tolerance=1e-8, 
    )
    # rootNode.addObject('DefaultPipeline')
    rootNode.addObject('CollisionPipeline')

    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    # rootNode.addObject('DefaultContactManager', response='FrictionContactConstraint', responseParams='mu=0.6')
    # rootNode.addObject('ContactManager', response='FrictionContactConstraint', responseParams='mu=0.6')
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection")


    rootNode.addObject('LocalMinDistance', name='Proximity', alarmDistance = 2, contactDistance=1, angleCone=0.0)
    rootNode.addObject('OglSceneFrame', style='Arrows', alignment='TopRight')


##  Material DBPM-F5000- R0.3  (BLUE)
    finger = rootNode.addChild('finger')

    #finger.addObject('EulerImplicitSolver', name="Solver_Ogden",rayleighMass = 0.1, rayleighStiffness = 0.1, firstOrder = False, trapezoidalScheme = False) 
    finger.addObject('EulerImplicitSolver', name="Solver_Mooney",rayleighMass = 0.0, rayleighStiffness = 0.0, firstOrder = False, trapezoidalScheme = False) 

    # With 0.0 at rayleighStiffness and rayleighMass the ODE solver tries to preserve the dynamics (no artificial dumping added). The euler solver is extremely useful if you want involve contacts (it is the faster and most stable).
    # OGDEN is an exponential model of hyperelasticity, with zero artificial dumping is not stable, Mooney Rivlin is polynomial, so is more stable in simulation in general, also with zero artificial dumping (recommended) 

    finger.addObject('SparseLDLSolver', name='directSolver')

    finger.addObject('MeshVTKLoader', name='loader', filename='design/finger1.vtu', translation = [0, 0.0, 0])
    finger.addObject('MechanicalObject', name='tetras', template='Vec3d', src = '@loader')
    finger.addObject('TetrahedronSetTopologyContainer', name="topo", src ='@loader')
    finger.addObject('TetrahedronSetTopologyModifier' ,  name="Modifier")
    finger.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d" ,name="GeomAlgo")

    finger.addObject('UniformMass', totalMass="28e-6", src = '@topo') # usually the finger should wight 28-30 g
    finger.addObject('BoxROI', name='boxROI1', box=[-12, -5, -2, 10,25, 5], drawBoxes=True)
    finger.addObject('FixedConstraint', indices = '@boxROI1.indices')



# OGDEN COEFFICIENT

    C1 = 0.214210367 #[MPa]
    alpha = 1.61052759
    D1_ogden = 1.16707703 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
    K_ogden = 2/D1_ogden

# MOONEY RIVLIN COEFFICIENT
    C10 = 0.08859383074 #[MPa]
    C01 = 0.01365494839 #[MPa]
    D1_mooney = 1.22250848 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
    K_mooney = 2/D1_mooney


    ## LINEAR ELASTIC COEFFICIENT
    E = 0.6988 #MPa
    nu = 0.44
    E = 0.988 #MPa
    nu = 0.5


    #finger.addObject('TetrahedronHyperelasticityFEMForceField', template='Vec3d', name='FEM', src ='@topo',materialName="Ogden", ParameterSet= str(C1)+' '+str(alpha) + ' '+str(K_ogden)) ## OGDEN

    #finger.addObject('TetrahedronHyperelasticityFEMForceField', template='Vec3d', name='FEM', src ='@topo',materialName="MooneyRivlin", ParameterSet= str(C10)+' '+str(C01) + ' '+str(K_mooney)) ## MOONEY-RIVLIN	

        
    finger.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', src ='@topo', method = 'large', youngModulus = E, poissonRatio = nu) ## Linear Elastic	

    finger.addObject('LinearSolverConstraintCorrection', linearSolver='@directSolver')



    cavity = finger.addChild('cavity')
    cavity.addObject('MeshOBJLoader', name='cavityLoader', filename='design/BSPA_New_Design_Inner.obj')
    cavity.addObject('MeshTopology', src='@cavityLoader', name='cavityMesh')
    cavity.addObject('MechanicalObject', name='cavity', rotation = [0, 0 , 0])
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3d', value= 0, triangles='@cavityMesh.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)


    effector = finger.addChild('effector')
    effectorMO = effector.addObject('MechanicalObject', name='effectorMO', template='Vec3d', position=[10.0,8.0,-100.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    effector.addObject('BarycentricMapping')

    origin = finger.addChild('origin')
    originMO = origin.addObject('MechanicalObject', name='effectorMO', template='Vec3d', position=[10.0,8.0,0.0], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    origin.addObject('BarycentricMapping')


    # I will add more points to define a more appropriate state.
    q2_point = finger.addChild('q2_point')
    q2_pointMO = q2_point.addObject('MechanicalObject', name='q2_pointMO', template='Vec3d', position=[10.0,8.0,-50.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q2_point.addObject('BarycentricMapping')

    q1_point = finger.addChild('q1_point')
    q1_pointMO = q1_point.addObject('MechanicalObject', name='q1_pointMO', template='Vec3d', position=[10.0,8.0,-25.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q1_point.addObject('BarycentricMapping')

    q3_point = finger.addChild('q3_point')
    q3_pointMO = q3_point.addObject('MechanicalObject', name='q3_pointMO', template='Vec3d', position=[10.0,8.0,-75.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q3_point.addObject('BarycentricMapping')




    # rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO))
    rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO, q1PointMO=q1_pointMO, q2PointMO=q2_pointMO, q3PointMO=q3_pointMO))

    modelVisu = finger.addChild('visu')
    modelVisu.addObject('MeshOBJLoader', name='loader', filename='design/BSPA_New_Design_Outer.obj', translation = [0.0,0.0,0.0])
    modelVisu.addObject('OglModel', src='@loader', color=[0,0.64,0,1])
    modelVisu.addObject('BarycentricMapping')



    return rootNode