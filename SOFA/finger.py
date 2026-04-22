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

import os


def calculate_angle(origin, tip, angle0 = 0, dt=0.01):
    dx = tip[0] - origin[0]
    dz = tip[2] - origin[2]
    angle = -math.atan(dx / dz) * (180 / math.pi) if abs(dz) > 1e-9 else 0.0
    dangle = (angle - angle0) / dt if dt > 1e-9 else 0.0
    return angle, dangle


class CylinderController(Sofa.Core.Controller):
    n_States = 4
    x = np.zeros((n_States,1))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = kwargs['node']
        self.effectorMO = kwargs['effectorMO']
        self.originMO = kwargs['originMO']

        self.q1PointMO = kwargs['q1PointMO']
        self.q2PointMO = kwargs['q2PointMO']
        self.q3PointMO = kwargs['q3PointMO']
        self.pressureValue = kwargs['pressure']

        self.file = open("new_new_finger_io_sin.txt", "w")
        self.file.write(
            "# time[s] u[MPa] "
            "x0[deg] x1[deg] x2[deg] x3[deg] "
            "dx0[deg/s] dx1[deg/s] dx2[deg/s] dx3[deg/s]\n"
        )
    def onAnimateEndEvent(self, event):
        time = float(self.node.time.value)

        # --- compute commanded input ---
        f = 0.7  # Hz
        pressure_cmd = 0.00015 + 0.00015 * math.sin(2 * math.pi * f * time)
        # pressure_cmd = np.random.rand() * 0.0002
        # pressure_cmd = 0.0003

        # if time > 2:
        #     pressure_cmd = 0.0
        # if time > 6:    
        #     pressure_cmd = 0.00015

        # --- apply input (handle scalar vs vector data) ---
        try:
            # if it's a 1-element vector-like Data
            self.pressureValue.value[0] = pressure_cmd
        except TypeError:
            # if it's a scalar Data
            self.pressureValue.value = pressure_cmd

        # --- outputs ---
        tip = self.effectorMO.position.value[0]

        q1_point = self.q1PointMO.position.value[0]

        q2_point = self.q2PointMO.position.value[0]

        q3_point = self.q3PointMO.position.value[0]

        # get the state of the robot 
        origin = self.originMO.position.value[0]
        x0, dx0 = calculate_angle(origin, q1_point, angle0 = self.x[0,-1], dt=self.node.dt.value)
        x1, dx1 = calculate_angle(origin, q2_point, angle0 = self.x[1,-1], dt=self.node.dt.value)
        x2, dx2 = calculate_angle(origin, q3_point, angle0 = self.x[2,-1], dt=self.node.dt.value)
        x3, dx3 = calculate_angle(origin, tip, angle0 = self.x[3,-1], dt=self.node.dt.value)

        self.file.write(
            f"{time:.6f} {pressure_cmd:.6e} "
            f"{x0:.6f} {x1:.6f} {x2:.6f} {x3:.6f} "
            f"{dx0:.6f} {dx1:.6f} {dx2:.6f} {dx3:.6f}\n"
        )    




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
    rootNode.addObject(
    "BlockGaussSeidelConstraintSolver",
    maxIterations=10000,
    tolerance=1e-8, 
    )
    rootNode.addObject('CollisionPipeline')

    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
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

#     C1 = 0.214210367 #[MPa]
#     alpha = 1.61052759
#     D1_ogden = 1.16707703 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
#     K_ogden = 2/D1_ogden

# # MOONEY RIVLIN COEFFICIENT
#     C10 = 0.08859383074 #[MPa]
#     C01 = 0.01365494839 #[MPa]
#     D1_mooney = 1.22250848 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
#     K_mooney = 2/D1_mooney


#     ## LINEAR ELASTIC COEFFICIENT
    E = 0.6988 #MPa
    nu = 0.44
    # trial 1
    E = 0.988 #MPa
    nu = 0.5
        
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
    # only endpoint.
    # rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO)) #, q1PointMO=q1_pointMO, q2PointMO=q2_pointMO, q3PointMO=q3_pointMO))
    # with more points to define the state of the robot.
    rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO, q1PointMO=q1_pointMO, q2PointMO=q2_pointMO, q3PointMO=q3_pointMO))

    modelVisu = finger.addChild('visu')
    modelVisu.addObject('MeshOBJLoader', name='loader', filename='design/BSPA_New_Design_Outer.obj', translation = [0.0,0.0,0.0])
    modelVisu.addObject('OglModel', src='@loader', color=[0,0.64,0,1] ) # [1,0.64,0,1]
    modelVisu.addObject('BarycentricMapping')



    return rootNode