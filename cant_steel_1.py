from __future__ import print_function
from fenics import *
from ufl import nabla_div


# Step 1: define geometry of the problem
length = .05
width = .005
height = .001

# Step 2: define material parameters
# Step 2a: defining properties of steel
youngs = 2e11 # 200 GPa
nu = 0.3 # poisson ratio
rho = 7800
_lambda = youngs*nu/((1.+nu)*(1.-2.*nu))        # Lame' parameter to be used later
mu = youngs/(2.*(1.+nu))                        # Lame' parameter to be used later

# Step 3: define your mesh
mesh = BoxMesh(Point(0.,0.,0.), Point(length,height,width), 20,6,6)

# Step 4: Function spaces
U = FunctionSpace(mesh,'P',1)
V = VectorFunctionSpace(mesh,'P',1)
W = TensorFunctionSpace(mesh,'P',1)

# Boundary conditions
def boundary_left(x,on_boundary):
	return on_boundary and near(x[0],0.0)

disp_value = Constant((0.,0.,0.))

bc_left = DirichletBC(V,disp_value,boundary_left)

# stresses and strains
def epsilon(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
	eps = epsilon(u)
	return _lambda*tr(eps)*Identity(3) + mu*(eps + eps.T)

# body forces
f = Constant((0.,0.,0.)) # ignoring gravity

# External tractions (force / area)
traction = 20000.0 # N/m^2

T = Expression(('0.0' ,' near(x[1],height) && x[0] >= 0.9*length && x[0] <= length ? - traction : 0.0 ' , '0.0'), degree=1, height=height, length=length, traction=traction)

u_init = TrialFunction(V)
v = TestFunction(V)

# This is the variational problem that we are solving. 
a = inner(sigma(u_init),epsilon(v))*dx
L = dot(f,v)*dx + dot (T,v)*ds

u_init = Function(V)

# This is where you actually solve the problem.
# Note that we are specifying the displacement BCs we defined earlier.
solve(a == L, u_init, bc_left)

#==========================================================
#	Dynamic problem
# d/dx (EA du/dx) + f = m d^2/dt^2 u
#==========================================================
t = 0
t_final = 5e-3  # 1x10^-3
dt = 2e-6	# 5x10^-5

T = Constant((0.0,0.0,0.0))
u = TrialFunction(V)
v = TestFunction(V)
u_n_1 = Function(V)	# u(x,t-dt)
u_n_2 = Function(V)	# u(x,t-2dt)

#==========================================================
# variational form
F = -(dt*dt)*inner(sigma(u),epsilon(v))*dx + (dt*dt)*dot(f,v)*dx - rho*dot(u,v)*dx + 2.0*rho*dot(u_n_1,v)*dx - rho*dot(u_n_2,v)*dx + (dt*dt)*dot(T,v)*ds	#dx => integrate over volume, ds => integrate over surface
a, L = lhs(F), rhs(F) # a = lhs(F) # L = rhs(F)

u = Function(V)
u_n_1.assign(u_init) # u^(n-1) = u_init (initial displacement is given)
u_n_2.assign(u_init) # u^(n-2) = u_init (initial velocity is zero)
#=======================================================
# Next, we set out output files
xdmffile_disp = XDMFFile('cant_steel_1_deflection.xdmf')
xdmffile_stress = XDMFFile('cant_steel_1_stress.xdmf')
xdmffile_vonmises = XDMFFile('cant_steel_1_vonMises.xdmf')

# We are setting some parameters to speed up the code
xdmffile_disp.parameters['rewrite_function_mesh'] = False
xdmffile_stress.parameters['rewrite_function_mesh'] = False
xdmffile_vonmises.parameters['rewrite_function_mesh'] = False

xdmffile_disp.parameters['flush_output'] = True
xdmffile_stress.parameters['flush_output'] = True
xdmffile_vonmises.parameters['flush_output'] = True
#=========================================================

i = 0
while t <= t_final:
	print('t = %f'%t)
	solve(a==L, u, bc_left)

	if (i % 10 == 0):
		xdmffile_disp.write(u.sub(1), t)

	# time marching
	u_n_2.assign(u_n_1)	# u_n_2 = u_n_1
	u_n_1.assign(u)		# u_n_1 = u_n

	# increment time
	t = t+dt
	i = i + 1

print('And we are done!')
