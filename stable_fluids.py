import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt

#visual markers
import cmasher as cmr
from tqdm import tqdm #makes a status bar in the terminal

DOMAIN_SIZE = 1.0
N_POINTS = 41 #can change this later to see what happens
N_TIME_STEPS = 100 #can change
TIME_STEP_LENGTH = 0.1

KINEMATIC_VISCOSITY = 0.0001 #how resistant fluid is to motion eg. honey and water

MAX_ITER_CG = None #we'll come back to this

#forcing function
def forcing_function(time, point):
    time_decay = np.maximum(
        2.0 - 0.5 * time, 0.0
    )

#these are the conditions
    forced_value = (time_decay*np.where(((point[0] > 0.4) & (point[0] < 0.6)) &((point[1] > 0.1) & (point[1]<0.3)),
        #these are what to do
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    ))

    return forced_value

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    scalar_shape = (N_POINTS, N_POINTS)
    scalar_dof = N_POINTS**2 #dof: degrees of freedom eg. area = l*b = x^2
    vector_shape = (N_POINTS, N_POINTS, 2)
    vector_dof = N_POINTS**2 * 2

    x = np.linspace(0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0, DOMAIN_SIZE, N_POINTS)

    #grid
    X, Y = np.meshgrid(x, y, indexing = "ij")

    coordinates = np.concatenate(
        (
            X[..., np.newaxis],
            Y[..., np.newaxis]
        ),
        axis = -1,
    )
    forcing_function_vectorized = np.vectorize(
        pyfunc = forcing_function,
        signature = "(), (d)->(d)",
    )

    def partial_derivative_x(field): #build partial derivatives
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] =(
            (
                field[2: , 1:-1]
                -
                field[0:-2,1:-1]
            ) / (2*element_length)
        )
        return diff
    
    def partial_derivative_y(field): #build partial derivatives
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] =(
            (
                field[1:-1, 2: ]
                -
                field[1:-1, 0:-2]
            ) / (2*element_length)
        )
        return diff

    def laplace(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1: -1] = (
            (
                field[0:-2, 1:-1]
                +
                field[1:-1, 0:-2]
                - 4 *
                field[1:-1, 1:-1]
                +
                field[2: , 1:-1]
                +
                field[1:-1, 2: ]
            ) / (
                element_length**2
            )
        )

        return diff
    
    def divergence(vector_field):
        divergence_applied = (
            partial_derivative_x(vector_field[...,0]) #applying partial derivative just on the 0th column or just the x component
            +
            partial_derivative_y(vector_field[...,1]) #applying on the y component
        )
        return divergence_applied
    
    def gradient(field):
        gradient_applied = np.concatenate(
            (
                partial_derivative_x(field)[..., np.newaxis],
                partial_derivative_y(field)[..., np.newaxis]
            ),
            axis = -1,
        )
        return gradient_applied

    def curl_2d(vector_field):
        curl_applied = (
            partial_derivative_x(vector_field[..., 1])
            -
            partial_derivative_y(vector_field[..., 0])
        )

        return curl_applied

    def advect(field, vector_field):
        backtraced_positions = np.clip(
            (
                coordinates - TIME_STEP_LENGTH * vector_field
            ),
            0.0,
            DOMAIN_SIZE,
        )
        advected_field = interpolate.interpn(
            points =(x,y),
            values = field,
            xi = backtraced_positions,
        )
    
        return advected_field
    
    def diffusion_operator(vector_field_flattened):
        vector_field = vector_field_flattened.reshape(vector_shape)

        diffusion_applied = (
            vector_field - KINEMATIC_VISCOSITY * TIME_STEP_LENGTH * laplace(vector_field)
        )
        return diffusion_applied.flatten()
    
    #poisson operator
    def poisson_operator(field_flattened):
        field = field_flattened.reshape(scalar_shape)

        poisson_applied = laplace(field)

        return poisson_applied.flatten()
    
    plt.style.use("dark_background")
    plt.figure(figsize=(5,5), dpi=160) #adjust later

    velocities_prev = np.zeros(vector_shape)

    time_current = 0.0
    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH

        forces = forcing_function_vectorized(
            time_current,
            coordinates,
        )

        # 1 apply forces
        velocities_forces_applied = (
            velocities_prev + TIME_STEP_LENGTH * forces
        )

        # 2 nonlinear convection = self advection
        velocities_advected = advect(
            field = velocities_forces_applied,
            vector_field = velocities_forces_applied,
        )

        # 3 diffusion
        velocities_diffused = splinalg.cg(
            A = splinalg.LinearOperator(
                shape = (vector_dof, vector_dof),
                matvec = diffusion_operator,
            ),
            b = velocities_advected.flatten(),
            maxiter = MAX_ITER_CG,
        )[0].reshape(vector_shape)

        # 4.1 compute a pressure correction
        pressure = splinalg.cg(
            A = splinalg.LinearOperator(
                shape = (scalar_dof, scalar_dof),
                matvec = poisson_operator,

            ),
            b = divergence(velocities_diffused).flatten(),
            maxiter = MAX_ITER_CG,
        )[0].reshape(scalar_shape)

        # 4.2 correct velocities to be incompressible
        velocities_projected = (
            velocities_diffused - gradient(pressure)
        )

        # move to next step
        velocities_prev = velocities_projected

        #plot
        curl = curl_2d(velocities_projected)
        plt.contourf(
            X,
            Y,
            curl,
            cmap=cmr.redshift,
            levels=100,
        )
        plt.quiver(
            X, 
            Y,
            velocities_projected[..., 0],
            velocities_projected[..., 1],
            color="dimgray",
        )
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    plt.show()      
    
if __name__ == "__main__":
    main()