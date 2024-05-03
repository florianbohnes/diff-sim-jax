# Differentiable Race Car Simulation with Jax
for Optimized Control Strategies

This repository is the implementation of my Bachelor's thesis: 

Differentiable Race Car Simulation with Jax
for Optimized Control Strategies

## Background 
The development of autonomous control systems demands sophisticated simulation to train
algorithms for those systems reliably and efficiently. Simulation allows developers to test au-
tonomous systems in a virtual environment before deploying them to the real world. While
classical methods for simulations have their merits, the demand for increasingly advanced
simulations arises from the need to expose control algorithms to highly interactive scenar-
ios, ensuring their adaptability to diverse real-world situations. Lately, much progress could
be made in that domain by employing techniques rooted in Reinforcement Learning. One
specific use case for such simulation is the development of autonomous vehicle controls. As
dynamics simulation is a quite complicated matter, the complexity of the systems necessary
to simulate dynamics grows rapidly. An increasing amount of computational resources are
taken up by that. There is an evident demand for simulation environments that are scalable
and computationally efficient

This is a port of the [F1TENTH gym environment](https://github.com/f1tenth/f1tenth_gym) to [JAX](https://github.com/google/jax)

## Install 
pip install jax  
pip install numpy   
pip install matplotlib  
pip install hydra-core  
pip install tqdm  
pip install gymnasium  
pip install numba  
pip install scipy  
pip install pyglet==1.5.0  
pip install -e jax_simulator/  


## License
GNU General Public License v3.0 only" (GPL-3.0) © [florian bohnes](https://github.com/florianbohnes) 
