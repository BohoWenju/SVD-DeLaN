import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial


'''
Creates a mask for selective training of parameters in an SVD-based model.

This function generates a boolean mask with the same structure as the parameter
tree, indicating which parameters should be updated during optimization. The mask
is constructed based on a user-provided dictionary specifying which modules to train,
and selectively enables updates for specific parameter types (e.g. weights, singular
values, and optionally biases).

@param[in] tree : haiku parameter tree (module_name -> parameter_name -> array).
@param[in] dix : dictionary mapping module names to booleans, indicating whether
                 each module should be trained.
@param[in] train_bias : whether to include bias parameters ("b" or "bias") in training.

@return mask : dictionary with the same structure as tree, containing boolean values
               that specify which parameters are trainable.
'''
def make_svd_mask(tree, dix,  train_bias=True):
    # tree is the haiku param tree (module_name -> param_name -> array)
    def leaf_mask(mod, leaf, dix):
        out = {}
        flag = dix[mod]
        # print(mod, flag)
        for k in leaf.keys():
            if k == 'log_s' or (train_bias and k in ("b", "bias")) or k in ("w"):
                out[k] = True and flag
            else:
                out[k] = False
        return out
    
    return {mod: leaf_mask(mod, leaf, dix) for mod, leaf in tree.items()}

'''
SVD factorization function. Takes a weight matrix W and returns its SVD components U, log_s, Vt.
Logarithm of singular values is used for better numerical stability during training. 
The rank parameter allows for truncation of the SVD to a specified number of components.
'''
def svd_factorize(W, rank=None, eps=1e-8):
    U, S, Vt = jnp.linalg.svd(W, full_matrices=False)
    if rank is not None:
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    log_s = jnp.log(S + eps)
    return U, log_s, Vt

# for naming issues.
def _strip_tilde_scope(path: str) -> str:
    return path.replace("/~/", "/")

'''
Loads parameters from a pretrained model into an initialized parameter tree,
handling both dense and SVD-based weight representations.

This function matches modules between the initialized model and the pretrained
model (accounting for naming differences), copies compatible parameters, and
converts dense weights into SVD form when required.

@param[in] init_tree : dictionary containing the initialized model parameters.
@param[in] pretrained_tree : dictionary containing the pretrained model parameters.
@param[in] rank : target rank used when factorizing dense weights into SVD form.

@return out : dictionary with the same structure as init_tree, populated with
              pretrained parameters where possible.
'''
def load_svd_from_pretrained(init_tree, pretrained_tree, rank):
    pretrained_by_norm = {_strip_tilde_scope(k): k for k in pretrained_tree.keys()}

    out = {}
    for mod_name, init_leaf in init_tree.items():
        leaf = dict(init_leaf)  

        pretrained_mod = pretrained_by_norm.get(mod_name)
        if pretrained_mod is None:
            out[mod_name] = leaf
            continue

        pre_leaf = pretrained_tree[pretrained_mod]

        
        if "b" in leaf and "b" in pre_leaf:
            leaf["b"] = pre_leaf["b"]

        if "w" not in pre_leaf:
            out[mod_name] = leaf
            continue

        Wpre = pre_leaf["w"]

        
        init_is_dense = "w" in init_leaf and ("U" not in init_leaf)
        init_is_svd   = "U" in init_leaf and "Vt" in init_leaf and "log_s" in init_leaf

        if init_is_dense:
            # Ensure no SVD keys remain
            leaf.pop("U", None)
            leaf.pop("Vt", None)
            leaf.pop("log_s", None)

            # Load W into the expected shape
            Wexp = leaf["w"]
            
            if getattr(Wpre, "ndim", None) == getattr(Wexp, "ndim", None) and Wpre.shape == Wexp.shape:
                leaf["w"] = Wpre
            elif getattr(Wpre, "ndim", None) == 2 and getattr(Wexp, "ndim", None) == 2 and Wpre.T.shape == Wexp.shape:
                leaf["w"] = Wpre.T
            elif getattr(Wpre, "ndim", None) == 2 and getattr(Wexp, "ndim", None) == 1:
                # handle scalar<->vector special cases if your old model stored 2D
                if Wpre.shape[0] == 1 and Wpre.shape[1] == Wexp.shape[0]:
                    leaf["w"] = Wpre.reshape(-1)          # (1, in) -> (in,)
                elif Wpre.shape[1] == 1 and Wpre.shape[0] == Wexp.shape[0]:
                    leaf["w"] = Wpre.reshape(-1)          # (out, 1) -> (out,)
                else:
                    raise ValueError(f"{mod_name}: cannot map pretrained w{Wpre.shape} to expected w{Wexp.shape}")
            else:
                raise ValueError(f"{mod_name}: cannot map pretrained w{getattr(Wpre,'shape',None)} to expected w{getattr(Wexp,'shape',None)}")

        elif init_is_svd:
            # Ensure no dense 'w' remains
            leaf.pop("w", None)

            if getattr(Wpre, "ndim", None) != 2:
                raise ValueError(f"{mod_name}: init expects SVD (U,Vt,log_s) but pretrained w is not 2D: {getattr(Wpre,'shape',None)}")

            U, log_s, Vt = svd_factorize(Wpre, rank=rank)
            leaf["U"] = U
            leaf["log_s"] = log_s
            leaf["Vt"] = Vt

        # else: init leaf is something unexpected; leave as-is

        out[mod_name] = leaf

    return out


'''
Definition of the SVD Linear module. During initialization the module 
will define the weights for each layer in the appropriate form.

@param[in] output_size : the output dimension of the linear layer.
@param[in] rank : the rank for the SVD decomposition. 
@param[in] with_bias : whether to include a bias term in the linear transformation.
@param[in] name : the name of the module.
''' 
class SVDLinear(hk.Module):
    def __init__(self, output_size, rank=None, with_bias=True, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.rank = rank
        self.with_bias = with_bias

    def __call__(self, x):
        in_size = x.shape[-1]
        r = self.rank if self.rank is not None else min(self.output_size, in_size)

        if in_size == 1: # definition of first layer.
            W = hk.get_parameter("w", (self.output_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
                y = x * W + b
        elif self.output_size == 1: # definition of last layer.
            W = hk.get_parameter("w", (in_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (1,), init=jnp.zeros)
                y = jnp.dot(x, W) + b
        else: # definition of hidden layers.
            U = hk.get_parameter("U", (self.output_size, r),
                                init=hk.initializers.RandomNormal(1e-2))
            Vt = hk.get_parameter("Vt", (r, in_size),
                                  init=hk.initializers.RandomNormal(1e-2))
            log_s = hk.get_parameter("log_s", (r,), init=jnp.zeros)
            # freeze bases
            U = jax.lax.stop_gradient(U)
            Vt = jax.lax.stop_gradient(Vt)
            s = jnp.exp(log_s)
            W = (U * s[None, :]) @ Vt
            y = x @ W.T
            if self.with_bias:
                b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
                y = y + b

        return y

'''
Definition of the network architecture that takes advantage of the previous SVD module.

@param[in] output_sizes : the output dimension of the layers.
@param[in] activation : the activation function.
@param[in] rank : the rank for the SVD decomposition.
@param[in] name : the name of the network. Could be e.g. potential energy.
''' 
class SVDMLP(hk.Module):
    def __init__(self, output_sizes, activation, rank=None, name=None):
        super().__init__(name=name)
        self.output_sizes = output_sizes
        self.activation = activation
        self.rank = rank

    def __call__(self, x):
        for i, out_size in enumerate(self.output_sizes[:-1]):
            # apply linear and activation for all layers except the last one.
            x = SVDLinear(out_size, rank=self.rank, with_bias=True, name=f"linear_{i}")(x)
            x = self.activation(x) 
        x = SVDLinear(self.output_sizes[-1], rank=self.rank, with_bias=True,
                      name=f"linear_{len(self.output_sizes)-1}")(x)
        return x

'''
Rest of the code is derived from : https://github.com/jingyueliu6/PINNs_LNN_HNN.
Slight changes are done to use the SVDMLP instead of the MLP.
'''

def rk4_step(f, x, y, t, h):
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1/2, y, t + h/2)
    k3 = h * f(x + k2/2, y, t + h/2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def dissipative_matrix(q, n_dof, shape, activation, rank=None):
    n_output = int((n_dof ** 2 + n_dof) / 2)

    net = SVDMLP(output_sizes=shape + (n_output,),
                 activation=activation,
                 rank=rank,
                 name="dissipative_matrix")

    scaler = 0.4
    l_diagonal, l_off = jnp.split(net(q), [n_dof], axis=-1)
    l_diagonal = jax.nn.sigmoid(l_diagonal)

    triangular = jnp.zeros((n_dof, n_dof))
    triangular = triangular.at[np.diag_indices(n_dof)].set(l_diagonal)
    triangular = triangular.at[np.tril_indices(n_dof, -1)].set(l_off)

    D = triangular @ triangular.T
    return D * scaler

def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift, rank = None):
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = SVDMLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      rank=rank,
                      name="mass_matrix")

    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    # Ensure positive diagonal:
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


def input_transform_matrix(q, n_dof, actuator_dof, shape, activation, rank=None):
    n_output = n_dof * actuator_dof
    net = SVDMLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      rank=rank,
                      name="input_transform_matrix")
    input_mat = net(q).reshape(n_dof, actuator_dof)
    return input_mat


def potential_energy_fn(q, shape, activation, rank=None):
    net = SVDMLP(output_sizes=shape +(1, ),
                      activation=activation,
                      rank=rank,
                      name="potential_energy")

    # Apply feature transform
    return net(q)

def kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift, rank=None):
    mass_mat = mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift, rank=rank)
    return 1./2. * jnp.dot(qd, jnp.dot(mass_mat, qd))

def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift, rank=None):
    e_kin = kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift, rank=rank)
    e_pot = potential_energy_fn(q, shape, activation, rank=rank).squeeze()
    return e_kin - e_pot


def forward_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, tau, t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums) # lagrangian_fn.apply(params, rng, q, qd)  ==> positional indices:   0      1   2  3
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        dissipative = dissipative_mat(d_params, key, q)

        # for A(q) as a net
        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        qdd_pred = jnp.linalg.pinv(d2Ld2qd) @ \
                   (input_transform @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)

        return jnp.concatenate([qd, qdd_pred])
    return equation_of_motion


def inverse_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, qdd=None,  t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        # def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        # def dissipative_matrix(qd, n_dof, shape, activation):
        dissipative = dissipative_mat(d_params, key, q)

        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        # Compute the inverse model
        tau = jnp.linalg.inv(input_transform) @ (d2Ld2qd @ qdd + d2L_dqddq @ qd - dLdq + dissipative @ qd)
        return tau
    return equation_of_motion

def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian, dissipative_mat, input_mat, n_dof, time_step=None):
    states = jnp.concatenate([q, qd], axis=1)
    targets = jnp.concatenate([q_next, qd_next], axis=1)

    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, input_mat=input_mat, n_dof=n_dof))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), (0, 0))(states, tau)
    else:
        preds = jax.vmap(f, (0, 0))(states, tau)

    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)

    # Compute Loss
    loss =  mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs


def loss_fn_experiment(params, q, qd, tau, q_next, qd_next, lagrangian, dissipative_mat, input_mat, n_dof, time_step=None):
    states = jnp.concatenate([q, qd], axis=1)
    targets = jnp.concatenate([q_next, qd_next], axis=1)

    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, input_mat=input_mat, n_dof=n_dof))

    if time_step is not None:
        # preds = jax.vmap(partial(rk4_step, f, t=0.0), (0, 0, 0))(states, tau, time_step)
        preds = jax.vmap(
            lambda x, y, h: rk4_step(f, x, y, t=0.0, h=h),
            in_axes=(0, 0, 0)
        )(states, tau, time_step)
    else:
        preds = jax.vmap(f, (0, 0))(states, tau)

    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)

    # Compute Loss
    loss =  mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs


