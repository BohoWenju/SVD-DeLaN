import numpy as np
import pickle
import os
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from haiku import data_structures as ds
import DeLaN_model_svd as delan_svd
import DeLaN_model_v4 as delan
from utils import ReplayMemory
import csv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import optax

SAVE_MODEL = True
USE_SVD = True   # comment/uncomment
RANK = None      # None = full rank; or 32/64 to compress
TRAIN_BIAS = True


'''
0. Data Loading and data preprocess
'''
print("Loading Data:")
cache_path = os.path.join(os.getcwd(), "data", "data.npy")
data = np.load(cache_path)

# data format: [q, dq, input, q_next, dq_next].
# Defining q and dq depends on the application.
# Below only the endpoint of the soft robot is considered, so q and dq 
# are the angle and angular velocity of the endpoint, respectively.
time_step = 0.01
n_states = 5
config_data = data[:,n_states-1]
dconfig_data = data[:,2*n_states-1]

deg2rad = np.pi / 180.0
scale_input = 1e4 # input is in MPa, scale to [0, 5]
config_data  = config_data     * deg2rad
dconfig_data = dconfig_data   * deg2rad
config_data_target      = data[:, 3*n_states]    * deg2rad
dconfig_data_target = data[:,-1] * deg2rad
inputs       = np.array(data[:, 2*n_states]  * scale_input).reshape(-1, 1)

targets = np.column_stack([config_data_target, dconfig_data_target])
states = np.column_stack((config_data, dconfig_data))


print(f"There are {states.shape[0]} data in this attempt.")
print("Set 80%  data as the training set and 20% as the test set.")

div = states.shape[0] * 8 // 10
train_states, test_states = states[:div, :], states[div:, :]
train_targets, test_targets = targets[:div, :], targets[div:, :]
train_inputs, test_inputs = inputs[:div], inputs[div:]


train_q, train_dq = jnp.split(train_states, 2, axis=1)
train_q_next, train_dq_next = jnp.split(train_targets, 2, axis=1)

test_q, test_dq = jnp.split(test_states, 2, axis=1)
test_q_next, test_dq_next = jnp.split(test_targets, 2, axis=1)

actuator_dof = train_inputs.shape[1]
config_dof = train_q.shape[1]

'''
1. Load model. This is a transfer learning method. 
The model is pretrained on a different dataset.
'''
t0 = time.perf_counter()
# Load model
model_path = os.path.join(os.getcwd(), "models", "one_segment_spatial_soft_robot_delan.jax")

with open(model_path, "rb") as f:
    data = pickle.load(f)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

hyper = data["hyper"]
params = data["params"]
print("hyperparameters:", hyper)

# transform the functions to haiku modules.
if USE_SVD:
    lagrangian_fn = hk.transform(partial(
        delan_svd.structured_lagrangian_fn,
        n_dof=hyper['n_dof'],
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation1']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
        rank=RANK,
    ))

    dissipative_fn = hk.transform(partial(
        delan_svd.dissipative_matrix,
        n_dof=hyper['n_dof'],
        shape=(5,) * 3,
        activation=activations[hyper['activation2']],
        rank=RANK,
    ))

    input_mat_fn = hk.transform(partial(
        delan_svd.input_transform_matrix,
        n_dof=hyper['n_dof'],
        actuator_dof=hyper['actuator_dof'],
        shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
        activation=activations[hyper['activation1']],
        rank=RANK,
    ))
else:
    lagrangian_fn = hk.transform(partial(
        delan.structured_lagrangian_fn,
        n_dof=hyper['n_dof'],
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation1']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
    ))

    dissipative_fn = hk.transform(partial(
        delan.dissipative_matrix,
        n_dof=hyper['n_dof'],
        shape=(5,) * 3,
        activation=activations[hyper['activation2']]
    ))

    input_mat_fn = hk.transform(partial(
        delan.input_transform_matrix,
        n_dof=hyper['n_dof'],
        actuator_dof=hyper['actuator_dof'],
        shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
        activation=activations[hyper['activation1']]
    ))


# store the data in easier format.
mem_dim = ((config_dof,), (config_dof,), (actuator_dof,), (config_dof,), (config_dof,))
mem = ReplayMemory(train_q.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_q, train_dq, train_inputs, train_q_next, train_dq_next])

rng = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng)

# Initialize parameters
q, dq, tau, q_next, dq_next = [jnp.array(x) for x in next(iter(mem))]

# Create SVD-shaped parameter trees (structure only) 
init_l = lagrangian_fn.init(init_key, q[0], dq[0])        # args must match apply signature
init_d = dissipative_fn.init(init_key, q[0])              # dissipative_matrix(q)
init_i = input_mat_fn.init(init_key, q[0])                # input_transform_matrix(q)

if USE_SVD:
    # init trees define correct naming & shapes
    l_params = delan_svd.load_svd_from_pretrained(init_l, params["lagrangian"], rank=RANK)
    d_params = delan_svd.load_svd_from_pretrained(init_d, params["dissipative"], rank=RANK)
    i_params = delan_svd.load_svd_from_pretrained(init_i, params["input_transform"], rank=RANK)
else:
    l_params = params["lagrangian"]
    d_params = params["dissipative"]
    i_params = params["input_transform"]

params_used = {"lagrangian": l_params, "dissipative": d_params, "input_transform": i_params}

p = ds.to_mutable_dict(l_params)
for mod_name, mod_params in p.items():
    print(mod_name, {k: v.shape for k, v in mod_params.items()})

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

# print("l_params keys:", l_params['mass_matrix/linear_0']['U'].shape)
# print("l_params keys:", l_params['mass_matrix/linear_0']['log_s'].shape)
# print("l_params keys:", l_params['mass_matrix/linear_0']['Vt'].shape)

# forward model is the same as the original DeLaN.
forward_model = delan_svd.forward_model(
    params=params_used, key=None,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    input_mat=input_mat,
    n_dof=config_dof
)
    
# Sanity check.
# m = [k for k,v in params["dissipative"].items() if "w" in v][0]
# W = params["dissipative"][m]["w"]
# U = params_used["dissipative"][m]["U"]
# log_s = params_used["dissipative"][m]["log_s"]
# Vt = params_used["dissipative"][m]["Vt"]
# W_rec = (U * jnp.exp(log_s)[None, :]) @ Vt
# print("max reconstruction err:", jnp.max(jnp.abs(W - W_rec)))

state0 = jnp.concatenate([q[0], dq[0]])
temp = forward_model(state0, tau[0])

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")


# # 2. Generate and initialize the optimizer
t0 = time.perf_counter()


mass_dict = {'mass_matrix/linear_0': True, 'mass_matrix/linear_1': True, 'mass_matrix/linear_2': True, 'mass_matrix/linear_3': True}
pot_dict = {'potential_energy/linear_0': True, 'potential_energy/linear_1': True, 'potential_energy/linear_2': True, 'potential_energy/linear_3': True}
lagrangian_dict = {**mass_dict, **pot_dict}

diss_dict = {'dissipative_matrix/linear_0': True, 'dissipative_matrix/linear_1': True, 'dissipative_matrix/linear_2': True, 'dissipative_matrix/linear_3': True}
input_dict = {'input_transform_matrix/linear_0': True, 'input_transform_matrix/linear_1': True, 'input_transform_matrix/linear_2': True, 'input_transform_matrix/linear_3': True}   


if USE_SVD:
    mask_l = delan_svd.make_svd_mask(params_used["lagrangian"],dix=lagrangian_dict, train_bias=TRAIN_BIAS)
    mask_d = delan_svd.make_svd_mask(params_used["dissipative"], dix=diss_dict, train_bias=TRAIN_BIAS)
    mask_i = delan_svd.make_svd_mask(params_used["input_transform"], dix=input_dict, train_bias=TRAIN_BIAS)

    optimizer1 = optax.masked(optax.adamw(hyper['learning_rate'], hyper['weight_decay']), mask_l)
    optimizer2 = optax.masked(optax.adamw(hyper['learning_rate'], hyper['weight_decay']), mask_d)
    optimizer3 = optax.masked(optax.adamw(hyper['learning_rate'], hyper['weight_decay']), mask_i)
else:
    optimizer1 = optax.adamw(hyper['learning_rate'], hyper['weight_decay'])
    optimizer2 = optax.adamw(hyper['learning_rate'], hyper['weight_decay'])
    optimizer3 = optax.adamw(hyper['learning_rate'], hyper['weight_decay'])


opt1 = optimizer1.init(params_used["lagrangian"])
opt2 = optimizer2.init(params_used["dissipative"])
opt3 = optimizer3.init(params_used["input_transform"])
# def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian,
#             dissipative_mat, input_mat, n_dof, time_step=None):
loss_fn = partial(
    delan_svd.loss_fn,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    input_mat=input_mat,
    n_dof=config_dof,
    time_step=time_step)

def update_fn(params, opt1, opt2, opt3, q, dq, tau, q_next, dq_next):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(
        params, q, dq, tau, q_next, dq_next
    )

    updates1, opt1 = optimizer1.update(grads["lagrangian"], opt1, params["lagrangian"])
    updates2, opt2 = optimizer2.update(grads["dissipative"], opt2, params["dissipative"])
    updates3, opt3 = optimizer3.update(grads["input_transform"], opt3, params["input_transform"])

    new_params = {
        "lagrangian": optax.apply_updates(params["lagrangian"], updates1),
        "dissipative": optax.apply_updates(params["dissipative"], updates2),
        "input_transform": optax.apply_updates(params["input_transform"], updates3),
    }
    return new_params, opt1, opt2, opt3, logs

update_fn = jax.jit(update_fn)
_, _, _, _, logs = update_fn(params_used, opt1, opt2, opt3, train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2])

t_build = time.perf_counter() - t0
print(f"Optimizer Build Time = {t_build:.2f}s")


'''
3. Start Training Loop:
'''
t0_start = time.perf_counter()

train_losses = {"forward_loss": [],
                "forward_var": []}

test_losses = {"forward_loss": [],
               "forward_var": []}


print("")
epoch_i = 0
step = 50
while epoch_i < hyper['max_epoch']:
    n_batches = 0
    logs = jax.tree.map(lambda x: x * 0.0, logs)

    for data_batch in mem:
        t0_batch = time.perf_counter()

        q, dq, tau, q_next, dq_next= [jnp.array(x) for x in data_batch]        
        params_used, opt1, opt2, opt3, batch_logs = update_fn(params_used, opt1, opt2, opt3, q, dq, tau, q_next, dq_next)

        # Update logs:
        n_batches += 1
        print(batch_logs)
        logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)
        t_batch = time.perf_counter() - t0_batch

    # Update Epoch Loss & Computation Time:
    epoch_i += 1
    logs = jax.tree.map(lambda x: x / n_batches, logs)

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        print("Epoch {0:05d}: ".format(epoch_i), end=" ")
        print("train: ", end=" ")
        print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
        print(f"For = {logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.2e}", end=";     ")
        train_losses["forward_loss"].append(logs['forward_mean'])
        train_losses["forward_var"].append(logs['forward_var'])

        # test loss computation:
        # (params, q, p, tau, q_next, p_next)
        test_loss, test_logs = loss_fn(params=params_used, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next)
        print("test: ", end=" ")
        print(f"For = {test_logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(test_logs['forward_var']):.2e}")
        test_losses["forward_loss"].append(test_logs['forward_mean'])
        test_losses['forward_var'].append(test_logs['forward_var'])

print(train_losses)
print(test_losses)

# Plot the loss picture
plt.style.use("seaborn-v0_8-whitegrid")
palette = plt.get_cmap('Set1')

iterations = list(range(0, hyper['max_epoch']+step, step))

f, ax = plt.subplots(1, 1)

# train forward loss
ax.plot(iterations, train_losses["forward_loss"], color=palette(1), label='train_forward_loss')
r1_forward = list(map(lambda x: x[0] - x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
r2_forward = list(map(lambda x: x[0] + x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
ax.fill_between(iterations, r1_forward, r2_forward, color=palette(1), alpha=0.2)


# test forward loss
ax.plot(iterations, test_losses["forward_loss"], color=palette(2), label='test_forward_loss')
t1_forward = list(map(lambda x: x[0] - x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
t2_forward = list(map(lambda x: x[0] + x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
ax.fill_between(iterations, t1_forward, t2_forward, color=palette(2), alpha=0.2)

ax.legend(loc='upper right')
ax.set_xlim(0, hyper['max_epoch']+1)
# ax.set_ylim(0, max(train_losses["forward_loss"][10:]))
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('loss', fontsize=12)
plt.show()



import csv
with open("./svd_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iterations", "train_loss", "train_variation", "test_loss", "test_variation"])

    for idx, i in enumerate(iterations):
        writer.writerow([
            i,
            train_losses['forward_loss'][idx],
            train_losses['forward_var'][idx],
            test_losses['forward_loss'][idx],
            test_losses['forward_loss'][idx]
        ])

if SAVE_MODEL:
    with open(f"./models/temp1.jax", "wb") as file:
        pickle.dump(
            {"epoch": epoch_i,
             "hyper": hyper,
             "params": params_used},
            file)