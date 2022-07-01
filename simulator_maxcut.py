from Library.QuantumSimulator import Simulator, Circuit, Gate
from Library.BasicFunSZZ import physical_fun as pf
from scipy.linalg import eigh
import torch as tc
import numpy as np
import time

# define device and dtype
device = 'cuda'
dtype = tc.complex64
results = list()

n_shots = 10000
range_value = [0, 0.25]
acc = 1e-5

# number of qubits
n_phi = 6
para = ((0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (0, 5), (3, 5))
# para = ((0, 1), (1, 2))
# para = ((0, 1), )

tmp_list = list()

# check

n_check = list()
for nn in range(20):
    n_check.append(2**nn)


# define hamiltonian
ham = pf.generate_hamiltonian(n_phi, 'MaxCut', para)
# ham = ham / (n_phi - 1)

eigv = eigh(ham, eigvals_only=True)
# print(eigv)
zoom = (eigv[-1] - eigv[0])/(range_value[1] - range_value[0])
shift = range_value[0] * zoom - eigv[0]
ham = (ham + shift * np.eye(2 ** n_phi)) / zoom
eigv0, eigs = eigh(ham)
right_state = tc.tensor(eigs[:, 0]).to(device).to(dtype).reshape(-1)

for nn_fre in list(range(6, 8, 1)):

    n_coin = 1
    n_repeat = int(np.pi / (4 * np.arcsin(2 ** (-n_phi / 2))) - 1 / 2)
    n_fre = nn_fre
    n_dirac = 2 ** n_fre - 1

    n_qubit = n_phi + n_coin + n_fre

    # initialize position

    position_phi = list(range(n_phi))
    position_coin = list(range(n_phi, n_phi + n_coin))
    position_f = list(range(n_phi + n_coin, n_phi + n_coin + n_fre))

    start = time.time()

    ham_cal = ham * 2 * np.pi
    ham_cal = tc.from_numpy(ham_cal)

    a = Simulator.SimulatorProcess(n_qubit, device=device, dtype=dtype)

    # initialize state

    init_gate = tc.from_numpy(eigs).to(device).to(dtype)
    unitary = Gate.time_evolution(ham_cal, 1, device=device, dtype=dtype)

    init_circuit = Circuit.Circuit(n_qubit, device=device, dtype=dtype)
    init_circuit.hadamard(position_phi)
    init_circuit.add_single_gate(init_gate, position_phi)
    init_circuit.qdc(unitary, position_phi, position_coin, position_f, n_f=n_dirac, inverse=False)
    oracle_circuit = Circuit.Circuit(n_qubit, device=device, dtype=dtype)
    oracle_circuit.not_gate(position_coin)
    oracle_circuit.phase_shift(np.pi, position_coin)
    oracle_circuit.not_gate(position_coin)
    diffusion_circuit = Circuit.Circuit(n_qubit, device=device, dtype=dtype)
    diffusion_circuit.not_gate(position_phi)
    diffusion_circuit.phase_shift(np.pi, [position_phi[0]], control=position_phi[1:])
    diffusion_circuit.not_gate(position_phi)

    for nn in range(n_repeat):
        a.circuit.compose(init_circuit)
        if nn in n_check:
            a.simulate()
            res = a.sampling(n_shots, if_print=False)
            counted_res = a.count_sample(res, '0', position_coin, if_print=False)
            num_right = 0
            for value in counted_res.values():
                num_right = num_right + value
            ratio = num_right / n_shots
            # print(ratio)
            if ratio > 0.1:
                print(counted_res)
                break
        a.circuit.compose(oracle_circuit)
        a.circuit.compose(init_circuit, inverse=True)
        a.circuit.compose(diffusion_circuit)

    a.circuit.compose(init_circuit)
    if ratio <= 0.1:
        a.simulate()
        res = a.sampling(n_shots, if_print=False)
        counted_res = a.count_sample(res, '0', position_coin, if_print=False)
        print(counted_res)
    end = time.time()
    print(str(device) + ' torch consume ' + str(end - start) + ' seconds, ' + str(dtype))