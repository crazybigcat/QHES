from Library.QuantumSimulator import Simulator
from Library.QuantumSimulator import Gate
from Library.BasicFunSZZ import wheel_function as wf
from Library.BasicFunSZZ import physical_fun as pf
from scipy.linalg import eigh
import torch as tc
import numpy as np
import time

# define device and dtype
device = 'cuda'
dtype = tc.complex64
results = list()

n_shots = 1000
range_value = [0.3, 0.7]
acc = 1e-5

# ancillary para
n_check = list()
for nn in range(20):
    n_check.append(2**nn)

# number of qubits
for nn_phi in list(range(6, 7, 1)):
    n_phi = nn_phi
    tmp_list = list()

    # define hamiltonian
    para = ((0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (0, 5), (3, 5))

    ham = pf.generate_hamiltonian(n_phi, 'MaxCut', para)
    # ham = ham / (n_phi - 1)

    eigv = eigh(ham, eigvals_only=True)
    zoom = (eigv[-1] - eigv[0])/(range_value[1] - range_value[0])
    shift = range_value[0] * zoom - eigv[0]
    ham = (ham + shift * np.eye(2 ** n_phi)) / zoom
    eigv0, eigs = eigh(ham)


    for nn_qpe in list(range(6, 8, 1)):

        n_qpe = nn_qpe
        # initialize repetition numbers
        n_dirac = wf.get_x_qpe(n_phi)
        w = wf.get_m_qpe(n_phi)
        n_repeat = int(np.pi / (4 * np.arcsin(2 ** (-n_phi / 2))) - 1 / 2)
        # n_repeat = int(np.pi / (2 * np.arcsin(2 ** (-n_phi / 2))) - 1)
        n_fre = round(np.log2(2*n_dirac + 1)) + 1

        n_qubit = n_phi + n_qpe + n_fre

        # initialize position

        position_phi = list(range(n_phi))
        position_qpe = list(range(n_phi, n_phi + n_qpe))
        position_f = list(range(n_phi + n_qpe, n_phi + n_qpe + n_fre))

        # guess value
        gv_min = 0
        gv_max = 1 - range_value[1]

        guess_shift0 = gv_min
        guess_shift1 = gv_max
        guess_shift = guess_shift0

        start = time.time()
        n_iter = 0

        while abs(guess_shift0 - guess_shift1) > acc:
            flag = 0
            # w = 1
            for ww in range(w):
                # ww = 3
                # guess_shift = 0.15
                ham_cal = ham + ww * np.eye(2 ** n_phi) / ((2 ** (n_qpe + 1)) * w)
                ham_cal = ham_cal + guess_shift * np.eye(2 ** n_phi)

                # eigs00 = eigh(ham_cal, eigvals_only=True)
                # print(eigs00)

                ham_cal = ham_cal * 2 * np.pi
                ham_cal = tc.from_numpy(ham_cal).to(device).to(dtype)

                a = Simulator.SimulatorProcess(n_qubit, device=device, dtype=dtype)
                unitary = Gate.time_evolution(ham_cal, 1, device=device, dtype=dtype)

                # initialize state

                init_gate = tc.from_numpy(eigs).to(device).to(dtype)

                # n_repeat = 0
                # print(n_repeat)

                for nn in range(n_repeat):
                    a.circuit.hadamard(position_phi)
                    a.circuit.add_single_gate(init_gate, position_phi, inverse=False)
                    a.circuit.qhc(unitary, position_phi, position_qpe, position_f, n_f=n_dirac, control=None, inverse=False)
                    if nn in n_check:
                        a.simulate()
                        res = a.sampling(n_shots, position=position_f, if_print=False)
                        right_str = '0' * n_fre
                        num = res[right_str]
                        if num / n_shots > 0.1:
                            # print(num/n_shots)
                            flag = 1
                            # print('flag')
                            break
                    a.circuit.not_gate(position_f)
                    a.circuit.phase_shift(np.pi, [position_f[0]], control=position_f[1:])
                    a.circuit.not_gate(position_f)
                    a.circuit.qhc(unitary, position_phi, position_qpe, position_f, n_f=n_dirac, control=None, inverse=True)
                    a.circuit.add_single_gate(init_gate, position_phi, inverse=True)
                    a.circuit.hadamard(position_phi)
                    a._state = -a.state
                    a.circuit.not_gate(position_phi)
                    a.circuit.phase_shift(np.pi, [position_phi[0]], control=position_phi[1:])
                    a.circuit.not_gate(position_phi)
                if flag == 1:
                    break
                a.circuit.hadamard(position_phi)
                a.circuit.add_single_gate(init_gate, position_phi, inverse=False)
                a.circuit.qhc(unitary, position_phi, position_qpe, position_f, n_f=n_dirac, control=None, inverse=False)
                a.simulate()
                res = a.sampling(n_shots, position=position_f, if_print=False)
                right_str = '0' * n_fre
                num = res[right_str]
                if num / n_shots > 0.1:
                    # print(num/n_shots)
                    flag = 1
                    # print('flag')
                    break
                del a
                # print(num / n_shots, w, guess_shift)
                # print(res)

            if flag == 1:
                guess_shift0 = guess_shift
                guess_shift = (guess_shift1 + guess_shift) / 2
            else:
                guess_shift1 = guess_shift
                guess_shift = (guess_shift0 + guess_shift) / 2
            # print(guess_shift, guess_shift0, guess_shift1, guess_shift0 - guess_shift1)
            n_iter = n_iter + 1
            # print(guess_shift, 'n_iter=', n_iter)
            guess_value = (0.5 - guess_shift) * zoom - shift
            error = abs(guess_value - eigv[0])

        print('guess_value=', guess_value, 'error=', error, 'n_iter=', n_iter, 'n_qpe=', n_qpe, 'n_phi=', n_phi, 'n_dirac=', n_dirac, 'w=', w, 'n_repeat=', n_repeat)
        tmp_list.append(error)
        end = time.time()
        print(str(device) + ' torch consume ' + str(end - start) + ' seconds, ' + str(dtype))
    results.append(tmp_list.copy())
    results0 = np.array(results);
    print(results0)
    np.savetxt('./maxcut_circuit.txt', results0)