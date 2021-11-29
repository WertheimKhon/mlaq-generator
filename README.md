# mlaq-generator
This package performs a search algorithm for finding strong and weak material structures of an molecular dynamics (MD) system created using Simplex noise. Each 3D system is represented as a 2D image.  Measurements for each material structure (sample) is done by molecular dynamics simulations using LAMMPS. The search algorithm uses a convolutional neural network (CNN) to create a model, where the strength of the system is the target value and 2D images of the MD system is the input/regressor.

For the time being this program is far from generalized, and is considered to be in development. Name comes from machine learning on alpha-quartz.

# Requirements
- torch
- lammps_simulator (https://github.com/evenmn/lammps-simulator)
- run_torch_model (https://github.com/chdre/run-torch-model)
- simplexgrid (https://github.com/chdre/simplexgrid)
- molecular_builder (https://github.com/chdre/molecular-builder - forked version which includes an extra function)
- data_analyzer (https://github.com/chdre/data-analyzer)

# Install 
Install using pip:
```
pip install git+https://github.com/chdre/mlaq-generator
```

# Usage
First we need to initiate Slurm and LAMMPS arguments
```
slurm_args = {"job-name": "",
              "partition": "normal",
              "ntasks": 1,
              "cpus-per-task": "2",
              "gres": "gpu:1",
              "output": "slurm.out"}
lmp_args = {"-pk": "kokkos newton on neigh half binsize 7.5",
            "-k": "on g 1",
            "-sf": "kk"}
```
Then we set the parameters for the Simplex noise as a dictionary. Since the package, at present time, creates Python scripts from inside itself we initiate all parameters which consists of more than one value as a string, which is then evaluated in the Python script.
```
params = {'octaves': 'np.arange(1, 4, 1)',
          'scales': 'np.arange(10, 51, 1)',
          'thresholds': 'np.arange(0, 1, 0.01)',
          'l1': 140, # Length of MD system
          'l2': 70,  # See above
          'n1': 200, # Number of grid cells for Simplex noise
          'n2': 100, # See above
          'bases': 'np.arange(0, 100, 1)'}
```

If we are running on multiple CPU cores we divide the tasks evenly among the cores. For this case we need the total number of samples. N is the total number of Simplex noise samples for each combination of Simplex parameters (each with a specific seed and base).
```
N = 100 
total_samples = eval(params['octaves']).shape[0] * \
                eval(params['scales']).shape[0] * \
                eval(params['thresholds']).shape[0] * \
                N
```

To initiate the module we need to point it towards the paths for LAMMPS scripts, potetial file, CNN, features and target values.

```
paths = {'lammps': '/some/path/lammps.script',
        'potential': '/some/path/potential.file',
        'cnn': '/some/path/cnn.py',
        'features': '/some/path/features.npy',
        'targets': '/some/path/targets.npy'}
```
The module is then initiated with

```
import generator

gen = generator.generator(paths, epochs=2500)
gen.initiate(Path('/path_of_new_project/'))
```
The module has now set of various folders inside a main folder for the initial generation, all neatly stored inside folder '/path_of_new_project/gen0'. 

To train the model we simply call

```
gen.train_cnn()
```

We can then create new samples we a given seed, Simplex noise parameters for n_task number of cores and n_nodes nodes. N is the number of samples for each combination of Simplex parameters.

```
gen.create_new_samples(initial_seed=0, parameters, n_tasks, n_nodes, N)
```

Having stored the new samples we can perform predictions and said samples and choose the 100 predictively weakest and 100 predictively strongest. 

```
gen.prediction_new_samples(n_tasks)
gen.choose_samples(atoms_cutspace, atoms_outside, N)
```
atoms_cutspace and atoms_outside are atomic simulation environment MD objects, containing information about the MD system. This is project-specific and will be changed in due time. 

To predict/measure the actual strength of the system we execute LAMMPS scripts and gather the data by the following two functions respectively
```
gen.execute_lammps_files(lmp_args, slurm_args, var, N)
gen.get_measured_strength(N, window_length=1001)
```

Now we are finished with a single generation and can then set up the folders for the next iteration of the generative algorithm

```
gen.next_generation()
```
