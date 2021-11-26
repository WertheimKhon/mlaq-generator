import numpy as np
import subprocess
from lammps_simulator import Simulator
from lammps_simulator.computer import SlurmGPU, SlurmCPU
from run_torch_model import create_dataloader, RunTorchCNN
from simplexgrid import SimplexGrid, SeedGenerator
from molecular_builder.geometry import ImageToSurfaceGeometry
from molecular_builder.core import carve_geometry
from pathlib import Path
from data_analyzer import Dataset
import os
import shutil
import torch
import sys
from cnn import Model
import pickle


class Generator:
    """Class which performs a "search algorithm". In short we
        1) train a NN on a dataset
        2) use the trained model to perform predictions on unseen data
        3) select some samples based on the predictions
        4) measure the true values
        5) add the new data points to the full dataset
        6) 1-5 for a given number of generations
    Points 1-5 is called a generation. The goal is to iteratively create a
    model which learns a specific task (i.e. locating samples which maximizes
    the target value, or class).

    At the present time the module is specific for molecular dynamics using
    LAMMPS and CNN.

    :param paths: Paths for potential, lammps and cnn files
    :type paths: dict
    :param criterion: Loss function for CNN. Defaults to None which leads to
                      torch.nn.MSELoss()
    :type criterion: str
    :param optimizer_args: Arguments for the chosen optimizer. Defaults to None,
                           which leads to lr 0.001, wd 0.01.
    :type optimizer_args: dict
    :param optimizer: Optimizer for CNN. Defaults to None, which leads to
                      torch.optim.Adam
    :type optimizer: str
    :param epochs: Number of iterations for training CNN
    :type epochs: int
    """

    def __init__(self, paths, criterion=None, optimizer_args=None,
                 optimizer=None, epochs=2500):

        if not isinstance(criterion, str) and criterion is not None:
            raise ValueError(
                'criterion must be of type str specifying which loss functions to use')
        if not isinstance(optimizer, str) and optimizer is not None:
            raise ValueError(
                'optimizer must be of type str specifying which optimizer to use')

        self.job_ids = None
        self.generation = 0
        self.path_potential = Path(paths['potential'])
        self.path_lmp = Path(paths['lammps'])
        self.path_cnn = Path(paths['cnn'])
        self.path_initial_features = paths['features']
        self.path_initial_targets = paths['targets']
        self.generation = 0
        self.criterion = criterion
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.epochs = epochs

        # Initiate a dataset
        x = np.load(Path(paths['features'])).astype(np.int8)
        y = np.load(Path(paths['targets'])).astype(np.int8)
        self.dataset = Dataset(x, y)

        return

    def initiate(self, path, jump_in_later_gen=False):
        """Initiates the project with starting folder and dataset.

        :param path: Path to store project
        :type path: pathlib.PosixPath or str
        """
        assert self.generation == 0, \
            'Cannot initiate project if it has already started'
        if isinstance(path, str):
            path = Path(path)
        if not jump_in_later_gen:  # WIP
            # Create directories needed for generation 0
            try:
                path.mkdir()
            except:
                i = 1
                while i < 101:
                    try:
                        p = Path(str(path) + f'{i}')
                        p.mkdir()
                        path = p
                        break
                    except:
                        i += 1
                if i == 100:
                    raise ValueError(f'{path} is not a valid path')

            (path / 'gen0' / 'ml' / 'training').mkdir(parents=True)
            (path / 'gen0' / 'ml' / 'predictions').mkdir()
            (path / 'gen0' / 'data' / 'samples').mkdir(parents=True)
            (path / 'gen0' / 'simulations' / 'weakest').mkdir(parents=True)
            (path / 'gen0' / 'simulations' / 'strongest').mkdir()

        shutil.copy(self.path_initial_features, path /
                    f'gen{self.generation}' / 'data' / 'features.npy')
        shutil.copy(self.path_initial_targets, path /
                    f'gen{self.generation}' / 'data' / 'targets.npy')

        # Set current working directory. Updated as we increase the number of generations
        self.proj_direc = path
        self.gen_direc = path / 'gen0'

        self.data_collect_job_id = None

    def next_generation(self):
        """Sets up for the next generation.
        """
        self.generation += 1
        self.gen_direc = self.proj_direc / f'gen{self.generation}'

        (self.gen_direc / 'ml' / 'training').mkdir(parents=True)
        (self.gen_direc / 'ml' / 'predictions').mkdir()
        (self.gen_direc / 'data' / 'samples').mkdir(parents=True)
        (self.gen_direc / 'simulations' / 'weakest').mkdir(parents=True)
        (self.gen_direc / 'simulations' / 'strongest').mkdir()

        self.dataset.save_data(self.gen_direc / 'data', y='y')

        print(f'===============================================================')
        print(f'Next generation: {self.generation}')

    @ staticmethod
    def generate_jobscript(arguments, exec_cmd, path):
        """Generates a jobscript with givens arguments.

        :param arguments: Arguments for jobscript
        :type arguments: dict
        :param exec_cmd: Command to execute script
        :type exec_cmd: str
        :param path: Path to save the jobscript
        :type path: pathlib.PosixPath
        """
        with open(path, 'w') as f:
            f.write('#!/bin/bash\n')
            for key, val in arguments.items():
                if val is None:
                    f.write(f'#SBATCH --{key}')
                else:
                    f.write(f'#SBATCH --{key}={val}\n')
            f.write('\n\n')
            f.write(exec_cmd)

    @ staticmethod
    def slurm_gpu(N=1, **kwargs):
        """Generates arguments for Slurm GPU jobscript.

        :param N: Number of GPUs. Defaults to 1.
        :type N: int
        """
        args = {'partition': 'normal',
                'ntasks': 1,
                'gres': f'gpu:{N}',
                'cpus-per-task': 2,
                'output': 'slurm.out'}
        if kwargs:
            for key, val in kwargs.items():
                args[key] = val

        return args

    @ staticmethod
    def slurm_cpu(n_tasks, n_nodes, tasks_per_node, **kwargs):
        """Generates arguments for Slurm CPU jobscript.

        :param n_tasks: Number of tasks per CPU node
        :type n_tasks: int
        :param n_nodes: Number of CPU nodes
        :param n_nodes: int
        """
        args = {'partition': 'cpu',
                'ntasks': n_tasks,
                'nodes': n_nodes,
                'ntasks-per-node': tasks_per_node,
                'output': 'slurm.out'}

        if kwargs:
            for key, val in kwargs.items():
                args[key] = val

        return args

    @ staticmethod
    def generate_trainer(path, optimizer, criterion, optimizer_args, epochs):
        """Generates a python script for training CNN

        :param criterion: Loss function for CNN. Defaults to None which leads to
                          torch.nn.MSELoss()
        :type criterion: str
        :param optimizer_args: Arguments for the chosen optimizer. Defaults to None,
                               which leads to lr 0.001, wd 0.01.
        :type optimizer_args: dict
        :param optimizer: Optimizer for CNN. Defaults to None, which leads to
                          torch.optim.Adam
        :type optimizer: str
        :param epochs: Number of iterations for training CNN
        :type epochs: int
        """

        if optimizer is None:
            optimizer = 'torch.optim.Adam'
        if criterion is None:
            criterion = 'torch.nn.MSELoss()'
        if optimizer_args is None:
            optimizer_args = {'lr': 1e-4,
                              'weight_decay': 0.01}

        with open(path / 'ml' / 'training' / 'run_cnn.py', 'w') as f:
            f.write('import torch \n')
            f.write('from run_torch_model import create_dataloader, RunTorchCNN \n')
            f.write('from cnn import Model \n')
            f.write('import numpy as np \n\n')
            f.write('torch.manual_seed(42) \n\n')
            f.write('model = Model() \n\n')
            f.write(
                f'features = np.load("{path / "data" / "features.npy"}") \n')
            f.write('features = features[:, np.newaxis, :, :] \n')
            f.write(
                f'targets = np.load("{path / "data" / "targets.npy"}") \n')
            f.write(f'targets = targets[:, np.newaxis] \n\n')
            f.write(f'optimizer = "{optimizer}" \n')
            f.write(f'criterion = {criterion} \n')
            f.write(f'optimizer_args = {optimizer_args} \n\n')
            f.write('torch.backends.cudnn.benchmark = True \n\n')
            f.write('dataloader_train, dataloader_test = create_dataloader(features=features, targets=targets, batch_size=128, train_size=0.8, test_size=0.2, shuffle=True, pin_memory=True) \n\n')
            f.write(
                f'train_cnn = RunTorchCNN(model, epochs={epochs}, optimizer=optimizer, optimizer_args=optimizer_args, dataloaders=(dataloader_train, dataloader_test), criterion=criterion, verbose=False, seed=42) \n\n')
            f.write('train_cnn() \n\n')
            f.write(
                f'train_cnn.save_model("{path / "ml" / "training" / "model.pt"}") \n')
            f.write(
                f'train_cnn.save_running_metrics("{path / "ml" / "training"}") \n')

    def train_cnn(self, file='run_cnn.py',
                  jobscript='job.sh',
                  method='slurm_gpu',
                  ):
        """Trains a CNN on data which has been measured by MD simulations of
        material structures. Queues a CNN to start when all MD simulations are
        completed.

        :param file: Name of file containing CNN. Defaults to run_cnn.py.
        :type file: str
        :param jobscript: Name of jobscript. Defaults to job.sh.
        :type jobscript: str
        :param method: Which method to use for jobscript. Defaults to slurm_gpu,
                       meaning the training will be done on a GPU using slurm
                       queue.
        :type method: str
        """
        os.chdir(self.gen_direc / 'ml' / 'training')
        shutil.copy(self.path_cnn, self.gen_direc /
                    'ml' / 'training' / 'cnn.py')

        if method == 'slurm_gpu':
            args = self.slurm_gpu(N=1)
        else:
            raise NotImplementedError('Method only supports training with GPU')

        args['job-name'] = 'cnn'

        # Generate script to run pytorch NN
        self.generate_trainer(self.gen_direc, self.optimizer,
                              self.criterion, self.optimizer_args, self.epochs)

        if self.data_collect_job_id is not None:
            args['dependency'] = f'afterok:{self.data_collect_job_id}'
        args['wait'] = None

        # Generate jobscript and store in .../ml
        self.generate_jobscript(arguments=args,
                                exec_cmd=f'python3 run_cnn.py',
                                path=self.gen_direc / 'ml' / 'training' / 'job.sh')

        output = subprocess.check_output(['sbatch', 'job.sh'],
                                         stderr=subprocess.PIPE)

        print(f'Gen. {self.generation}: training complete')
        os.chdir(self.proj_direc)

        return

    @ staticmethod
    def generate_samples_creator(path, parameters, initial_seed, N, iter):
        """Creates python file which executes the creation of new samples

        :param parameters: Parameters for Simplex noise.
        :type params: dict
        :param initial_seed: Initial seed for Simplex noise.
        :type initial_seed: int
        :param N: Number of samples for each set of params.
        :type N: int
        :param iter: Iteration of Simplex algorithm. Typically tells which
                     process created the noise.
        :type iter: int
        """

        with open(path / 'data' / 'samples' / f'new_samples{iter}.py', 'w') as f:
            f.write(
                'from simplexgrid import CreateMultipleSimplexGrids, SeedGenerator\n')
            f.write('import pickle\n')
            f.write('import numpy as np\n\n')

            f.write('np.random.seed(42)\n\n')

            f.write(f'octaves = {parameters["octaves"]}.astype(np.int8)\n')
            f.write(f'scales = {parameters["scales"]}.astype(np.int8)\n')
            f.write(
                f'thresholds = {parameters["thresholds"]}.astype(np.float16)\n')
            f.write(f'bases = {parameters["bases"]}.astype(int)\n')
            f.write(f'l1 = {parameters["l1"]}\n')
            f.write(f'l2 = {parameters["l2"]}\n')
            f.write(f'n1 = {parameters["n1"]}\n')
            f.write(f'n2 = {parameters["n2"]}\n')

            f.write(f'N = {N}\n\n')

            f.write(
                f'seedgen = SeedGenerator(start={initial_seed}, step=1)\n\n')

            f.write('def criterion(x):\n')
            f.write('   x.sum() / (x.shape[0] * x.shape[1])\n')
            f.write('   N = x.shape[0]\n')
            f.write('   n_12 = x.shape[1] * x.shape[2]\n')
            f.write('   porosity = x.sum(axis=(1, 2)) / n_12\n')
            f.write('   inds = np.logical_and(porosity > 0.09, porosity < 0.51)\n')
            f.write('   if inds.sum() == 0:\n')
            f.write('       return np.array([])\n')
            f.write('   else:\n')
            f.write('       inds = np.arange(N)[inds]\n')
            f.write('   return inds\n\n')

            f.write(
                'simpgrid = CreateMultipleSimplexGrids(scales, thresholds, bases, octaves, l1, l2, n1, n2, N, seedgen, criterion=criterion)\n')
            f.write('dictionary = simpgrid()\n\n')

            f.write(
                f'print("Finished samples for iteration {iter}", flush=True)\n\n')

            f.write(
                f'with open("{path}/data/samples/samples_{iter}", "wb") as f:\n')
            f.write(
                f'  pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)\n')

        return

    def create_new_samples(self, initial_seed, parameters, n_tasks, n_nodes, N,
                           tasks_per_node=16, method='slurm_cpu',
                           jobname='new_samples', **kwargs):
        """Create new samples to perform predictions on.

        :param initial_seed: Initial seed for Simplex noise.
        :type initial_seed: int
        :param parameters: Parameters for Simplex noise.
        :type params: dict
        :param n_tasks: Number of tasks for Slurm queue.
        :type n_tasks: int
        :param n_nodes: Number of nodes for Slurm queue.
        :type n_nodes: int
        :param N: Number of samples for each set of params.
        :type N: int
        :param tasks_per_node: Tasks per node for Slurm queue.
        :type tasks_per_node: int
        :param method: Which method to use for jobscript. Defaults to slurm_cpu,
                       meaning the training will be done on a CPU using slurm
                       queue.
        :type method: str
        :param jobname: Name of job. Defaults to new_samples.
        :type jobname: str

        TO-DO:
            - Rewrite while flag part of code to make it as vectorized as
              possible. Can use int division to remove some iterations of the
              for loop, but what more?
        """
        # Split tasks evenly among nodes
        tmp = N // n_tasks
        # Creating number of samples for each node
        N_per_node = tmp * np.ones(n_tasks, dtype=int)
        # If number of tasks does not split evenly among nodes
        if N - tmp * n_tasks != 0:
            difference = N - int(tmp * n_tasks)
            flag = True
            while flag:
                # Manually add 1 to each element in N_per_node until difference = 0
                for i in range(n_tasks):
                    N_per_node[i] += 1
                    difference -= 1
                    if difference == 0:
                        flag = False
                        break

        N_params = eval(parameters['octaves']).shape[0] * eval(
            parameters['scales']).shape[0] * eval(parameters['thresholds']).shape[0]

        init_seeds = [initial_seed]
        for i in range(1, n_tasks):
            init_seeds.append(
                init_seeds[i - 1] + N_params * N_per_node[i - 1])
        init_seeds = np.array(init_seeds)
        # init_seeds = initial_seed + \
        #     (N_per_node * N_params * np.arange(n_tasks))

        for i in range(n_tasks):
            self.generate_samples_creator(
                self.gen_direc, parameters, init_seeds[i], N_per_node[i], i)

        try:
            args = eval('self.' + method)(n_tasks,
                                          n_nodes, tasks_per_node, **kwargs)
        except:
            raise NotImplementedError(
                f'{method} is not a valid argument for method')

        args['job-name'] = jobname
        args['wait'] = None

        command = f"""for i in $(seq 0 {n_tasks - 1}); do
    srun --exclusive --ntasks=1 --nodes=1 --mem=2gb --output=slurm.out$i python3 new_samples$i.py &
done
wait
        """

        self.generate_jobscript(
            args, command, self.gen_direc / 'data' / 'samples' / 'job.sh')

        sshProcess = subprocess.Popen(['ssh', '-tt', 'egil'],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      universal_newlines=True,
                                      bufsize=0)

        sshProcess.stdin.write(f"cd {self.gen_direc}/data/samples\n")
        sshProcess.stdin.write(f"sbatch job.sh\n")
        sshProcess.stdin.write("logout\n")
        sshProcess.wait()
        sshProcess.stdin.close()
        print(f'Gen. {self.generation}: created samples')
        # for line in sshProcess.stdout:
        #     if line == "END\n":
        #         break
        #     print(line, end="")

    @ staticmethod
    def gather_new_samples(path, N):
        """Collects samples created by multiple instances of SimplexGrid.

        :param path: Path of new samples.
        :type path: Pathlib.PosixPath
        :param N: Number of tasks, i.e. number of files where samples are
                  distributed.
        :type N: int
        """
        cwd = Path.cwd()
        os.chdir(path)
        files = [f'samples_{i}' for i in range(N)]
        dicts = []

        for file in files:
            with open(file, 'rb') as f:
                d = pickle.load(f)
            dicts.append(d)

        dictionary = dicts[0]
        dicts.remove(dicts[0])

        for d in dicts:
            for key, val in d.items():
                dictionary[key].extend(d[key])
        with open(path / 'all_samples_grids', 'wb') as f:
            pickle.dump(dictionary['grid'], f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        del dictionary['grid']
        with open(path / 'all_samples_parameters', 'wb') as f:
            pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.chdir(cwd)

    @staticmethod
    def create_preds_new_samples(ml_path, samples_path, epochs, optimizer,
                                 criterion, optimizer_args, N=32):
        """Creates a python script that performs

        :param ml_path: Path to ml directory.
        :type ml_path: pathlib.PosixPath or str
        :param samples_path: Path to samples.
        :type samples_path: pathlib.PosixPath or str
        :param epochs: Number of training iterations.
        :type epochs: int
        :param criterion: Loss function for CNN. Defaults to None which leads to
                          torch.nn.MSELoss()
        :type criterion: str
        :param optimizer_args: Arguments for the chosen optimizer. Defaults to None,
                               which leads to lr 0.001, wd 0.01.
        :type optimizer_args: dict
        :param optimizer: Optimizer for CNN. Defaults to None, which leads to
                          torch.optim.Adam
        :type optimizer: str
        :param N: Number of files to read. Defaults to 32 (16 threads on two
                  nodes).
        :type N: int
        """

        if optimizer is None:
            optimizer = 'torch.optim.Adam'
        if criterion is None:
            criterion = 'torch.nn.MSELoss()'
        if optimizer_args is None:
            optimizer_args = {'lr': 1e-4,
                              'weight_decay': 0.01}

        with open(ml_path / 'predictions' / 'preds.py', 'w') as f:
            f.write('from run_torch_model import RunTorchCNN\n')
            f.write('import torch\n')
            f.write('import pickle\n')
            f.write('from cnn import Model\n')
            f.write('import numpy as np\n\n')

            f.write('model = Model()\n')
            f.write(
                f'run_model = RunTorchCNN(model=model, epochs={epochs}, optimizer="{optimizer}", optimizer_args={optimizer_args}, dataloaders=None, criterion={criterion})\n')
            f.write(f'run_model.load_model("{ml_path}/training/model.pt")\n')
            f.write(
                f'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n')
            f.write(f'model.to(device)\n')
            f.write('predictions = []\n\n')

            f.write(f'for i in range({N}):\n')
            f.write(
                f'  with open(f"{samples_path}/samples_" + str(i), "rb") as f:\n')
            f.write('       features = np.array(pickle.load(f)["grid"])\n')
            f.write(f'  n = features.shape[0]\n')
            f.write(f'  r = np.arange(0, n, 5000)\n')
            f.write(
                f'  r = np.concatenate((r, np.array([n])), axis=0).astype(int)\n\n')
            f.write(f'  for i in range(r.shape[0] - 1):\n')
            f.write(
                f'      tensor = torch.Tensor(features[r[i]:r[i + 1], np.newaxis, :, :]).to(device)\n')
            f.write(f'      p = run_model.predict(tensor)\n')
            f.write(
                f'      p = p.detach().cpu().numpy().astype(np.float32)\n')
            f.write(f'      predictions.append(p)\n\n')

            f.write(f'predictions = np.concatenate(predictions, axis=0)\n\n')
            f.write(
                f'np.save("{ml_path}/predictions/predictions", predictions)\n')

    def prediction_new_samples(self, N=32):
        """Performs predictions on new samples. Runs on GPU

        :param N: Number of files to read. Defaults to 32 (16 threads on two
                  nodes).
        :type N: int
        """
        os.chdir(self.gen_direc / 'ml' / 'predictions')

        shutil.copy(self.path_cnn, self.gen_direc / 'ml' / 'predictions')

        self.gather_new_samples(self.gen_direc / 'data' / 'samples', N)

        self.create_preds_new_samples(self.gen_direc / 'ml',
                                      self.gen_direc / 'data' / 'samples',
                                      epochs=self.epochs,
                                      optimizer=self.optimizer,
                                      criterion=self.criterion,
                                      optimizer_args=self.optimizer_args,
                                      N=N)

        args = self.slurm_gpu()
        args['wait'] = None
        args['job-name'] = 'preds'
        self.generate_jobscript(args,
                                'python3 preds.py',
                                self.gen_direc / 'ml' / 'predictions' / 'job.sh',)

        output = subprocess.check_output(['sbatch', 'job.sh'],
                                         stderr=subprocess.PIPE)

        print(f'Gen. {self.generation}: predictions on samples complete')

        with open(self.gen_direc / 'data' / 'samples' / 'all_samples_parameters', 'rb') as f:
            self.features_params_new = pickle.load(f)

        self.predictions = np.load(
            self.gen_direc / 'ml' / 'predictions' / 'predictions.npy')[:, 0]

        os.chdir(self.proj_direc)

    @staticmethod
    def check_porosity_samples(image, N_atoms, atoms, normal=(0, 1, 0)):
        """Check the actual porosity for the chosen samples.

        :param image: Image for CNN.
        :type image. ndarray
        :param N_atoms: Number of atoms before removing particles.
        :type N_atoms: int
        :param atoms: Object containing atom information.
        :type atoms: ASE.atoms
        :param normal: Normal vector to space where particles are removed.
                       Defaults to (0, 1, 0), i.e. removing particles in x-z
                       plane along y axis.
        :type normal: array_like
        :returns porosity: Porosity
        :rtype porosity: float
        :returns atoms_copy: Carved atomic system.
        :rtype atoms_copy: ASE.atoms
        """
        atoms_copy = atoms.copy()
        geometry = ImageToSurfaceGeometry(normal=normal, image=image)
        num_carved = carve_geometry(atoms_copy, geometry, side="out")
        porosity = num_carved / N_atoms

        return porosity, atoms_copy

    def choose_samples(self, atoms_cutspace, atoms, N=100):  # , criterions):
        """Choose samples by the N strongest and N weakest. The porosity
        is checked for each sample, making sure the sample has porosity
        between 0.1 and 0.5. The sample is then stored in simulations folder.

        :param atoms_cutspace, atoms: ASE object
        :type atoms_cutspace, atoms: atoms object
        :param N: Number of samples to proceed with. Defaults to 100.
        :type N: int
        """
        # if isinstance(criterions, list):
        #     inds = []
        #     for criterion in criterions:
        #         tmp = criterion(self.predictions)
        #         inds.append(tmp)
        #
        # else:
        #     inds = criterion(self.predictions)

        inds_sorted = np.argsort(self.predictions)
        inds_strongest = []
        inds_weakest = []

        N_atoms = len(atoms_cutspace)
        lx, _, lz = atoms.cell.cellpar()[:3]

        i = 0
        idx = 0
        porosity = []
        noise_grid_weakest = np.zeros((N, 200, 100), dtype=np.int8)
        while len(inds_weakest) < N:
            ind = inds_sorted[i]
            simpgrid = SimplexGrid(scale=self.features_params_new['scale'][ind],
                                   threshold=self.features_params_new['threshold'][ind],
                                   l1=lx,
                                   l2=lz,
                                   n1=200,
                                   n2=100)
            noise_grid = simpgrid(seed=self.features_params_new['seed'][ind],
                                  base=self.features_params_new['base'][ind])

            p, atoms_carved = self.check_porosity_samples(noise_grid,
                                                          N_atoms,
                                                          atoms_cutspace)
            if p < 0.5 and p > 0.1:
                inds_weakest.append(ind)
                new_atoms = atoms_carved + atoms
                new_atoms.write(self.gen_direc / 'simulations' / 'weakest' /
                                f'weakest_{idx}.data', format='lammps-data')
                noise_grid_weakest[idx] = noise_grid
                porosity.append(p)
                idx += 1
            i += 1
            if ind == inds_sorted[-1]:
                print('No weakest found')
                break

        i = -1
        idx = 0
        noise_grid_strongest = np.zeros((N, 200, 100), dtype=np.int8)
        while len(inds_strongest) < N:
            ind = inds_sorted[i]
            simpgrid = SimplexGrid(scale=self.features_params_new['scale'][ind],
                                   threshold=self.features_params_new['threshold'][ind],
                                   l1=lx,
                                   l2=lz,
                                   n1=200,
                                   n2=100)
            noise_grid = simpgrid(seed=self.features_params_new['seed'][ind],
                                  base=self.features_params_new['base'][ind])
            p, atoms_carved = self.check_porosity_samples(noise_grid,
                                                          N_atoms,
                                                          atoms_cutspace)

            if p < 0.5 and p > 0.1:
                inds_strongest.append(ind)
                new_atoms = atoms_carved + atoms
                new_atoms.write(self.gen_direc / 'simulations' / 'strongest' /
                                f'strongest_{idx}.data', format='lammps-data')
                noise_grid_strongest[idx] = noise_grid
                porosity.append(p)
                idx += 1
            i -= 1
            if ind == inds_sorted[0]:
                print('No strongest found')
                break

        assert not np.any(np.in1d(inds_strongest, inds_weakest)), \
            "Same geometry present in both strongest and weakest"

        self.dataset.extend_data(features=np.concatenate(
            (noise_grid_weakest, noise_grid_strongest), axis=0))

        np.save(self.gen_direc / 'data' / 'samples' /
                'inds_weakest', inds_weakest)
        np.save(self.gen_direc / 'data' / 'samples' /
                'inds_strongest', inds_strongest)
        np.save(self.gen_direc / 'data' / 'samples' /
                'porosity', np.asarray(porosity))

        print(f'Gen. {self.generation}: samples chosen')

        return

    def execute_lammps_files(self, lmp_args, slurm_args, var, N=100):
        """Executes the LAMMPS files chosen above.

        :param lmp_args: Lammps arguments.
        :type lmp_args: dict
        :slurm_args: Slurm arguments.
        :type slurm_args: dict
        :param var: Variables for Lammps script.
        :type var: dict
        :param N: Number of simulations.
        :type N: int
        """
        job_ids_w = np.zeros(N, dtype=int)

        for i in range(N):
            slurm_args['job-name'] = f'w{i}'
            var['datafile'] = f'weakest_{i}.data'

            computer = SlurmGPU(lmp_exec='lmp',
                                slurm_args=slurm_args,
                                lmp_args=lmp_args,
                                jobscript='job.sh')
            sim = Simulator(directory=str(self.gen_direc /
                            'simulations' / 'weakest' / 'run' / f'{i}'))
            sim.copy_to_wd(str(self.gen_direc / 'simulations' / 'weakest' /
                           f'weakest_{i}.data'), str(self.path_potential))
            sim.set_input_script(str(self.path_lmp), **var)
            job_ids_w[i] = sim.run(computer=computer)

        jobid_string = ':'.join(map(str, job_ids_w))

        job_ids_s = np.zeros(N, dtype=int)
        for i in range(N):
            slurm_args['job-name'] = f's{i}'
            slurm_args['dependency'] = f'afterok:{jobid_string}'
            var['datafile'] = f'strongest_{i}.data'

            computer = SlurmGPU(lmp_exec='lmp',
                                slurm_args=slurm_args,
                                lmp_args=lmp_args,
                                jobscript='job.sh')
            sim = Simulator(directory=str(self.gen_direc /
                            'simulations' / 'strongest' / 'run' / f'{i}'))
            sim.copy_to_wd(str(self.gen_direc / 'simulations' / 'strongest' /
                           f'strongest_{i}.data'), str(self.path_potential))
            sim.set_input_script(str(self.path_lmp), **var)
            job_ids_s[i] = sim.run(computer=computer)

            self.job_ids = np.concatenate((job_ids_w, job_ids_s))

        print(f'Gen. {self.generation}: LAMMPS simulations in queue')

        return

    @staticmethod
    def create_collect_data(path,
                            subpath='weakest',
                            N=100,
                            low=int(np.ceil(30 / 0.0005) / 10),
                            window_length=1001):
        """Path has to be path to generation folder

        :param path: Project path.
        :type path: str
        :param subpath: "Path" of subsamples. Defaults to weakest.
        :type subpath: str
        :param N: Number of simulations. Defaults to 100.
        :type N: int
        :param low: Initial index of lammps logfile. Typically the index after
                    equilibriation. Defaults to int(np.ceil(30 / 0.0005) / 10)),
                    30 ps runtime, dt 0.0005 and a print interval of 10.
        :type low: int
        :param window_length: Window length for savgol_filter. Defaults to 1001.
        :type window: int
        """
        with open(path / 'simulations' / subpath / 'collect_data.py', 'w') as f:
            f.write('import numpy as np\n')
            f.write('from lammps_logfile_reader import readLog\n')
            f.write('from pathlib import Path\n')
            f.write('from data_analyzer import Dataset\n\n')

            f.write(f'low = {low}\n')
            f.write(f'yield_stress = np.zeros({int(N)})\n\n')

            f.write(f'for i in range({N}):\n')
            f.write(f'  p = "{path}/simulations/{subpath}/run"\n')
            f.write('  logfile = f"{p}/{i}/log.lammps"\n')
            # f.write(
            #     f'  logfile = f"{path}/simulations/{subpath}/run/sim" + str(i) + "/log.lammps"\n')
            f.write(f'  logdict = readLog(logfile).read()\n')
            f.write(
                f'  pyy = -np.array(logdict.get("Pyy"), dtype=float)[low:] / 1e4\n')
            f.write(f'  dset = Dataset(np.zeros(pyy.shape), pyy)\n')
            f.write(f'  dset.prep_data_mlaq(window_length={window_length})\n')
            f.write(f'  yield_stress[i] = dset.get_ymax()\n\n')

            f.write(
                f'np.save(Path("{path}") / "simulations" / "{subpath}" / "yield_stress", yield_stress)')

    def get_measured_strength(self, N=100, window_length=1001):
        """Collects the strength measured by MD simulations of the 100 strongest
        and 100 weakest predicted samples.

        :param N: Number of simulations. Defaults to 100.
        :type N: int
        :param window_length: Window length for savgol_filter smoothing.
        :type window_length: int
        """

        # Creating python scripts to generate yield stress files
        self.create_collect_data(self.gen_direc,
                                 subpath='weakest',
                                 N=N,
                                 window_length=window_length)
        self.create_collect_data(self.gen_direc,
                                 subpath='strongest',
                                 N=N,
                                 window_length=window_length)

        args = self.slurm_gpu()

        # Creating a dependency for all the simulations, such that we do not try
        # to collect yield stress before it has been measured
        jobid_string = ':'.join(map(str, self.job_ids))
        args['dependency'] = f'afterok:{jobid_string}'
        args['wait'] = None
        args['job-name'] = 'collect'

        # Creating job scripts for creating yield stress files
        self.generate_jobscript(arguments=args,
                                exec_cmd='python3 collect_data.py',
                                path=self.gen_direc / 'simulations' / 'weakest' / 'job.sh')
        self.generate_jobscript(arguments=args,
                                exec_cmd='python3 collect_data.py',
                                path=self.gen_direc / 'simulations' / 'strongest' / 'job.sh')
        os.chdir(self.gen_direc / 'simulations' / 'weakest')
        output = subprocess.check_output(['sbatch', 'job.sh'],
                                         stderr=subprocess.PIPE)
        tmp1 = np.load('yield_stress.npy')

        os.chdir(self.gen_direc / 'simulations' / 'strongest')
        output = subprocess.check_output(['sbatch', 'job.sh'],
                                         stderr=subprocess.PIPE)
        tmp2 = np.load('yield_stress.npy')

        print(f'Gen. {self.generation}: yield for new samples stored to files')

        # Load new yield stresses and concatenate to single array, thereafter
        # add to dataset object

        self.targets_new = np.concatenate((tmp1, tmp2), axis=0)
        self.dataset.extend_data(targets=self.targets_new)

        os.chdir(self.proj_direc)
