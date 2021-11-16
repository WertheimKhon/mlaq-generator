import numpy as np
import subprocess
from lammps_simulator import Simulator
from lammps_simulator.computer import SlurmGPU, SlurmCPU
from run_torch_model import create_dataloader, RunTorchCNN
from simplexgrid import SimplexGrid, SeedGenerator
from pathlib import Path
from data_analyzer import Dataset
import os
import shutil
import torch
import sys


class Generator:
    """The goal is for the class to generate a project which consists of data

    1) Create a machine learning model and train it on a base dataset
    2) Create new samples that the network has not seen
    3) Use ML model to do predictions on the unseen data
    4) Choose samples to realize (simulate and measure values)
    5) Retrain the ML model
    6) Repeat steps 2-4

    Required functions:
        a) creating samples
        b) perform predictions
            - use subprocess to wait for predictions to finish
        c) choose samples by some user specified criteria
        d)



    sbatch job.sh
    # starts e.g. 50 simulations with job id ranging from 1-50
    sbatch --dependency=afterok:$(sbatch --parsable job.sh)
    """

    def __init__(self, paths, criterion=None, params=None,
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
        self.generation = 0
        self.criterion = criterion
        self.params = params
        self.optimizer = optimizer
        self.epochs = epochs

        # self.x = np.load(self.path_features)
        # self.y = np.load(self.path_targets)

        return

    def initiate(self, path, path_features, path_targets):
        """Initiates the project with starting folder and dataset.

        :param path:
        :type path:
        """
        assert self.generation == 0, \
            'Cannot initiate project if it has already started'
        if isinstance(path, str):
            path = Path(path)

        # Create directories needed for generation 0
        try:
            (path / 'gen0' / 'ml').mkdir(parents=True)
            (path / 'gen0' / 'data').mkdir()
        except:
            i = 0
            while True:
                try:
                    p = Path(str(path) + f'_{i}')
                    (p / 'gen0' / 'ml').mkdir(parents=True)
                    (p / 'gen0' / 'data').mkdir()
                    path = p
                    break
                except:
                    i += 1

        shutil.copy(path_features, path / 'gen0' / 'data' / 'features.npy')
        shutil.copy(path_targets, path / 'gen0' / 'data' / 'targets.npy')

        # Set current working directory. Updated as we increase the number of generations
        self.proj_direc = path / 'gen0'

        self.data_collect_job_id = None

    @staticmethod
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
                f.write(f'#SBATCH --{key}={val}\n')
            f.write('\n\n')
            f.write('echo $CUDA_VISIBLE_DEVICES\n')
            f.write(exec_cmd)

    @staticmethod
    def slurm_gpu(N, **kwargs):
        """Generates arguments for Slurm GPU jobscript.

        :param N: Number of GPUs
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

    @staticmethod
    def slurm_cpu(n_tasks, n_nodes, **kwargs):
        """Generates arguments for Slurm CPU jobscript.

        :param n_tasks: Number of tasks per CPU node
        :type n_tasks: int
        :param n_nodes: Number of CPU nodes
        :param n_nodes: int
        """
        args = {'partition': 'normal',
                'ntasks': n_tasks,
                'nodes': n_nodes,
                'output': 'slurm.out'}

        if kwargs:
            for key, val in kwargs.items():
                args[key] = val

        return args

    @staticmethod
    def generate_trainer(path, optimizer, criterion, params, epochs):
        """Generates a python script for training CNN
        """

        if optimizer is None:
            optimizer = 'torch.optim.Adam'
        if criterion is None:
            criterion = 'torch.nn.MSELoss()'
        if params is None:
            params = {'lr': 1e-4,
                      'weight_decay': 0.01}

        with open(path / 'ml' / 'run_cnn.py', 'w') as f:
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
            f.write(f'params = {params} \n\n')
            f.write('torch.backends.cudnn.benchmark = True \n\n')
            f.write('dataloader_train, dataloader_test=create_dataloader(features=features, targets=targets, batch_size=128, train_size=0.8, test_size=0.2, shuffle=True) \n\n')
            f.write(
                f'train_cnn = RunTorchCNN(model, epochs={epochs}, optimizer=optimizer, optimizer_args=params, dataloaders=(dataloader_train, dataloader_test), criterion=criterion, verbose=False, seed=42) \n\n')
            f.write('train_cnn() \n\n')
            f.write(f'train_cnn.save_model("{path / "ml" / "model.pt"}") \n')
            f.write(f'train_cnn.save_running_metrics("{path / "ml"}") \n')

    def train_cnn(self, file='run_cnn.py',
                  jobscript='job.sh',
                  method='slurm_gpu',
                  ):
        """Trains a CNN on data which has been measured by MD simulations of
        material structures. Queues a CNN to start when all MD simulations are
        completed.

        To do:
        subprocess which executes a python script that
            1) creates dataloader
            2) trains cnn
            3) saves dictionary of findings
            4) Wait argument when passing CNN training (stops the script from
               executing _all_ the generations at once)

        TO-DO:
            Figure out the correct argument for check_call
        """
        # Changing dir to ML
        os.chdir(self.proj_direc / 'ml')

        shutil.copy(self.path_cnn, Path.cwd() / 'cnn.py')

        if method == 'slurm_gpu':
            args = self.slurm_gpu(N=1)
        else:
            raise NotImplementedError('Method only supports training with GPU')

        args['job-name'] = 'cnn'

        # Generate script to run pytorch NN
        self.generate_trainer(self.proj_direc, self.optimizer,
                              self.criterion, self.params, self.epochs)
        # Generate jobscript and store in .../ml
        self.generate_jobscript(args,
                                'python3 run_cnn.py',
                                Path.cwd() / 'job.sh')

        if self.data_collect_job_id is not None:
            dependency = f'--dependency=afterok:{self.data_collect_job_id}'
        else:
            dependency = None

        sshProcess = subprocess.Popen(['ssh', '-tt', 'bigfacet'],  # , ';', f"cd {self.proj_direc}/ml", ';',  f"sbatch job.sh", ';', "logout"],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      bufsize=0)  # as sshProcess:

        # out, err = sshProcess.communicate(f"""
        #     cd {self.proj_direc}/ml
        #     sbatch {dependency} --wait job.sh
        # """)
        # print(out, err)
        sshProcess.stdin.write(f"cd {self.proj_direc}/ml\n")
        sshProcess.stdin.write("ls\n")
        # if dependency is None:
        sshProcess.stdin.write(f"sbatch job.sh\n")
        # else:
        #     sshProcess.stdin.write(f"sbatch {dependency} job.sh\n")
        sshProcess.stdin.write("logout\n")
        # result = sshProcess.stdout.readlines()
        # if result == []:
        #     error = sshProcess.stderr.readlines()
        #     print(sys.stderr, "ERROR: %s" % error)
        # else:
        #     print(result)
        sshProcess.stdin.close()

        for line in sshProcess.stdout:
            if line == "END\n":
                break
            print(line, end="")

        return

    @ staticmethod
    def generate_sample_creator(path, parameters):
        """Creates python file which executes the creation of new samples
        """

        with open(path / 'data' / 'new_samples', 'w') as f:
            f.write(
                'from simplexgrid import CreateMultipleSimplexGrids, SeedGenerator\n')
            f.write('import numpy as np\n')
            f.write('np.random.seed(42)\n\n')
            f.write('seedgen = SeedGenerator(start=1, step=1)\n\n')
            f.write(f'octaves = {parameters["octaves"]}\n')
            f.write(f'scales = {parameters["scales"]}\n')
            f.write(f'thresholds = {parameters["thresholds"]}\n')
            f.write(f'bases = {parameters["bases"]}\n')
            f.write(f'')

    # def create_new_samples(self, parameters):
    #     """Create new samples to perform predictions on.
    #
    #     ssh to egil
    #     sbatch cpu job
    #         - pass **parameters to simplexgrid
    #         - creates new samples
    #     """
    #
    # def prediction_new_samples(self):
    #     """Performs predictions on new samples. Runs on GPU
    #     """
    #
    #     return
    #
    # def choose_samples(self):
    #     """Choose 100 strongest and 100 weakest.
    #     """
    #
    #     return
    #
    # def execute_lammps_files(self, lmp_args, slurm_args, var, N):
    #     self.job_ids = np.zeros(N)
    #
    #     for i in range(N):
    #         slurm_args['job_name'] = slurm_args['job_name'] + f'{i}'
    #         var['datafile'] = f'{i}.data'
    #
    #         computer = SlurmGPU(lmp_exec='lmp',
    #                             slurm_args=slurm_args,
    #                             lmp_args=lmp_args,
    #                             job_script='job.sh')
    #         sim = Simulator(directory=self.path_data / 'run' / f'{i}')
    #         sim.copy_to_wd(self.path_data / f'{i}.data', self.path_potential)
    #         sim.set_input_script(self.path_lmp, **var)
    #         self.job_ids[i] = sim.run(computer=computer)
    #
    #     return
    #
    # @staticmethod
    # def create_job_collect_data(path):
    #
    # def gather_samples(self, data_paths, output_path):
    #     """Gathers samples from different data paths.
    #     """
    #     path_new_features = data_paths['new_features']
    #     path_new_targets = data_paths['new_targets']
    #
    #     jobid_string = ':'.join(map(str, self.job_ids))
    #     dependency = f'--dependency=afterok:{jobid_string}'
    #
    #     # Execute program which opens the existing features/targets and appends
    #     # the new features/targets, and then creates dataloaders to be used
    #     # during training
    #     file_collect_data = 'job_collect_data.sh'
    #     var = f'{data_path}'
    #
    #     # str(subprocess.check_output(["sbatch", self.jobscript], stderr=stderr))
    #     output = str(subprocess.check_output(
    #         ['sbatch', f'{dependency}', f'{file_collect_data}', f'{var}'],
    #         stderr=subprocess.PIPE))
    #     self.gather_samples_job_id = str(re.findall('([0-9]+)', output)[0])
    #
    # def create_direcs(self, gen):
