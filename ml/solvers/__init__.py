from ml.solvers.multitask_solver import MultitaskSolver
from ml.solvers.simple_solver import SimpleSolver
from ml.solvers.transfer_solver import TransferSolver


def get_solver(config, args):
    if config.env.learning_type == 'simple_learning':
        return SimpleSolver(config, args)

    elif config.env.learning_type == 'transfer_learning':
        return TransferSolver(config, args)

    elif config.env.learning_type == 'multitask_learning':
        return MultitaskSolver(config, args)

    elif config.env.learning_type == 'meta_learning':
        raise NotImplementedError()


    else:
        raise ValueError(f'Wrong learning type: {config.env.learning_type}')
