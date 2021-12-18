
import dataclasses
from dataclasses import dataclass, field
import os
import enums


@dataclass
class RuntimeArguments:
    """Arguments for runtime."""

    only_test: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to test only'}
    )

    trained_vocab: str = field(
        # default='../pre_trained/vocabs/',
        default=None,
        metadata={'help': 'Directory of trained vocabs'}
    )

    trained_model: str = field(
        # default='../pre_trained/models/all/',
        default=None,
        metadata={'help': 'Directory of trained model'}
    )

    random_seed: int = field(
        default=42,
        metadata={'help': 'Specific random seed manually for all operations, 0 to disable'}
    )

    n_epoch: int = field(
        default=200,
        metadata={'help': 'Number of data iterations for training'}
    )

    batch_size: int = field(
        default=4,
        metadata={'help': 'Batch size for training on each device'}
    )

    eval_batch_size: int = field(
        default=8,
        metadata={'help': 'Batch size for evaluation on each device'}
    )

    beam_width: int = field(
        default=5,
        metadata={'help': 'Beam width when using beam decoding, 1 to greedy decode'}
    )

    logging_steps: int = field(
        default=100,
        metadata={'help': 'Log training state every n steps'}
    )

    cuda_visible_devices: str = field(
        default=None,
        metadata={'help': 'Visible cuda devices, string formatted, device number divided by \',\', e.g., \'0, 2\', '
                          '\'None\' will use all'}
    )

    fp16: bool = field(
        default=True,
        metadata={'action': 'store_true',
                  'help': 'Whether to use mixed precision'}
    )


@dataclass
class DatasetArguments:
    """Arguments for dataset loading."""

    dataset_root: str = field(
        default='../../dataset/',
        metadata={'help': 'Root of the dataset'}
    )

    train_subset_ratio: float = field(
        default=None,
        metadata={'help': 'Ratio of train subset'}
    )


@dataclass
class SavingArguments:
    """Arguments for saving and loading."""

    model_name: str = field(
        default='default_model',
        metadata={'help': 'Name of the model'}
    )

    dataset_save_dir: str = field(
        default=os.path.join(DatasetArguments.dataset_root, 'dataset_saved'),
        metadata={'help': 'Directory to save and load dataset pickle instance'}
    )

    vocab_save_dir: str = field(
        default=os.path.join(DatasetArguments.dataset_root, 'vocab_saved'),
        metadata={'help': 'Directory to save and load vocab pickle instance'}
    )


@dataclass
class PreprocessingArguments:
    """Arguments for data preprocessing."""

    code_vocab_size: int = field(
        default=50000,
        metadata={'help': 'Maximum size of uni-vocab'}
    )

    max_code_len: int = field(
        default=256,
        metadata={'help': 'Maximum length of code sequence'}
    )

    max_path_len: int = field(
        default=100,
        metadata={'help': 'Maximum length of path sequence'}
    )

    max_id_len: int = field(
        default=10,
        metadata={'help': 'Maximum length of id sequence'}
    )

    max_node_len: int = field(
        default=30,
        metadata={'help': 'Maximum length of node sequence'}
    )

    max_nl_len: int = field(
        default=64,
        metadata={'help': 'Maximum length of the nl sequence'}
    )

    tokenize_method: str = field(
        default='bpe',
        metadata={'help': 'Tokenize method',
                  'choices': ['word', 'bpe']}
    )


@dataclass
class ModelArguments:
    """Arguments for model related hyper-parameters"""

    d_model: int = field(
        default=512,
        metadata={'help': 'Dimension of the model'}
    )

    d_ff: int = field(
        default=2048,
        metadata={'help': 'Dimension of the feed forward layer'}
    )

    n_head: int = field(
        default=8,
        metadata={'help': 'Number of head of self attention'}
    )

    n_layer: int = field(
        default=6,
        metadata={'help': 'Number of layer'}
    )

    dropout: float = field(
        default=0.1,
        metadata={'help': 'Dropout probability'}
    )


@dataclass
class OptimizerArguments:
    """Arguments for optimizer, early stopping, warmup, grad clipping, label smoothing."""

    learning_rate: float = field(
        default=5e-5,
        metadata={'help': 'Learning rate'}
    )

    lr_decay_rate: float = field(
        default=0,
        metadata={'help': 'Decay ratio for learning rate, 0 to disable'}
    )

    early_stop_patience: int = field(
        default=20,
        metadata={'help': 'Stop training if performance does not improve in n epoch, 0 to disable'}
    )

    warmup_steps: int = field(
        default=1000,
        metadata={'help': 'Warmup steps for optimizer, 0 to disable'}
    )

    grad_clipping_norm: float = field(
        default=1.0,
        metadata={'help': 'Gradient clipping norm, 0 to disable'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Gradient accumulation steps, default to 1'}
    )

    label_smoothing: float = field(
        default=0.1,
        metadata={'help': 'Label smoothing ratio, 0 to disable'}
    )


@dataclass
class TaskArguments:
    """Arguments for specific tasks"""

    summarization_language: str = field(
        default='java',
        metadata={'help': 'Language of the source code in code summarization, also the directory of the dataset dir'}
    )


def transfer_arg_name(name):
    return '--' + name.replace('_', '-')


def add_args(parser):
    """Add all arguments to the given parser."""
    for data_container in [RuntimeArguments, DatasetArguments, SavingArguments,
                           PreprocessingArguments, ModelArguments, OptimizerArguments, TaskArguments]:
        group = parser.add_argument_group(data_container.__name__)
        for data_field in dataclasses.fields(data_container):
            if 'action' in data_field.metadata:
                group.add_argument(transfer_arg_name(data_field.name),
                                   default=data_field.default,
                                   **data_field.metadata)
            else:
                group.add_argument(transfer_arg_name(data_field.name),
                                   type=data_field.type,
                                   default=data_field.default,
                                   **data_field.metadata)
