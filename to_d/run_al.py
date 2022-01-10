from bert_trainer_utils import *
# from evaluation_utils import *
from arg_config import arguments
import json
import re

from bert_modeling_custom import BertForSequenceClassification_c
from bert_classifier_model import Bert_classifier_adversarial
from swag_modeling import SWAG_adversarial

from active_learning import Active_learner
from data_handler import Data_Handler
from aquisition import aquisition_function,ALL_AQUISITIONS
from al_params import params_active_l_n,params_active_l_b

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)




@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
        
        
    dataset_name_source: Optional[str] = field(
        default=None, metadata={"help": "a name to the training data folder."}
    )
    main_data_dir: Optional[str] = field(
        default=None, metadata={"help": "the main data directory consisting all datasets"}
    )
        
    use_adversarial: bool = field(
        default=False, metadata={"help": "whether to use adversarial training or not."}
    )
        
    adversarial_epsilon: Optional[float] = field(
        default=None,
        metadata={
            "help": "parmeter used for adversarial attack"
        },
    )
        
    lambdaa: Optional[float] = field(
        default=None,
        metadata={
            "help": "parameter that weights ths losses i.e constrastive loss and crossentropy losses"
        },
    )
        
        
    swag_per_start: Optional[float] = field(
        default=None,
        metadata={
            "help": "percentage of training to be done before starting swag"
        },
    )
        
    temperature_contrastive: Optional[float] = field(
        default=None,
        metadata={
            "help": "use for scaling the output of cosine similirity between Z and Z_attacked"
        },
    )
        
    adv_attk_tpe: Optional[str] = field(
        default=None, metadata={"help": "adversarial attack type"}
    )

        
 

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
        
def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "labels" in examples:
        result["labels"] = [(label_to_id[l] if l != -1 else -1) for l in examples["labels"]]
    return result


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if data_args.task_name is not None:
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()
               }
    
def sample_data(train_dataset,n=100):
    
    """
    function to sample datasets
    
    """

    df_data = train_dataset.to_pandas()

    df_data = df_data.sample(n=n,replace=False)

    filter_df_indexes = df_data.index.values
    
    sample_train = train_dataset.select(list(filter_df_indexes))
    
    
    return sample_train


def generate_active_lr_results(trials_strategy=2,
                               n_epochs_train=5,
                               T_aq_uncert=5,
                               n_query_pick=10,
                               bays_adv_approach=False,
                               active_learning_iters=2,
                               initial_train_data_size=10,
                               n_sample_size_pool_inference=250,
                               use_swag=False,
                               
                              ):
    
    average_results = {i:{} for i in ALL_AQUISITIONS}

    for strategey in ALL_AQUISITIONS:

        # matrices to average results
        all_trials_ac_performace = np.zeros((trials_strategy,active_learning_iters)) 
        all_trials_b_ac_performace = np.zeros((trials_strategy,active_learning_iters)) 
        all_trials_f1_performace = np.zeros((trials_strategy,active_learning_iters)) 


        for trial in range(trials_strategy):

            # data handler
            data_handler = Data_Handler(train_dataset,
                                        eval_dataset,
                                        predict_dataset
                                       )

            # epochs used for training
            training_args.num_train_epochs = n_epochs_train

            # active learner
            active_model = Active_learner(
                data_handler=data_handler,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                criterion=criterion,
                aquisiiton_f=aquisition_function,
                num_labels=num_labels,
                bays_aversarial=bays_adv_approach,
                iters=active_learning_iters)

            # run active learning experiment
            active_model.active_train(
                initial_n_train=initial_train_data_size,
                n_sample_infer=n_sample_size_pool_inference,
                T_aq=T_aq_uncert,
                n_query=n_query_pick,
                aquisition_strategy=strategey,
                use_swag=use_swag
            )


            # trial performance
            ac_performace_temp = active_model.test_performance_ac
            b_ac_performace_temp = active_model.test_performance_b_ac
            f1_performance = active_model.test_performance_f1

            # number of train data
            number_of_train_data = active_model.number_of_train_data

            # adding metrics to matrix for averagind
            all_trials_ac_performace[trial,:] = ac_performace_temp
            all_trials_b_ac_performace[trial,:] = b_ac_performace_temp
            all_trials_f1_performace[trial,:] = f1_performance

        # average results
        average_results[strategey]['acc'] = all_trials_ac_performace.mean(axis=0).tolist()
        average_results[strategey]['b_acc'] = all_trials_b_ac_performace.mean(axis=0).tolist()
        average_results[strategey]['f1'] = all_trials_f1_performace.mean(axis=0).tolist()
        average_results[strategey]['n_train'] = number_of_train_data
    
    return average_results



if __name__ == '__main__':
    
    # create output folder if doesnt exist
    output_dir = arguments['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # write a temporary json file
    json_file_to_parse = os.path.join(output_dir,"temp_args.json")
    with open(json_file_to_parse, 'w') as f:
        json.dump(arguments, f)

    # pass arguments to respective argument classes
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=json_file_to_parse)


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")


    # data source
    data_source = os.path.join(data_args.main_data_dir,data_args.dataset_name_source)

    # set data paths for train, dev and test set
    data_args.train_file = os.path.join(data_source,'train.csv')
    data_args.validation_file = os.path.join(data_source,'dev.csv')
    data_args.test_file = os.path.join(data_source,'test.csv')

    # load data
    if data_args.dataset_name is not None:

        print('s')
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:

        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

    # load datasets
    raw_datasets = load_dataset(extension, data_files=data_files)

    # drop unwanted columns
    raw_datasets = raw_datasets.remove_columns(['Unnamed: 0'])


    # change column names
    raw_datasets = raw_datasets.rename_column('label','labels')

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["labels"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("labels")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    # configurations for BERT
    config = BertConfig.from_pretrained(    
        pretrained_model_name_or_path=model_args.model_name_or_path
    )



    if data_args.use_adversarial:
        # BERT model
        model = BertForSequenceClassification_c.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=None
        )

    else:
        # BERT model
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=None
        )

        raise NotImplementedError

    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "labels"]
    custom_choice = True


    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "labels"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:

                if custom_choice:
                    sentence1_key, sentence2_key = non_label_column_names[1], None
                else:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None


    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    elif data_args.task_name is None and not is_regression:

        label_to_id = {v: i for i, v in enumerate(label_list)}


    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    if data_args.use_adversarial:

        clasifier_adversarial = Bert_classifier_adversarial(
            bert_seq_class=model

        )

        # swag adversarial
        SWAG_adversarial_m = SWAG_adversarial(
            BertForSequenceClassification_c,
            no_cov_mat=not True,
            max_num_models=20,
            num_classes=num_labels,
            config=config,
            model_args=model_args

        )

    else:

        raise NotImplementedError


    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    #     desc="Running tokenizer on dataset",
    )



    # datasets and reducing size if needed
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))


    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
        
    # criterion for loss function
    criterion = nn.CrossEntropyLoss()
    
    #ALL_AQUISITIONS = ALL_AQUISITIONS[:2]
    
    
    # normal model
    average_results_n = generate_active_lr_results(
        trials_strategy=params_active_l_n["trials_strategy"],
        n_epochs_train=params_active_l_n["n_epochs_train"],
        T_aq_uncert=params_active_l_n["T_aq_uncert"],
        n_query_pick=params_active_l_n["n_query_pick"],
        bays_adv_approach=params_active_l_n["bays_adv_approach"],
        active_learning_iters=params_active_l_n["active_learning_iters"],
        initial_train_data_size=params_active_l_n["initial_train_data_size"],
        n_sample_size_pool_inference=params_active_l_n["n_sample_size_pool_inference"],
        use_swag=params_active_l_n["use_swag"]
        
    )
    
    
    # bay_adversarial model
    average_results_b_adv = generate_active_lr_results(
        trials_strategy=params_active_l_b["trials_strategy"],
        n_epochs_train=params_active_l_b["n_epochs_train"],
        T_aq_uncert=params_active_l_b["T_aq_uncert"],
        n_query_pick=params_active_l_b["n_query_pick"],
        bays_adv_approach=params_active_l_b["bays_adv_approach"],
        active_learning_iters=params_active_l_b["active_learning_iters"],
        initial_train_data_size=params_active_l_b["initial_train_data_size"],
        n_sample_size_pool_inference=params_active_l_b["n_sample_size_pool_inference"],
        use_swag=params_active_l_b["use_swag"]
    )
    
    
    
    path_save = './results_al/'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # save normal results
    name = params_active_l_n['name']
    with open(os.path.join(path_save,name), 'w') as f:
        json.dump(average_results_n, f)
        
    # save bays_adv results
    name_b = params_active_l_b['name']
    with open(os.path.join(path_save,name_b), 'w') as f:
        json.dump(average_results_b_adv, f)
        
        
        

        


        
