from transformers import BartConfig, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    IntervalStrategy, SchedulerType

from transformers import BartForConditionalGeneration

import logging
from typing import Union, Tuple
import os

from data.vocab import Vocab, load_vocab, init_vocab
from data.dataset import init_dataset
from utils.general import count_params, human_format, layer_wise_parameters
from eval.metrics import bleu, meteor, rouge_l, avg_ir_metrics, accuracy_for_sequence
from utils.callbacks import LogStateCallBack
from utils.trainer import CodeTrainer
import enums

logger = logging.getLogger(__name__)


def run_summarization(
        args,
        trained_model: Union[BartForConditionalGeneration, str] = None,
        trained_vocab: Union[Vocab, str] = None,
        only_test=False):
    """
    Fine-tuning from given pre-trained model and vocabs, or training from scratch.
    """
    logger.info('-' * 100)
    logger.info(f'Code summarization on language: {args.summarization_language}')
    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    splits = ['test'] if only_test else ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = init_dataset(args=args,
                                       language=args.summarization_language,
                                       split=split)
        logger.info(f'The size of {split} set: {len(datasets[split])}')
    if args.train_subset_ratio and 'train' in datasets:
        datasets['train'] = datasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(datasets['train'])))

    logger.info('Datasets loaded successfully')

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab:
        if isinstance(trained_vocab, Vocab):
            logger.info('Vocabularies are passed through parameter')
            uni_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            uni_vocab = load_vocab(vocab_root=trained_vocab, name=args.vocab_name)
    else:
        logger.info('Building vocabularies')
        uni_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                               name=args.vocab_name,
                               method=args.tokenize_method,
                               vocab_size=args.vocab_size,
                               datasets=[datasets['train'].codes, datasets['train'].nls],
                               ignore_case=True,
                               save_root=args.vocab_root)
    logger.info(f'The size of uni-vocabulary: {len(uni_vocab)}')
    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForConditionalGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForConditionalGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                 config=config)
    else:
        logger.info('Building the model')
        config = BartConfig(vocab_size=len(uni_vocab),
                            max_position_embeddings=1024,
                            encoder_layers=args.n_layer,
                            encoder_ffn_dim=args.d_ff,
                            encoder_attention_heads=args.n_head,
                            decoder_layers=args.n_layer,
                            decoder_ffn_dim=args.d_ff,
                            decoder_attention_heads=args.n_head,
                            activation_function='gelu',
                            d_model=args.d_model,
                            dropout=args.dropout,
                            use_cache=True,
                            pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                            bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            is_encoder_decoder=True,
                            decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            max_length=args.max_nl_len,
                            min_length=1,
                            num_beams=args.beam_width,
                            num_labels=2)
        model = BartForConditionalGeneration(config)
    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = uni_vocab.decode_batch(preds)
        decoded_labels = uni_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    # compute metrics
    def compute_valid_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        return result

    def compute_test_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        try:
            result.update(meteor(references=refs, candidates=cans))
        except Exception:
            pass
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root),
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.EPOCH,
                                             prediction_loss_only=False,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=args.logging_steps,
                                             save_strategy=IntervalStrategy.EPOCH,
                                             save_total_limit=5,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             metric_for_best_model='bleu',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    trainer = CodeTrainer(main_args=args,
                          uni_vocab=uni_vocab,
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=datasets['train'] if 'train' in datasets else None,
                          eval_dataset=datasets['valid'] if 'valid' in datasets else None,
                          tokenizer=uni_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    logger.info('Running configurations initialized successfully')

    # --------------------------------------------------
    # train
    # --------------------------------------------------
    if not only_test:
        logger.info('-' * 100)
        logger.info('Start training')
        train_result = trainer.train()
        logger.info('Training finished')
        trainer.save_model(args.model_root)
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics(split='train', metrics=metrics)
        trainer.save_metrics(split='train', metrics=metrics)

        # --------------------------------------------------
        # eval
        # --------------------------------------------------
        # logger.info('-' * 100)
        # logger.info('Start evaluating')
        # eval_metrics = trainer.evaluate(metric_key_prefix='valid',
        #                                 max_length=args.max_decode_step,
        #                                 num_beams=args.beam_width)
        # trainer.log_metrics(split='valid', metrics=eval_metrics)
        # trainer.save_metrics(split='valid', metrics=eval_metrics)

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Start testing')
    trainer.compute_metrics = compute_test_metrics
    predict_results = trainer.predict(test_dataset=datasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_nl_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')
