import argparse
import pytorch_lightning as pl
from pl_module import InteractiveCLIP
from pl_data_module import CIRDataModule

def parse_args():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--savedir', type=str, default='')
    # parser.add_argument('--inspect', action="store_true")

    parser.add_argument('--dataset', type=str, default='css3d')
    parser.add_argument('--dataset_path', type=str, default='')

    # parser.add_argument('--learning_rate', type=float, default=1e-2)
    # parser.add_argument('--learning_rate_decay', type=float, default=0.1)
    # parser.add_argument('--learning_rate_decay_frequency', type=int, default=9999999)
    # parser.add_argument('--lr_decay_only_once', action="store_true")
    # more flexible learning rate scheduling. both args must be set or we default to old scheme
    # parser.add_argument('--scheduled_lr_rates', type=str, default="",
    #     help="Separate rates by commas." +
    #     "The learning_rate argument sets the initial rate; " +
    #     "this param sets rates after each scheduled_lr_iters entry" +
    #     "If empty string, old regular decay schedule is used.")
    # parser.add_argument('--scheduled_lr_iters', type=str, default="",
    #     help="Separate iteration numbers by commas." +
    #          "If empty string, old regular decay schedule is used.")  

    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--num_iters', type=int, default=210000)
    # parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('-t', "--test_only", action="store_true")
    # parser.add_argument('-l', '--load', type=str, default="")

    # parser.add_argument('--dropout_rate', type=float, default=0.1)

    # parser.add_argument(
    #     '--drop_worst_flag', action='store_true',
    #     help='If added the model will ingore the highest --drop_worst_rate losses')
    # parser.add_argument('--drop_worst_rate', type=float, default=0.2)

    # parser.add_argument(
    #     '--freeze_img_model', action='store_true',
    #     help='If added the loaded image model weights will not be finetuned')
    # parser.add_argument('--pretrained_weight_lr_factor_image', type=float, default=0.1)
    # parser.add_argument('--pretrained_weight_lr_factor_text', type=float, default=1.)
    # parser.add_argument('--image_model_arch', type=str, default='resnet50')
    # parser.add_argument('--image_model_path', type=str, default='')
    # parser.add_argument(
    #     '--not_pretrained', action='store_true',
    #     help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')

    # parser.add_argument(
    #     '--image_with_unk_string', action='store_true',
    #     help='If added, images wihout modifying captions (i.e., target images) are fed through text+image composer with copies of <UNK>')

    # parser.add_argument(
    #     '--freeze_text_model', action='store_true',
    #     help='If added the loaded text model weights will not be finetuned')
    # parser.add_argument('--text_model_arch', type=str, default='lstm')

    # parser.add_argument('--whole_model_path', type=str, default='')
    # parser.add_argument('--pretraining_dataset', type=str, default='')
    # parser.add_argument('--pretraining_dataset_path', type=str, default='')

    # parser.add_argument('--text_model_layers', type=int, default=1)
    # parser.add_argument('--threshold_rare_words', type=int, default=0)

    # parser.add_argument('--number_attention_blocks', type=int, default=1)
    # parser.add_argument('--width_per_attention_block', type=int, default=256)
    # parser.add_argument('--number_attention_heads', type=int, default=8)
    # parser.add_argument('--att_layer_spec', type=str, default="3_4")
    # parser.add_argument('--attn_positional_encoding', default=None)
    # parser.add_argument('--resolutionwise_pool', action='store_true')

    # specific to sequence concat attention composition
    # parser.add_argument('--sequence_concat_include_text', action="store_true",
    #     help="use post-attn text embeddings in pooling to get final composed embedding")
    # parser.add_argument('--sequence_concat_img_through_attn', action="store_true",
    #     help="target image pathway goes through embedding layers")
    # parser.add_argument('--attn_softmax_replacement', type=str, default="none")
    # parser.add_argument('--attn_2stream_mode', type=str, default="xxx_xmm_xff")

    parser.add_argument('--train_on_validation_set', action="store_true")

    # parser.add_argument(
    #     '--save_every', type=int, default=100,
    #     help="keep checkpoints this often in epochs")
    # parser.add_argument(
    #     '--eval_every', type=int, default=3,
    #     help="run eval on val set this often in epochs")
    # parser.add_argument('--final_eval_on_test', action="store_true")

    args = parser.parse_args()
    # if args.load == "":
    #     args.load = None
    # if args.image_model_path in ["", "none", "None"]:
    #     args.image_model_path = None
    # if args.image_model_arch in ["", "none", "None"]:
    #     args.image_model_arch = None
    # if args.whole_model_path == '':
    #     args.whole_model_path = None
    return args


def main():
    args = parse_args()
    model = InteractiveCLIP()
    dm = CIRDataModule(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        processor=model.processor,
        num_workers=args.num_workers
    )
    dm.setup()
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=1,
                         auto_select_gpus=True, 
                         strategy="ddp",
                         auto_scale_batch_size="binsearch",
                         max_epochs = 100,
                         num_sanity_val_steps=-1)
    # trainer.validate(model=model, datamodule=dm)
    trainer.fit(model=model, datamodule=dm)
    # trainer = pl.Trainer()
    # ckpt_path = "/home/lishi/workspace/MAAF/interactive_clip/lightning_logs/version_15/checkpoints/epoch=7-step=1503.ckpt"
    # model = model.load_from_checkpoint(ckpt_path)
    # for category in ["dress", "toptee", "shirt"]:
    #     dls = dm.val_dataloader(category=category)
    #     print("evaluating on {}".format(category))
    #     trainer.validate(model=model, dataloaders=dls)


if __name__ == '__main__':
    main()