import argparse

def para_config():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # learning type configuration
    parser.add_argument("--multigpu",
            type=bool, 
            help="Local rank. Necessary for using the torch.distributed.launch utility.",
            default= False)
    parser.add_argument("--resume",
            type=int,
            default= 0,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=int,
            default = 0,
            help="if test, choose True.")

    # hyperparameter setting
    parser.add_argument("--epochs",
            type=int,
            default=30)
    parser.add_argument("--train-batch-size",
            type=int,
            default= 64)
    parser.add_argument("--test-batch-size",
            type=int,
            default=100)
    parser.add_argument("--random-seed",
            type=int,
            default=0)
    parser.add_argument("--eval-every",
            type=int,
            default=1)
    parser.add_argument("--pin-memory",
            type=bool,
            default=True)

    # hyperparameter for generator and discrminator
    parser.add_argument("--patch-size",
            type=int,
            default=8)
    parser.add_argument("--patch-target",
            type=int,
            default=4)
    parser.add_argument("--patch-rand",
            type=bool, 
            default= False)
    parser.add_argument("--patch-num",
            type=int, 
            default= 3)
    parser.add_argument("--n-gf",
            type=int,
            default=128,
            help="number of generating feature base")
    parser.add_argument("--n-df",
            type=int,
            default=128,
            help="number of discriminator feature base")
    parser.add_argument("--latent-size",
            type=int,
            default=100)
    parser.add_argument("--w-attack",
            type=float,
            default=0.3)
    parser.add_argument("--w-recon",
            type=float,
            default=0.0)
    parser.add_argument("--attack-labelsmooth",
            type=float,
            default=0.0)
    parser.add_argument("--gan-labelsmooth",
            type=float,
            default=0.0)
    parser.add_argument("--p-flip",
            type=float,
            default=0.0)

    # dataset 
    parser.add_argument("--dataset",
            type=str,
            default="emnist",
            help = "choose one of mnist, emnist, fashion")
    parser.add_argument("--n-classes",
            type=int,
            default=10)
    parser.add_argument("--img-size",
            type=int,
            default=32)
    parser.add_argument("--num-workers",
            type=int,
            default=8)
    parser.add_argument("--num-channel",
            type=int,
            default=1)
    
    # save path configuration
    parser.add_argument("--ckpt-fpath",
            type=str,
            default="../experiments/attacker/emnist",
            help="path for saving the attacker model")
    parser.add_argument("--ckpt-fpath-class",
            type=str,
            default="../experiments/classifier/mnist",
            help="path for restoring the classifier")

    # WANDB SETTINGS
    parser.add_argument("--wandb-project",
            type=str,
            default="PVMI-MNIST")
    parser.add_argument("--wandb-id",
            type=str,
            default="jonggyujang0123")
    parser.add_argument("--wandb-name",
            type=str,
            default="ResNet50")
    parser.add_argument("--wandb-active",
            type=bool,
            default=True)





    # optimizer setting 
    parser.add_argument("--weight-decay",
            type=float,
            default = 5.0e-4)
    parser.add_argument("--beta-1",
            type=float,
            default = 0.5)
    parser.add_argument("--beta-2",
            type=float,
            default = 0.999)
    parser.add_argument("--decay-type",
            type=str,
            default="linear",
            help="choose linear or cosine")
    parser.add_argument("--warmup-steps",
            type=int,
            default=100)
    parser.add_argument("--lr",
            type=float,
            default=1e-4,
            help = "learning rate")
    

    args = parser.parse_args()
    return args
