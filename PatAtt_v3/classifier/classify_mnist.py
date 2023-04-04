"""========Default import===="""
from __future__ import print_function
import argparse
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
import torch.nn as nn
import torch
import os
import torch.optim as optim
import shutil
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
import wandb
""" ==========END================"""

""" =========Configurable ======="""
#from models.resnet34 import R_34_MNIST as resnet
#from models.resnet18 import R_18_MNIST as resnet
from models.resnet import resnet10
os.environ['WANDB_SILENT']='true'
""" ===========END=========== """



def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #----------Default Optimizer
    parser.add_argument("--resume",
            type=bool,
            default= False,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=bool,
            default = False,
            help="if test, choose True.")
    parser.add_argument("--pin-memory", 
            type=bool,
            default = True)
    parser.add_argument("--weight-decay", 
            type=float,
            default = 5e-4)
    parser.add_argument("--decay-type",
            type=str,
            default='cosine')
    parser.add_argument("--epochs",
            type=int,
            default=13)
    parser.add_argument("--lr",
            type=float,
            default= 1e-1)
    parser.add_argument("--random-seed",
            type=int,
            default = 0)
    parser.add_argument("--warmup-steps",
            type=int,
            default = 100)
    #----------Dataset Loader
    parser.add_argument("--dataset", 
            type=str,
            default = 'mnist')
    parser.add_argument("--num-classes", 
            type=int,
            default = 10)
    parser.add_argument("--img-size", 
            type=int,
            default = 32)
    parser.add_argument("--num-workers", 
            type=int,
            default = 8)
    parser.add_argument("--num-channel", 
            type=int,
            default = 1)
    parser.add_argument("--train-batch-size", 
            type=int,
            default = 32)
    parser.add_argument("--test-batch-size", 
            type=int,
            default = 32)
    #------------configurations
    parser.add_argument("--ckpt-fpath",
            type=str,
            default='../experiments/classifier/mnist',
            help="save path.")
    parser.add_argument("--interval-val",
            type=int,
            default = 1)
    #--------------wandb
    parser.add_argument("--wandb-project",
            type=str,
            default='classifier_MNIST')
    parser.add_argument("--wandb-id",
            type=str,
            default= 'jonggyujang0123')
    parser.add_argument("--wandb-name",
            type=str,
            default= 'ResNet18_lr')
    parser.add_argument("--wandb-active",
            type=bool,
            default=False)

    args = parser.parse_args()
    return args

def get_data_loader(args):
    if args.dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader
    if args.dataset in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader
    return get_loader(args)





def load_ckpt(checkpoint_fpath, is_best =False):
    """
    Latest checkpoint loader
        checkpoint_fpath : 
    :return: dict
        checkpoint{
            model,
            optimizer,
            epoch,
            scheduler}
    example :
    """
    if is_best:
        ckpt_path = checkpoint_fpath+'/'+'best.pt'
    else:
        ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    try:
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path)
    except:
        print(f"No checkpoint exists from '{ckpt_path}'. Skipping...")
        print("**First time to train**")
    return checkpoint


def save_ckpt(checkpoint_fpath, checkpoint, is_best=False):
    """
    Checkpoint saver
    :checkpoint_fpath : directory of the saved file
    :checkpoint : checkpoiint directory
    :return:
    """
    ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    # Save the state
    if not os.path.exists(checkpoint_fpath):
        os.makedirs(checkpoint_fpath)
    torch.save(checkpoint, ckpt_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
#    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
#        .format(ckpt_path, checkpoint['epoch']))
    if is_best:
        ckpt_path_best = checkpoint_fpath+'/'+'best.pt'
        print("This is the best model\n")
        shutil.copyfile(ckpt_path,
                        ckpt_path_best)




def train(wandb, args, model):
    ## Setting 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    train_loader, test_loader, _ = get_data_loader(args=args)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_loader))
    val_acc = .0; best_acc = .0; start_epoch =0
    AvgLoss = AverageMeter()
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(args.ckpt_fpath)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1

        

    # Prepare dataset and dataloader
    for epoch in range(start_epoch, args.epochs):
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        model.train()
        tepoch = tqdm(train_loader)
        for batch, data in enumerate(tepoch):
            optimizer.zero_grad()
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = model(inputs)
            loss =criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            AvgLoss.update(loss.item(), n =len(data))
            tepoch.set_description(f'Epoch {epoch}: train_loss: {AvgLoss.avg:2.2f}, lr : {scheduler.get_lr()[0]:.2E}')

        # save and evaluate model routinely.
        if epoch % args.interval_val == 0:
            val_acc = evaluate(args, model=model, device=args.device, val_loader=test_loader)
            ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                    }
            save_ckpt(checkpoint_fpath=args.ckpt_fpath, checkpoint=ckpt)
            if best_acc < val_acc:
                save_ckpt(checkpoint_fpath=args.ckpt_fpath, checkpoint=ckpt, is_best=True)
                best_acc = val_acc
            if args.wandb_active:
                wandb.log({
                    "loss": AvgLoss.val,
                    "val_acc" : val_acc
                    })
            print("-"*75+ "\n")
            print(f"| {epoch}-th epoch, training loss is {AvgLoss.avg}, and Val Accuracy is {val_acc*100:2.2f}%\n")
            print("-"*75+ "\n")
        AvgLoss.reset()
        tepoch.close()


def evaluate(args, model, val_loader, device):
    model.eval()
    acc = AverageMeter()
    valepoch = tqdm(val_loader, 
                  unit= " batch")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valepoch):
            inputs = inputs.to(device, non_blocking=True)
            # compute the output
            output = model(inputs).detach().cpu()

            # Measure accuracy
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
            valepoch.set_description(f'Validation Acc: {acc.avg:2.2f}')
    valepoch.close()
    result  = torch.tensor([acc.sum,acc.count]).to(device)

    return result[0].cpu().item()/result[1].cpu().item()

def test(args, model):
    _, _, test_loader = get_data_loader(args= args)
    print("Loading checkpoints ... ")
    model.eval()
    acc = AverageMeter()
    #example_images = []
    correct_list = []
    with torch.no_grad():
        tepoch = tqdm(test_loader, 
                     unit= " batch", 
                     )
        for batch, (inputs, labels) in enumerate(tepoch):
            inputs = inputs.to(args.device, non_blocking=True)
            # compute the output
            output = model(inputs).cpu().detach()
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
            #example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].detach().item(), target[0])
        #wandb.log({"Examples":example_images})
    result  = torch.tensor([acc.sum,acc.count]).to(cfg.device)

    print("-"*75+ "\n")
    print(f"| Testset accuracy is {result[0].cpu().item()/result[1].cpu().item()} = {result[0].cpu().item()}/{result[1].cpu().item()}\n")
    print("-"*75+ "\n")



def main():
    args = parse_args()

    if args.test==False and args.wandb_active:
        wandb.init(project = args.wandb_project,
                   entity = args.wandb_id,
                   config = dict(args),
                   name = f'{args.wandb_name}_lr:{args.lr}',
                   group = args.dataset
                   )
    print(args)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = args.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    args.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = resnet10(num_classes=args.num_classes, num_channel= args.num_channel)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(args.device)
    if args.resume or args.test:
        ckpt = load_ckpt(args.ckpt_fpath, is_best = args.test)
        model.load_state_dict(ckpt['model'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter is {pytorch_total_params:.2E}')

    if args.test:
        test(args, model)
    else:
        train(wandb, args, model)


if __name__ == '__main__':
    main()
