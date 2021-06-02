# Inspired by https://github.com/facebookresearch/Adversarial-Continual-Learning

import os,argparse,time
import numpy as np
from omegaconf import OmegaConf

import torch

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils

from model_wrapper import Base, Bert, Word2Vec

import pickle
from models import alexnet

from torchvision import datasets

import json

tstart=time.time()

cifar100_class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# # Load the config file
# parser.add_argument('--config',  type=str, default='./configs/config_cifar100_experimental.yml')
# flags =  parser.parse_args()
# args = OmegaConf.load(flags.config)

########################################################################################################################

from dataloaders import cifar100 as datagenerator

########################################################################################################################

def run(args, run_id):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loader
    print('Instantiate data generators and model...')
    args.data_dir = init_root()
    dataloader = datagenerator.DatasetGen(args)
    taskcla= dataloader.taskcla
     

    checkpoint = f"{args.data_dir}/continual/{args.data}_{args.model}"
    if args.expt_name:
        checkpoint+= f"_{args.expt_name}"
    final_results_path = checkpoint + "/"
    checkpoint+= f"/run_{run_id}/"

    print(f"Making directories: {checkpoint}")
    make_dirs(checkpoint)
    utils.save_code(checkpoint)

    kwargs = {
        "checkpoints_path":checkpoint,
        "epochs":args.nepochs,
        "pick_best_model":True
    }

    # Model
    if args.emb :
        print("Using BERT embeddings")
        net = alexnet.AlexNet(768)
        kwargs["class_names"] = cifar100_class_names
        wrapper=Bert
    else:
        print("Using Base Model")
        net = alexnet.AlexNet(dataloader.num_classes)
        wrapper = Base

    if args.emb == "word2vec":
        print("Using word2vec Model")
        net = alexnet.AlexNet(300)
        kwargs["class_names"] = cifar100_class_names
        wrapper=Word2Vec

    kwargs["model"] = net
    appr=wrapper(**kwargs)
    
    appr.train_learning_type = args.train_learning_type
    appr.test_learning_type = args.test_learning_type

    # Loop tasks
    acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

    labels_per_task = []

    for t,ncla in taskcla:
        print('*'*250)
        dataset = dataloader.get(t)
        assert len(set(dataset[t]["train_labels"])) == 100/args.ntasks
        labels_per_task.append(list(set(dataset[t]["train_labels"])))
        print()
        print('Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
        print('*'*250)

        # Train
        appr.task_labels= dataset[t]["task_labels"]
        appr.train_model(t,dataset, deep_test=True)
        print('-'*250)
        print()

        # Test
        for u in range(t+1):
            appr.task_labels= dataset[u]["task_labels"]
            test_res = appr.test_model(u, dataset[u]['test'])
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, dataset[u]['name'],
                                                                                          test_res['loss'],
                                                                                          test_res['acc']))

            acc[t, u] = test_res['acc']
            lss[t, u] = test_res['loss']


        # Save
        print()
        print('Saved accuracies at '+os.path.join(checkpoint,args.output))
        make_dirs(os.path.join(checkpoint,args.output))
        np.savetxt(os.path.join(checkpoint,args.output),acc,'%.6f')

    avg_acc, gem_bwt = print_log_acc_bwt(taskcla, acc, lss, output_path=checkpoint, run_id=run_id, labels_per_task=labels_per_task)

    return avg_acc, gem_bwt, final_results_path


def print_log_acc_bwt(taskcla, acc, lss, output_path, run_id, labels_per_task):

    print('*'*100)
    print('Accuracies =')
    for i in range(1, acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['avg_acc'] = str(avg_acc)
    logs['gem_bwt'] = str(gem_bwt)
    logs['ucb_bwt'] = str(ucb_bwt)
    logs['taskcla'] = str(taskcla)
    logs['acc'] = str(acc)
    logs['loss'] = str(lss.tolist())
    logs['rii'] = str(np.diag(acc))
    logs['rij'] = str(acc[-1])
    logs['labels_per_task'] = str(labels_per_task)

    # pickle


    path = os.path.join(output_path, f'logs_run_id_{run_id}.json')
    with open(path, 'w') as output:
        json.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt



#######################################################################################################################


def main(args):


    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)


    accuracies, forgetting = [], []
    for n in range(args.num_runs):
        args.seed = n
        args.output = f'{args.data}_{args.model}_{args.ntasks}_tasks_seed_{args.seed}.txt'
        print ("args.output: ", args.output)

        print (" >>>> Run #", n)
        acc, bwt, ch = run(args, n)
        accuracies.append(acc)
        forgetting.append(bwt)


    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))

    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))

    logs = {
        "runs": str(args.num_runs),
        "avg_acc": str([np.array(accuracies).mean(), np.array(accuracies).std()]),
        "avg_bwt": str([np.array(forgetting).mean(), np.array(forgetting).std()]),
        "time": str((time.time()-tstart)/(60))
    }
    path = os.path.join(ch, f'Final_results.json')
    with open(path, 'w') as output:
        json.dump(logs, output)

    utils.print_time()

def init_root():
    if os.path.exists("/nethome/bdevnani3/raid"):
        return "/nethome/bdevnani3/raid"
    else:
        return "."

def make_dirs(path: str):
    """ Why is this not how the standard library works? """
    path = os.path.split(path)[0]
    if path != "":
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

#######################################################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
    # Load the config file
    parser.add_argument('--config',  type=str, default='./configs/config_cifar100_experimental.yml')
    flags =  parser.parse_args()
    args = OmegaConf.load(flags.config)
    # for conf in ["cifar100_base_icl.yml", "cifar100_base_itl.yml"]:
    #     print("###############", conf, "###############")
    #     parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
    #     parser.add_argument('--config',  type=str, default=f'./configs/{conf}')
    #     flags =  parser.parse_args()
    #     args = OmegaConf.load(flags.config)
    main(args)