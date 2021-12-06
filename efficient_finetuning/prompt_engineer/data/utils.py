import clip
import numpy as np
import os
import random
import torch

from tqdm import tqdm

def get_batch_concat_embedding(data_loader, classes, clip_model, device="cuda", phrase="a {}", dataset_name=None):
    """Given a dataloader object, generate concatenated embeddings using clip model.
    For some reason using task sampler keeps giving OOM error, this doesn't. This is valuable during test
    time when batch sizes are rather large.
    """
    if dataset_name is not None:
        f = f"/nethome/bdevnani3/raid/processed_data/prompt_engineer/{dataset_name}.pt"
        if os.path.isfile(f):
            out = torch.load(f)
            return out

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            
            images_emb = clip_model.encode_image(images.to(device))
            images_emb = images_emb/images_emb.norm(dim=-1, keepdim=True)
            
            temp_labels =[]
            for label in labels:
                class_name = classes[label]
                class_name = class_name.replace("_", " ")
                text = clip.tokenize(phrase.format(class_name))
                class_embeddings = clip_model.encode_text(text.cuda())
                class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                temp_labels.append(class_embeddings)
            temp_labels = torch.cat(temp_labels,dim=0)
            
            final_input = torch.cat((images_emb, temp_labels),dim=1)
            all_features.append(final_input)
            all_labels.append(labels)
    out = torch.cat(all_features).float().to(device), torch.cat(all_labels).to(device)
    torch.save(out, f)
    return out

## Large datasets when the entire embedding version can't be loaded in memory 
class LargeDataset():
    """
    To future self who will end up using this for datasets like pets/imagenet
    To speed up potentially consider:
    1) Not calling the DataLoader twice
    2) Invest time into creating a dataset of embeddings that can potentially directly use the class sampler?
    3) Download the entire thing as dataset shards that can be used to generate episodes. (Potentially best option)
    """
    def get_training_episode(self,n_way, n_shot, dataset, classes, clip_model, device="cuda", phrase="a {}"):

        # pick n random classes
        selected_classes = random.sample(range(len(classes)), n_way)

        support_indices = []
        query_indices = []
        for _cls in selected_classes:
            
            # pick all indices in the dataset where selected classes are activated
            _class_indices = set(list(np.argwhere(dataset.targets==_cls).flatten()))

            # generate support and query set
            support_temp = random.sample(list(_class_indices),n_shot)
            query_temp = list(_class_indices - set(support_temp))
            
            support_indices.extend(support_temp)
            query_indices.extend(query_temp)

        support_dataset = torch.utils.data.Subset(dataset, support_indices)
        support_dataloader = torch.utils.data.DataLoader(support_dataset, shuffle=True, num_workers=2, batch_size=10)
        support_inputs, support_targets = get_batch_concat_embedding(support_dataloader, dataset.classes, clip_model, device, phrase)

        query_dataset = torch.utils.data.Subset(dataset, query_indices)
        query_dataloader = torch.utils.data.DataLoader(query_dataset, shuffle=True, num_workers=2, batch_size=25)
        query_inputs, query_targets = get_batch_concat_embedding(query_dataloader, dataset.classes, clip_model, device, phrase)

        original_index_mapping = {}
        if n_way < len(classes):
            for i, _class in enumerate(selected_classes):
                original_index_mapping[i]=_class
                support_targets[support_targets==_class] = i
                query_targets[support_targets==_class] = i

        return support_inputs, support_targets, query_inputs, query_targets, original_index_mapping


    def get_training_epoch(self, n_episodes, n_way, n_shot, dataset, classes, clip_model, device="cuda"):

        full_epoch = []
        for _ in range(n_episodes):
            full_epoch.append(self.get_training_episode(n_way, n_shot, dataset, classes, clip_model, device, self.dataset_obj.phrase))
        return full_epoch

## For embedding datasets that can be entirely loaded in memory
class SmallDataset():

    def __init__(self, dataset_obj, test=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device)
        train_dataset = dataset_obj.get_train_dataset(transform_fn=clip_preprocess)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=100,
            num_workers=4,
            shuffle=True
        )
        dataset_name = dataset_obj.name + "_train"
        self.all_items, self.all_labels = get_batch_concat_embedding(train_loader,  dataset_obj.classes, clip_model, phrase=dataset_obj.phrase, dataset_name=dataset_name)
        self.classes = dataset_obj.classes
        self.test = test

        if test:
            test_dataset = dataset_obj.get_test_dataset(transform_fn=clip_preprocess)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=100,
                num_workers=4,
                shuffle=True
            )
            dataset_name = dataset_obj.name + "_test"
            self.test_items, self.test_labels = get_batch_concat_embedding(test_loader,  dataset_obj.classes, clip_model, phrase=dataset_obj.phrase, dataset_name=dataset_name)

    def get_training_episode(self,n_way, n_shot):

        # pick n random classes
        selected_classes = random.sample(range(len(self.classes)), n_way)
        
        support_indices = []
        query_indices = []
        for _cls in selected_classes:
            
            # pick all indices in the dataset where selected classes are activated
            _class_indices = set(list(np.argwhere(self.all_labels.cpu().numpy()==_cls).flatten()))

            # generate support and query set
            support_temp = random.sample(list(_class_indices),n_shot)
            query_temp = list(_class_indices - set(support_temp))
            
            support_indices.extend(support_temp)
            query_indices.extend(query_temp)

        support_inputs, support_targets = self.all_items[support_indices], self.all_labels[support_indices]
        query_inputs, query_targets = self.all_items[query_indices], self.all_labels[query_indices]

        original_index_mapping = {}
        if n_way < len(self.classes):
            for i, _class in enumerate(selected_classes):
                original_index_mapping[i]=_class
                support_targets[support_targets==_class] = i
                query_targets[query_targets==_class] = i

        return support_inputs, support_targets, query_inputs, query_targets, original_index_mapping


    def get_training_epoch(self, n_episodes, n_way, n_shot):

        full_epoch = []
        for _ in range(n_episodes):
            full_epoch.append(self.get_training_episode(n_way, n_shot))
        return full_epoch
    
    def get_test_episode(self, n_shot, n_way=None, selected_classes=None):
        """
        Main difference here is support set will be generated from the training set and query from test set.
        This can be potentially changed later.
        """

        if n_way == None:
            n_way = len(self.classes)

        assert self.test == True; "Please initialize with test == True"

        if selected_classes == None:
            # pick n random classes
            selected_classes = random.sample(range(len(self.classes)), n_way)
        
        support_indices = []
        query_indices = []
        for _cls in selected_classes:
            
            # pick all indices in the dataset where selected classes are activated
            _class_indices = set(list(np.argwhere(self.all_labels.cpu().numpy()==_cls).flatten()))
            support_temp = random.sample(list(_class_indices),n_shot)            
            support_indices.extend(support_temp)

            _class_indices = set(list(np.argwhere(self.test_labels.cpu().numpy()==_cls).flatten()))
            query_temp = list(_class_indices - set(support_temp))
            query_indices.extend(query_temp)

        support_inputs, support_targets = self.all_items[support_indices], self.all_labels[support_indices]
        query_inputs, query_targets = self.test_items[query_indices], self.test_labels[query_indices]

        original_index_mapping = {}
        if n_way < len(self.classes):
            for i, _class in enumerate(selected_classes):
                original_index_mapping[i]=_class
                support_targets[support_targets==_class] = i
                query_targets[support_targets==_class] = i

        return support_inputs, support_targets, query_inputs, query_targets, original_index_mapping

    def get_test_episode_set(self, n_episodes, n_shot, n_way=None):
        out = []
        for i in range(n_episodes):
            out.append(self.get_test_episode(n_shot, n_way=n_way))
        return out

