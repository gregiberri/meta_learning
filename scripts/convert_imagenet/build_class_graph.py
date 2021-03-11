# set current workdir to project dir
import csv
import glob
import json
import os
import sys
import numpy as np
import tqdm

current_working_dir = os.getcwd()
new_working_dir = '/' + os.path.join(*current_working_dir.split('/')[:-2])
os.chdir(new_working_dir)
sys.path.insert(0, new_working_dir)

np.random.seed(0)


class ClassGraphBuilder:
    def __init__(self):
        with open('scripts/convert_imagenet/dataset_specs/ilsvrc_2012_dataset_spec.json', 'r') as f:
            dataset_spec = json.load(f)

        self.train_nodes = dataset_spec['split_subgraphs']['TRAIN']
        self.val_nodes = dataset_spec['split_subgraphs']['VALID']

        self.goal_dir = 'dataset_files/imagenet/'
        if not os.path.exists(self.goal_dir):
            os.makedirs(self.goal_dir)

    def get_children(self, node, all_nodes):
        nodelist = []
        for node_id in node['children_ids']:
            nodelist.append(self.get_from_id(node_id, all_nodes))
        return nodelist

    def get_children_number(self, node, all_nodes):
        return len(self.get_children(node, all_nodes))

    def get_nodes_with_children_number(self, node_group, all_nodes, children_number):
        nodes_with_end_node_number = []
        print(f'Get nodes with children_number={children_number}')
        with tqdm.tqdm(total=len(node_group)) as pbar_memory_load:
            for node in node_group:
                if self.get_children_number(node, all_nodes) == children_number:
                    nodes_with_end_node_number.append(node)
                pbar_memory_load.update(1)

        return nodes_with_end_node_number

    def get_nodes_with_end_node_number(self, node_group, all_nodes, end_node_number):
        nodes_with_end_node_number = []
        print(f'Get nodes with end_node_number={end_node_number}')
        with tqdm.tqdm(total=len(node_group)) as pbar_memory_load:
            for node in node_group:
                if self.get_end_node_number(node, all_nodes) == end_node_number:
                    nodes_with_end_node_number.append(node)
                pbar_memory_load.update(1)

        return nodes_with_end_node_number

    def get_end_node_number(self, node, all_nodes):
        children_list = []
        children = self.get_children(node, all_nodes)

        if not children:
            return 1
        for node_id in children:
            children_list.append(self.get_end_node_number(node_id, all_nodes))

        return sum(children_list)

    def get_depth(self, node, all_nodes):
        depths = []
        children = self.get_children(node, all_nodes)

        if not children:
            return 0
        for child in children:
            child_depth = self.get_depth(child, all_nodes)
            depths.append(child_depth + 1)

        return max(depths)

    def get_nodes_with_depth(self, node_group, all_nodes, depth):
        nodes_with_depth = []
        print(f'Get nodes with depth={depth}')
        with tqdm.tqdm(total=len(node_group)) as pbar_memory_load:
            for node in node_group:
                if self.get_depth(node, all_nodes) == depth:
                    nodes_with_depth.append(node)
                pbar_memory_load.update(1)

        return nodes_with_depth

    def get_from_id(self, node_id, all_nodes):
        for current_node in all_nodes:
            if node_id == current_node['wn_id']:
                return current_node
        raise ValueError(f'Node with id: {node_id} not found.')

    def get_from_word(self, node_word, all_nodes):
        for current_node in all_nodes:
            current_words = current_node['words'].split(', ')
            current_words = [current_word.lower() for current_word in current_words]
            if node_word in current_words:
                return current_node
        raise ValueError(f'Node with word: {node_word} not found.')

    def generate_train_orig_goal_class(self, train_goal_class_number):
        print('Generating training original-goal pairs.')

        end_nodes = self.get_nodes_with_children_number(self.train_nodes, self.train_nodes, 0)

        nodes_w_depth = self.get_nodes_with_depth(self.train_nodes, self.train_nodes, 1)
        nodes_w_endnn = self.get_nodes_with_end_node_number(nodes_w_depth, self.train_nodes, 2)

        class_number_to_mix = (len(end_nodes) - train_goal_class_number)
        classes_to_mix = list(np.random.choice(nodes_w_endnn, class_number_to_mix, replace=False))

        classes_children_to_mix = []
        for class_to_mix in classes_to_mix:
            classes_children_to_mix.extend(self.get_children(class_to_mix, self.train_nodes))

        orig_goal_list = []
        for end_node in end_nodes:
            try:
                self.get_from_id(end_node['wn_id'], classes_children_to_mix)
                orig_goal_list.append([end_node['wn_id'], end_node['parents_ids'][0]])
            except ValueError:
                orig_goal_list.append([end_node['wn_id'], end_node['wn_id']])

        # check whether we have the desired number of classes
        assert len(np.unique([orig_goal[1] for orig_goal in orig_goal_list])) == train_goal_class_number

        with open(os.path.join(self.goal_dir, 'train_orig_goal_class_pairs.txt'), 'w') as f:
            for row in orig_goal_list:
                s = " ".join(map(str, row))
                f.write(s + '\n')

    def generate_val_orig_goal_class(self):
        print('Generating validation original-goal pairs.')

        end_nodes = self.get_nodes_with_children_number(self.val_nodes, self.val_nodes, 0)

        with open('scripts/convert_imagenet/dataset_specs/val_classes.csv', 'r') as f:
            reader = csv.reader(f)
            goal_words = [word[0].strip('\ufeff') for word in list(reader)]

        goal_elements = [self.get_from_word(word, self.val_nodes) for word in goal_words]

        goal_element_children_to_mix = []
        for goal_element in goal_elements:
            goal_element_children_to_mix.extend(self.get_children(goal_element, self.val_nodes))

        orig_goal_list = []
        for end_node in end_nodes:
            try:
                self.get_from_id(end_node['wn_id'], goal_element_children_to_mix)
                orig_goal_list.append([end_node['wn_id'], end_node['parents_ids'][0]])
            except ValueError:
                orig_goal_list.append([end_node['wn_id'], end_node['wn_id']])

        # check whether we have the desired number of classes
        assert len(np.unique([orig_goal[1] for orig_goal in orig_goal_list])) == 100

        with open(os.path.join(self.goal_dir, 'val_orig_goal_class_pairs.txt'), 'w') as f:
            for row in orig_goal_list:
                s = " ".join(map(str, row))
                f.write(s + '\n')


if __name__ == '__main__':
    # orig_path = '../data/imagenet'
    train_aim_number = 700

    class_graph_builder = ClassGraphBuilder()
    class_graph_builder.generate_train_orig_goal_class(train_aim_number)
    class_graph_builder.generate_val_orig_goal_class()
