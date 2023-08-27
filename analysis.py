from ogb.graphproppred import PygGraphPropPredDataset


if __name__ == '__main__':

    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset/')

    len_dataset = len(dataset)
    print(f'Total {len_dataset} graphs in dataset.')

    print(f'Number of node features: {dataset.num_node_features}.')
    print(f'Number of edge features: {dataset.num_edge_features}.')

    positive_classes = sum([int(g.y[0]) for g in dataset if int(g.y[0]) == 1])
    print(f'\nPositive classes: {positive_classes}')
    print(f'Positive classes percentage: {(positive_classes / len_dataset) * 100}')
    negative_classes = len_dataset - positive_classes
    print(f'Negative classes: {negative_classes}')
    print(f'Negative classes percentage: {(negative_classes / len_dataset) * 100}')

    split_idx = dataset.get_idx_split()
    training = dataset[split_idx['train']]
    len_training = len(training)
    print(f'\nTraining subset size: {len_training} graphs.')
    positive_classes_training = sum([int(g.y[0]) for g in training if int(g.y[0]) == 1])
    print(f'Positive classes in training subset: {positive_classes_training}')
    print(f'Positive classes percentage in training subset: {(positive_classes_training / len_training) * 100}')
    negative_classes_training = len_training - positive_classes_training
    print(f'Negative classes in training subset: {negative_classes_training}')
    print(f'Negative classes percentage in training subset: {(negative_classes_training / len_training) * 100}')

    validation = dataset[split_idx['valid']]
    len_validation = len(validation)
    print(f'\nValidation subset size: {len_validation} graphs.')
    positive_classes_validation = sum([int(g.y[0]) for g in validation if int(g.y[0]) == 1])
    print(f'Positive classes in validation subset: {positive_classes_validation}')
    print(f'Positive classes percentage in validation subset: {(positive_classes_validation / len_validation) * 100}')
    negative_classes_validation = len_validation - positive_classes_validation
    print(f'Negative classes in validation subset: {negative_classes_validation}')
    print(f'Negative classes percentage in validation subset: {(negative_classes_validation / len_validation) * 100}')

    testing = dataset[split_idx['test']]
    len_testing = len(testing)
    print(f'\nTesting subset size: {len_testing} graphs.')
    positive_classes_testing = sum([int(g.y[0]) for g in testing if int(g.y[0]) == 1])
    print(f'Positive classes in testing subset: {positive_classes_testing}')
    print(f'Positive classes percentage in testing subset: {(positive_classes_testing / len_testing) * 100}')
    negative_classes_testing = len_testing - positive_classes_testing
    print(f'Negative classes in testing subset: {negative_classes_testing}')
    print(f'Negative classes percentage in testing subset: {(negative_classes_testing / len_testing) * 100}')

    average_nodes = sum([g.num_nodes for g in dataset]) / len_dataset
    print(f'\nAverage number of nodes per graph: {average_nodes}')
    average_edges = sum([g.num_edges for g in dataset]) / len_dataset
    print(f'Average number of edges per graph: {average_edges}')
    average_degree = average_edges / average_nodes
    print(f'Average node degree: {average_degree}')

    print()
