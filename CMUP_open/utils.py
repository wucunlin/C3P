import os
import torch
import torch.optim as optim
import json
import time
import logging

def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)
    inputs1_diff = points[:, 0:3, :].unsqueeze(1).expand(cur_train_size, sample_num_level2, 3, sample_num_level1) \
                   - points[:, 0:3, 0:sample_num_level2].transpose(1, 2).unsqueeze(-1).expand(cur_train_size,
                                                                                              sample_num_level2, 3,
                                                                                              sample_num_level1)  # B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)  # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)  # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False,
                                    sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64

    # ball query
    invalid_map = dists.gt(ball_radius)  # B * 128 * 64, invalid_map.float().sum()
    # pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:, jj, :][invalid_map.data[:, jj, :]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size, 1, sample_num_level2 * knn_K).expand(cur_train_size,
                                                                                              points.size(1),
                                                                                              sample_num_level2 * knn_K)
    inputs_level2 = points.gather(2, idx_group_l1_long).view(cur_train_size, points.size(1), sample_num_level2,
                                                             knn_K)  # B*131*128*64

    inputs_level2_center = points[:, 0:3, 0:sample_num_level2].unsqueeze(3)  # B*3*128*1
    inputs_level2[:, 0:3, :, :] = inputs_level2[:, 0:3, :, :] - inputs_level2_center.expand(cur_train_size, 3,
                                                                                            sample_num_level2,
                                                                                            knn_K)  # B*3*128*64
    return inputs_level2, inputs_level2_center, inputs1_idx

def group_points_fast(points, sample_num, knn_K, inputs1_idx):
    # group points using knn and ball query
    # points: B*(3+C1)*N1, inputs1_idx: B * N2 * knn_K
    # return inputs_level2: B*(3+C1)*N2*knn_K, inputs_level2_center: B*3*N2*1
    cur_train_size = points.size(0)
    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,-1).expand(-1,points.size(1),-1) # B*(3+C1)*(N2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num,knn_K) # B*(3+C1)*N2*knn_K

    inputs_level2_center = points[:,0:3,0:sample_num].unsqueeze(-1)       # B*3*N2*1
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num,knn_K) # B*3*N2*knn_K
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*(3+C1)*N2*knn_K, inputs_level2_center: B*3*N2*1

def KNN(k, unknown_points, known_points):
    '''
    :param k:
    :param unknown_points: B * 3 * n
    :param known_points: B * 3 * m
    :return: index, B * 3 * n
    '''
    points_diff = unknown_points.unsqueeze(2).expand(unknown_points.shape[0], unknown_points.shape[1],
                                                     known_points.shape[2], unknown_points.shape[2]) \
                  - known_points.unsqueeze(-1).expand(known_points.shape[0], known_points.shape[1],
                                                      known_points.shape[2], unknown_points.shape[2])
    dist = torch.sum(points_diff * points_diff, dim=1)
    k_dist, idx = torch.topk(dist, k, largest=False, dim=1)

    k_dist_recip = 1.0 / (k_dist + 1e-8)
    norm = torch.sum(k_dist_recip, dim=1, keepdim=True)
    weight = k_dist_recip / norm

    return idx, weight


def KNN_interpolate(unknown_coords, known_coords, known_feats, k=3):
    '''
    :param unknown_coords: B * 3 * n
    :param known_coords: B * 3 * m
    :param known_feats: B * C * m
    :return:
    '''
    batch_size = known_feats.shape[0]
    feats_num = known_feats.shape[1]
    known_num = known_feats.shape[2]
    unknown_num = unknown_coords.shape[2]

    idx, weight = KNN(k, unknown_coords, known_coords)
    # idx: B * k * n  weight: B * k * n
    # print(idx, '\n', weight)

    idx = idx.unsqueeze(1).expand(batch_size, feats_num, k, unknown_num)    # B * C * k * n
    known_feats = known_feats.unsqueeze(-1).expand(batch_size, feats_num, known_num, unknown_num)   # B * C * m * n

    selected_feats = known_feats.gather(2, idx)     # B * C * k * n
    inter_feats = torch.sum(weight.unsqueeze(1) * selected_feats, dim=2)    # (B * 1 * k * n) * (B * C * k * n)

    return inter_feats      # B * C * n


def load_cache(file_name):
    ''' Opens the cache file if it exists and loads the JSON into
    the CACHE_DICT dictionary.
    if the cache file doesn't exist, creates a new cache dictionary

    Parameters
    ----------
    file_name: str
        the name of cache file

    Returns
    -------
    The opened cache: dict
    '''
    cache_file = open(file_name, 'r')
    cache_contents = cache_file.read()
    cache_dict = json.loads(cache_contents)
    cache_file.close()

    return cache_dict

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def create_logger(cfg, phase='train'):
    output_dir = cfg.OUTPUT_DIR

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger