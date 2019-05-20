#!/usr/bin/env python
import pickle
import argparse
import numpy as np
import os

WORK_DIR = "/home/adrian/Proyecto"
POSES_DIR = WORK_DIR+"/poses"
OBJ_DIR = WORK_DIR+"/working_objects"


def create_branch_dict(branches_file):
    f = open(branches_file)
    with open(branches_file, "rb") as f:
        branch_dict = {}
        index = 1
        for branch_type in f:
            branch_dict[branch_type.strip()] = index
            index += 1
    return branch_dict


def encode_branches(branches, branch_dict):
    encoded_branches = [branch_dict[br] for br in branches]
    return encoded_branches


def encode_distances(distances):
    bins = [(2 + n*(4.0/19)) for n in range(20)]
    dist_encoded = np.digitize(distances, bins)
    return dist_encoded


def create_pose_matrix(branch_dict, branches_file):
    m = max(branch_dict.values())+1
    with open(branches_file, "rb") as f:

        pose = []
        rmsd, score = [float(x) for x in f.readline().split()]
        branches = []
        branch = f.readline().strip()
        branches.append((0.0, branch))
        for line in f:
            if len(line.split(",")) <= 1:
                if len(branches) == 0:
                    empty_signal_br = [m for i in range(5)]
                    empty_signal_ds = encode_distances([6.0, 6.0, 6.0,
                                                        6.0, 6.0])
                    pose.append(np.array([empty_signal_br, empty_signal_ds]))

                elif len(branches) >= 5:
                    sorted_branches = sorted(branches, key=lambda b: b[0])
                    close_distances, close_branches = zip(*sorted_branches[:5])
                    encoded_branches = encode_branches(close_branches, branch_dict)
                    encoded_distances = encode_distances(close_distances)
                    pose.append(np.array([encoded_branches, encoded_distances]))
                branch = line.strip()
                branches = [(0.0, branch)]
            else:
                dist, branch = map(lambda x: x.strip(), line.split(","))
                branches.append((float(dist), branch))

        if len(branches) >= 5:
            sorted_branches = sorted(branches, key=lambda b: b[0])
            close_distances, close_branches = zip(*sorted_branches[:5])
            encoded_branches = encode_branches(close_branches, branch_dict)
            encoded_distances = encode_distances(close_distances)
            pose.append(np.array([encoded_branches, encoded_distances]))
    return pose


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def create_dataset(branch_dict, filename):
    print "Processing {}...".format(filename)
    logfile = open('not_done.txt', 'w')
    with open(filename, "rb") as f:
        total = file_len(filename)
        current = 1

        dataset = []
        for line in f:
            current_file, pose, rmsdi, score = line.split()
            if current % 1000 == 0:
                print "{}_{} -> {}/{}".format(current_file, pose,
                                              current, total)
            rmsdi = float(rmsdi)

            if rmsdi <= 3:
                i = 1
            else:
                i = 0

            label = i

            pose = "{}/{}/{}_{}".format(POSES_DIR, current_file.split(".")[0],
                                        pose, current_file)

            if os.path.isfile(pose) and float(rmsdi) >= 0:
                pose_matrix = create_pose_matrix(branch_dict, pose)
                if pose_matrix:
                    dataset.append((pose_matrix, label))
                else:
                    logfile.write("Problem with pose matrix: %4s %16s\n"%(label,pose.split('/')[-1]))
            else:
                logfile.write("File not found: %4s %16s rmsd: %6s\n"%(label,current_file,rmsdi))
            current += 1

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Branch dictionary")
    parser.add_argument("-p", help="Poses file")
    args = parser.parse_args()

    dataset_txt = args.p
    basename = dataset_txt.split('/')[-1].split('.')[0]

    branch_dict = create_branch_dict(args.d)
    dataset = create_dataset(branch_dict, dataset_txt)

    print("{} poses generated".format(len(dataset)))
    with open(OBJ_DIR + '/' + basename + ".pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
