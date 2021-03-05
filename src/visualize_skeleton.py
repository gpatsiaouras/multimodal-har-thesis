import argparse
from datasets import UtdMhadDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def main(args):
    train_dataset = UtdMhadDataset(modality='skeleton')

    # Choose a sample from the dataset. 0 sample is first action which swipe right
    sample, _ = train_dataset[5]

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.view_init(21, -51)

    # Create cubic bounding box to simulate equal aspect ratio
    X = sample[0, :, 0]
    Y = sample[0, :, 2]
    Z = sample[0, :, 1]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # set limits to display properly, or if dataset is normalized use just 0 and 1
    x_lim_min = mid_x - max_range
    x_lim_max = mid_x + max_range
    y_lim_min = mid_y - max_range
    y_lim_max = mid_y + max_range
    z_lim_min = mid_z - max_range
    z_lim_max = mid_z + max_range

    def update_joints(frame):
        # Reset axes
        ax.clear()
        # Re-set limits to keep axis from moving
        ax.set_xlabel('X')
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylabel('Z')
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_zlabel('Y')
        ax.set_zlim(z_lim_min, z_lim_max)

        # Print joints as points
        ax.scatter(sample[frame, :, 0], sample[frame, :, 2], sample[frame, :, 1])
        # Print the index of each joint next to it
        if args.show_joints:
            for i in range(sample.shape[1]):
                ax.text(
                    sample[frame, i, 0],
                    sample[frame, i, 2],
                    sample[frame, i, 1],
                    str(train_dataset.joint_names[i]),
                    size='x-small'
                )
        # Print lines connecting the joints
        for bone in train_dataset.bones:
            ax.plot(
                [sample[frame, bone[0], 0], sample[frame, bone[1], 0]],
                [sample[frame, bone[0], 2], sample[frame, bone[1], 2]],
                [sample[frame, bone[0], 1], sample[frame, bone[1], 1]],
            )

    if args.continuous or args.frame is None:
        joints_anim = animation.FuncAnimation(fig,
                                              func=update_joints,
                                              frames=sample.shape[0],
                                              interval=200)
    else:
        update_joints(args.frame)

    if args.save and not args.continuous:
        plt.savefig('skeleton_%s' % args.frame)

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=None, help='Frame to plot')
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--show_joints', action='store_true')
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.set_defaults(save=False, continuous=False, normalize=False, show_joints=False, show=True)
    args = parser.parse_args()
    main(args)
