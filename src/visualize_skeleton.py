from datasets import UtdMhadDataset, UtdMhadDatasetConfig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from transforms import Normalize, RandomEulerRotation, Compose

dataset_config = UtdMhadDatasetConfig()

# Parameters (change accordingly)
n_frames = 125
continuous = True
fix_view_point = False
normalize = True

utdMhadConfig = UtdMhadDatasetConfig()
mean = utdMhadConfig.modalities['skeleton']['mean']
std = utdMhadConfig.modalities['skeleton']['std']

train_dataset = UtdMhadDataset(modality='skeleton', train=True, transform=Compose([
    Normalize('skeleton', mean, std),
    RandomEulerRotation(-5, 5, 5),
]))

# set limits to display properly, or if dataset is normalized use just 0 and 1
x_lim_min = -0.35 if not normalize else -3
x_lim_max = 0.52 if not normalize else 3
y_lim_min = 2.2 if not normalize else -3
y_lim_max = 3.0 if not normalize else 3
z_lim_min = -1.1 if not normalize else -3
z_lim_max = 0.55 if not normalize else 3

# Choose a sample from the dataset. 0 sample is first action which swipe right
sample, _ = train_dataset[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

bone_list = [
    [0, 1],
    [1, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [1, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [1, 2],
    [2, 3],
    [3, 16],
    [16, 17],
    [17, 18],
    [18, 19],
    [3, 12],
    [12, 13],
    [13, 14],
    [14, 15],
]


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

    if fix_view_point:
        ax.view_init(20, -50)

    # Print joints as points
    ax.scatter(sample[frame, :, 0], sample[frame, :, 2], sample[frame, :, 1])
    # Print the index of each joint next to it
    for i in range(sample.shape[1]):
        ax.text(
            sample[frame, i, 0],
            sample[frame, i, 2],
            sample[frame, i, 1],
            str(dataset_config.joint_names[i]),
            size='x-small'
        )
    # Print lines connecting the joints
    for bone in dataset_config.bones:
        ax.plot(
            [sample[frame, bone[0], 0], sample[frame, bone[1], 0]],
            [sample[frame, bone[0], 2], sample[frame, bone[1], 2]],
            [sample[frame, bone[0], 1], sample[frame, bone[1], 1]],
        )


if continuous:
    joints_anim = animation.FuncAnimation(fig,
                                          func=update_joints,
                                          frames=sample.shape[0],
                                          interval=200)
else:
    update_joints(0)

plt.show()
