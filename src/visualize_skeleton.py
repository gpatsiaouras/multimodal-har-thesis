from datasets import UtdMhadDataset, UtdMhadDatasetConfig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from transforms import Resize, SwapJoints, Normalize, Compose

dataset_config = UtdMhadDatasetConfig()

# Parameters (change accordingly)
n_frames = 125
normalize = True
continuous = True
fix_view_point = False

train_dataset = UtdMhadDataset(modality='skeleton', train=True, transform=Compose([
    SwapJoints(),
    Normalize((1, 2)),
]))

# set limits to display properly, or if dataset is normalized use just 0 and 1
x_lim_min = -0.35 if not normalize else 0
x_lim_max = 0.52 if not normalize else 1
y_lim_min = 2.2 if not normalize else 0
y_lim_max = 3.0 if not normalize else 1
z_lim_min = -1.1 if not normalize else 0
z_lim_max = 0.55 if not normalize else 1

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
    ax.scatter(sample[0, :, frame], sample[2, :, frame], sample[1, :, frame])
    # Print the index of each joint next to it
    for i in range(sample.shape[1]):
        ax.text(
            sample[0, i, frame],
            sample[2, i, frame],
            sample[1, i, frame],
            str(dataset_config.joint_names[i]),
            size='x-small'
        )
    # Print lines connecting the joints
    for bone in dataset_config.bones:
        ax.plot(
            [sample[0, bone[0], frame], sample[0, bone[1], frame]],
            [sample[2, bone[0], frame], sample[2, bone[1], frame]],
            [sample[1, bone[0], frame], sample[1, bone[1], frame]],
        )


if continuous:
    joints_anim = animation.FuncAnimation(fig,
                                          func=update_joints,
                                          frames=sample.shape[2],
                                          interval=200)
else:
    update_joints(0)

plt.show()
