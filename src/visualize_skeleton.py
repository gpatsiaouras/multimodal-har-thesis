from datasets import UtdMhadDataset, UtdMhadDatasetConfig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dataset_config = UtdMhadDatasetConfig()
train_dataset = UtdMhadDataset(modality='skeleton', train=True)

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
    ax.set_xlim(-0.35, 0.52)
    ax.set_ylabel('Z')
    ax.set_ylim(2.2, 3.0)
    ax.set_zlabel('Y')
    ax.set_zlim(-1.1, 0.55)
    ax.view_init(20, -50)
    # Print joints as points
    ax.scatter(sample[:, 0, frame], sample[:, 2, frame], sample[:, 1, frame])
    # Print the index of each joint next to it
    for i in range(sample.shape[0]):
        ax.text(
            sample[i, 0, frame],
            sample[i, 2, frame],
            sample[i, 1, frame],
            str(dataset_config.joint_names[i]),
            size='x-small'
        )
    # Print lines connecting the joints
    for bone in dataset_config.bones:
        ax.plot(
            [sample[bone[0], 0, frame], sample[bone[1], 0, frame]],
            [sample[bone[0], 2, frame], sample[bone[1], 2, frame]],
            [sample[bone[0], 1, frame], sample[bone[1], 1, frame]],
        )


joints_anim = animation.FuncAnimation(fig,
                                      func=update_joints,
                                      frames=sample.shape[2],
                                      interval=200)
plt.show()
