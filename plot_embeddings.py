import matplotlib.pyplot as plt
import numpy as np

words = ["AI", "chip", "export", "policy", "cat", "coffee"]

vectors = np.array([
    [0.52, 0.48, 0.50],  # AI
    [0.55, 0.51, 0.47],  # chip
    [0.48, 0.23, 0.52],  # export
    [0.50, 0.36, 0.75],  # policy
    [0.10, 0.10, 0.15],  # cat
    [0.90, 0.85, 0.95],  # coffee
])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
ax.scatter(xs, ys, zs)

# 给每个点设置不同的偏移量，单位很小（0.01 左右），这样文字不重叠
offsets = np.array([
    [0.00,  0.00,  0.03],   # AI
    [0.03,  0.00,  0.00],   # chip
    [-0.03, 0.00,  0.00],   # export
    [0.00, -0.03,  0.00],   # policy
    [0.02,  0.02,  0.02],   # cat
    [-0.04, -0.02, 0.00],   # coffee
])

for (word, x, y, z, (ox, oy, oz)) in zip(words, xs, ys, zs, offsets):
    ax.text(x + ox, y + oy, z + oz, word,
            fontsize=9 if word in ["AI", "chip", "export", "policy"] else 10)

ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.set_title("Example: 3D embedding space (word → vector)")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.tight_layout()
plt.savefig("embedding_3d_example.png", dpi=300)
plt.close()
print("Saved to embedding_3d_example.png")
