import torch
import matplotlib.pyplot as plt

member_properties = torch.load("../res/Saves/test/member_test_properties.pt")
plt.plot(member_properties["training_loss"])
plt.show()
plt.plot(member_properties["eval_loss"])
plt.show()