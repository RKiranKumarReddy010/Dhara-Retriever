import torch
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
