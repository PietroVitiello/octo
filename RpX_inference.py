from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
import jax


from octo.model.octo_model import OctoModel

# XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

class RpX_Octo():

    def __init__(self,
                 run_name: str = "octo_1") -> None:
        self.model = OctoModel.load_pretrained(f"octo_models/{run_name}")
        self.stats = self.model.dataset_statistics["action"]


        # print(self.model.example_batch)

    def predict(self,
                rgb: np.ndarray,
                language_description: str) -> np.ndarray:
        
        assert rgb.shape == (480, 640, 3), "Input image shape should be (480, 640, 3)"
        assert isinstance(language_description, str), "Language description should be a string"

        rgb = cv2.resize(
            rgb, (256, 256), 
            interpolation = cv2.INTER_LINEAR
        )
        rgb = rgb[None, None ,...]
        observation = {
            "image_primary": rgb,
            # "pad_mask": np.array([[True]]),
            "timestep": np.array([[0]]),
            "timestep_pad_mask": np.array([[True]])
        }
        task = self.model.create_tasks(texts=[language_description])
        action = self.model.sample_actions(
            observation, task, rng=jax.random.PRNGKey(0),
            unnormalization_statistics=self.stats)
        action = np.reshape(action[0, 0], (40, 4, 3))
        return action
    

if __name__ == "__main__":
    model = RpX_Octo("octo_try")
    
    episode_path = "/home/pita/Documents/Projects/octo/data/episode_0.npy"
    data = np.load(episode_path, allow_pickle=True).item()   # this is a list of dicts in our case
    # actions = np.zeros((40, 4, 3), dtype=np.float32)
    # actions[:len(data['actions'])] = data['actions']
    # data['actions'] = actions.reshape(-1, 12) # to undo actions.reshape((40, 12))
    rgb = data['image']
    lang_desc = data['language_instruction']
    actions = np.zeros((40, 4, 3), dtype=np.float32)
    actions[:len(data['actions'])] = data['actions']

    pred_actions = model.predict(rgb, lang_desc)
    print(pred_actions, "\n\n")
    print(f"Predicted actions shape: {pred_actions.shape}")
    print(f"Error: {np.sqrt(np.mean((actions - pred_actions)**2))}")
