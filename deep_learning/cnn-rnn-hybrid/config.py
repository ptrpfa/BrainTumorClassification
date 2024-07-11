""" Configurations """
# Folders
EXPORT_FOLDER = "../../export"
MAIN_DATASET_FOLDER = "../../dataset"
DATASET_FOLDER = f"{MAIN_DATASET_FOLDER}/dataset_1"
TRAIN_FOLDER = "../../dataset/train_test_dataset/train"
TEST_FOLDER = "../../dataset/train_test_dataset/test"
MODEL_FOLDER = "./models"

# Parameters
TEST_SIZE = 0.3
RANDOM_STATE = 123456

IMAGE_SIZE = 224
BATCH_SIZE = 32

MODEL_FILE = "models/vgg19_model.keras"
INTIAL_MODEL_FILE = "models/initial_vgg19_model.keras"
