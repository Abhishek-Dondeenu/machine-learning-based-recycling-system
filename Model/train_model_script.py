from object_detection import model_lib_v2
from object_detection.utils import config_util

pipeline_config_path = 'models/trainingModel/pipeline.config'

model_path = 'models/trainingModel'

configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

model_config = configs['model']

train_config = configs['train_config']

num_train_steps = train_config.num_steps

model_lib_v2.train_loop(
    pipeline_config_path=pipeline_config_path,
    model_dir=model_path,
    train_steps=num_train_steps,
    use_tpu=False,
    checkpoint_every_n=1000
)
