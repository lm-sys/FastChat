from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
def get_text_generation_pipeline(model_name, revision):
    model_path = '/cpfs/29cd2992fe666f2a/shared/public/baichuan_model'
    text_generation = pipeline(task=Tasks.text_generation,
                               model=model_name,
                               device_map='auto',
                               model_revision=revision,model_dir=model_path)
    return text_generation
# 1. baichuan-7b-base
pipeline_instance_7b_base = get_text_generation_pipeline('baichuan-inc/baichuan-7B', 'v1.0.5')

# 2. baichuan-13b-base
pipeline_instance_13b_base = get_text_generation_pipeline('baichuan-inc/Baichuan-13B-Base', 'v1.0.1')

# 3. baichuan-13b-chat
pipeline_instance_13b_chat = get_text_generation_pipeline('baichuan-inc/Baichuan-13B-Chat', 'v1.0.4')

# 4. baichuan-2-7b-base
pipeline_instance_2_7b_base = get_text_generation_pipeline('baichuan-inc/Baichuan-2-7B-Base', 'v1.0.1')

# 5. baichuan-2-7b-chat
pipeline_instance_2_7b_chat = get_text_generation_pipeline('baichuan-inc/Baichuan-2-7B-Chat', 'v1.0.1')

# 6. baichuan-2-13b-base
pipeline_instance_2_13b_base = get_text_generation_pipeline('baichuan-inc/Baichuan2-13B-Base', 'v1.0.1')

# 7. baichuan-2-13b-chat
pipeline_instance_2_13b_chat = get_text_generation_pipeline('baichuan-inc/Baichuan2-13B-Chat', 'v1.0.2')