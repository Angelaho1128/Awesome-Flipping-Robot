"""Baseten Training Jobs infrastructure. Importing does not submit a job."""
from truss_train import (TrainingProject,TrainingJob,Image,Compute,Runtime,
                         CacheConfig,CheckpointingConfig,LoadCheckpointConfig,BasetenCheckpoint)
from truss.base.truss_config import AcceleratorSpec


def project(mode="smoke",resume_job_id=None,resume_checkpoint=None):
    # Process settings already live in run.sh. Avoid platform-owned environment
    # overrides in the submission; python -u provides unbuffered training logs.
    env={}
    runtime_args={}
    if resume_job_id:
        if not resume_checkpoint:raise ValueError("Specify the exact synced checkpoint name to resume")
        env["PANCAKE_RESUME_DIR"]="/tmp/pancake_loaded_checkpoint"
        runtime_args["load_checkpoint_config"]=LoadCheckpointConfig(
            enabled=True,download_folder=env["PANCAKE_RESUME_DIR"],checkpoints=[
                BasetenCheckpoint.from_named_checkpoint(checkpoint_name=resume_checkpoint,job_id=resume_job_id)])
    runtime=Runtime(start_commands=[f"bash run.sh {mode}"],environment_variables=env,
                    cache_config=CacheConfig(enabled=True),
                    checkpointing_config=CheckpointingConfig(enabled=True),**runtime_args)
    return TrainingProject(name="awesome-pancake-240-100-warp",job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
        compute=Compute(cpu_count=16,memory="32Gi",accelerator=AcceleratorSpec(accelerator="H100",count=1)),
        runtime=runtime))
