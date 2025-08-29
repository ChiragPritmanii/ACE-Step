"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Gradio App with Configurable Options")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to the checkpoint directory. Downloads automatically if empty."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="The server name to use for the Gradio app."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="The port to use for the Gradio app."
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="The CUDA device ID to use."
    )
    parser.add_argument(
        "--share",
        type=bool,
        default=False,
        help="Whether to create a public, shareable link for the Gradio app."
    )

    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Use bfloat16 precision. Turn off if using MPS."
    )

    parser.add_argument(
        "--torch_compile",
        type=bool,
        default=False,
        help="Whether to use torch.compile."
    )

    parser.add_argument(
        "--cpu_offload",
        type=bool,
        default=False,
        help="Whether to use CPU offloading (only load current stage's model to GPU)."
    )

    parser.add_argument(
        "--overlapped_decode",
        type=bool,
        default=False,
        help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)."
    )

    return parser.parse_args()

if __name__ == "__main__":
    from acestep.ui.components import create_main_demo_ui
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.data_sampler import DataSampler

    args = parse_args()

    checkpoint_path = args.checkpoint_path
    server_name = args.server_name
    port = args.port
    device_id = args.device_id
    share = args.share
    bf16 = args.bf16
    torch_compile = args.torch_compile
    cpu_offload = args.cpu_offload
    overlapped_decode = args.overlapped_decode
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
    )
    demo.launch(server_name=server_name, server_port=port, share=share)