import torch
from dqn_agent import DQNAgent

agent = DQNAgent(state_dim=6, action_dim=3)
agent.load_model("right_agent.pth")
agent.model.eval()

dummy_input = torch.zeros(1, 6)

torch.onnx.export(
    agent.model,
    dummy_input,
    "right_agent_single.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    export_params=True,
    dynamo=False       # embed all weights in the file
)

print("Exported right_agent_single.onnx successfully")