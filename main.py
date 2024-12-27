import gradio as gr
import ollama


USER_LABEL = "User"
ASSISTANT_LABEL = "Assistant"
WELCOME_MSG = "Hello, ask me anything!  :)"
TEXTBOX_MSG = "Type a messsage ..."

MODEL_ID = "llama3.2:1b"

INPUT_EXAMPLES = [
    ["What is the capital of Thailand?"],
    ["What is the capital of Peru?"],
    ["What is the capital of Antartica?"],
]


def list_models_on_device() -> list:

    try:
        models = [
            models_dict["name"] for models_dict in ollama.list()["models"]
        ]
        return models
    
    except Exception as e:
        raise ConnectionError(
            f"{str(e)}: please ensure that ollama is running, use "
            f"`ollama serve`."
        )
    

MODELS_ON_DEVICE = list_models_on_device()


def user_prompt_is_invalid(user_prompt: str) -> bool:
    """Check if """
    return user_prompt == ""

    
def respond(user_prompt, model_id, history):
    """Generate a response to a user prompt. The conversation history is used as context.

    Args:
        user_prompt: user input message.
        history: conversation history to be used as context in the response generation.
    """

    if user_prompt_is_invalid(user_prompt):
        raise gr.Error(
            "Empty user prompt detected, please write something.", 
            duration=5,
        )

    history.append((user_prompt, ""))
    streaming_history = history.copy()

    full_prompt = ""
    for user, assistant in history:
        full_prompt += f"User: {user}\nAssistant: {assistant}\n"
    full_prompt += f"User: {user_prompt}\nAssistant:"

    try:
        stream = ollama.chat(
            model = model_id,
            messages = [{'role': 'user', 'content': full_prompt}],
            stream = True,
        )

        partial_response = ""
        for chunk in stream:
            partial_response += chunk['message']['content']
            streaming_history[-1] = (user_prompt, partial_response)
            yield streaming_history

        history[-1] = (user_prompt, partial_response)
        return history

    except Exception as e:
        if e.status_code == 404:
            try:
                gr.Info(
                    f"Attempting to download model '{MODEL_ID}'...",
                    duration=10
                )
                ollama.pull(MODEL_ID)
            except ollama.ResponseError as t:
                raise gr.Error(
                    f"{str(t)}: {MODEL_ID} not a valid Ollama model id, "
                    f"see https://ollama.com/search.", 
                    duration=10
                )
                yield history
        else:
            raise gr.Error(
                f"{str(e)}: status code {e.status_code}", 
                duration=10
            )
            yield history


with gr.Blocks(
    theme = gr.themes.Glass(
        font = [gr.themes.GoogleFont("IBM Plex Sans")]
    )
) as app:
        
    chatbot = gr.Chatbot(placeholder=WELCOME_MSG, label=ASSISTANT_LABEL)

    with gr.Row():
        user_prompt = gr.Textbox(placeholder=TEXTBOX_MSG, label=USER_LABEL)
        submit_btn = gr.Button("Send")

    with gr.Row():
        model_id = gr.Dropdown(
            choices=MODELS_ON_DEVICE, value=MODEL_ID, label="Model"
        )

    with gr.Row():
        _ = gr.Examples(
            examples = INPUT_EXAMPLES,
            inputs = [user_prompt],
        )

    submit_btn.click(respond, inputs=(user_prompt, model_id, chatbot), outputs=(chatbot))


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=5555)

