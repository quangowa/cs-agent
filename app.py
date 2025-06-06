import os
import gradio as gr
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from llama_index.indices.managed.llama_cloud import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_cloud.types import CompositeRetrievalMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# --- Configuration ---
# Replace with your actual LlamaCloud Project Name
LLAMA_CLOUD_PROJECT_NAME = "CustomerSupportProject"

# Configure Anthropic LLM
# Ensure ANTHROPIC_API_KEY is set in your environment variables
Settings.llm = Anthropic(model="claude-sonnet-4-0", temperature=0)
print(f"[INFO] Configured LLM: {Settings.llm.model}")

# Configure LlamaTrace (Arize Phoenix)
PHOENIX_PROJECT_NAME = os.environ["PHOENIX_PROJECT_NAME"]
PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint="https://app.phoenix.arize.com/v1/traces",
    auto_instrument=True,
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
print("[INFO] LlamaIndex tracing configured for LlamaTrace (Arize Phoenix).")

# --- Assume LlamaCloud Indices are pre-created ---
# In a real scenario, you would have uploaded your documents to these indices
# via LlamaCloud UI or API. Here, we connect to existing indices.
print("[INFO] Connecting to LlamaCloud Indices...")

try:
    product_manuals_index = LlamaCloudIndex(
        name="ProductManuals",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
    )
    faq_general_info_index = LlamaCloudIndex(
        name="FAQGeneralInfo",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
    )
    billing_policy_index = LlamaCloudIndex(
        name="BillingPolicy",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
    )
    print("[INFO] Successfully connected to LlamaCloud Indices.")

except Exception as e:
    print(
        f"[ERROR] Error connecting to LlamaCloud Indices. Please ensure they exist and API key is correct: {e}"
    )
    print(
        "[INFO] Exiting. Please create your indices on LlamaCloud and set environment variables."
    )
    exit()  # Exit if indices cannot be connected, as the rest of the code depends on them

# --- Create LlamaCloudCompositeRetriever for Agentic Routing ---
print("[INFO] Creating LlamaCloudCompositeRetriever...")
composite_retriever = LlamaCloudCompositeRetriever(
    name="Customer Support Retriever",
    project_name=LLAMA_CLOUD_PROJECT_NAME,
    create_if_not_exists=True,
    mode=CompositeRetrievalMode.ROUTING,  # Enable intelligent routing
    rerank_top_n=5,  # Rerank and return top 5 results from the chosen indices
)

# Add indices to the composite retriever with descriptive descriptions
# These descriptions are crucial for the agent's routing decisions.
print("[INFO] Adding sub-indices to the composite retriever with descriptions...")
composite_retriever.add_index(
    product_manuals_index,
    description="Information source for detailed product features, technical specifications, troubleshooting steps, and usage guides for various products.",
)
composite_retriever.add_index(
    faq_general_info_index,
    description="Contains common questions and answers, general company policies, public announcements, and basic information about services.",
)
composite_retriever.add_index(
    billing_policy_index,
    description="Provides information related to pricing, subscriptions, invoices, payment methods, and refund policies.",
)
print("[INFO] Sub-indices added.")

# --- Create CondensePlusContextChatEngine ---
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=composite_retriever,
    memory=memory,
    system_prompt=(
        """
        You are a Smart Customer Support Triage Agent. 
        Always be polite and friendly. 
        Provide accurate answers from product manuals, FAQs, and billing policies by intelligently routing queries to the most relevant knowledge base.
        Provide accurate, precise, and useful information directly. 
        Never refer to or mention your information sources (e.g., "the manual says", "from the document"). 
        State facts authoritatively.        
        When asked about file-specific details like the author, creation date, or last modification date, retrieve this information from the document's metadata if available in the provided context.
        """
    ),
    verbose=True,
)
print("[INFO] ChatEngine initialized.")

# --- Gradio Chat UI ---
def chat_with_agent(message, history):
    """
    Handles the chat interaction with the agent.
    `history` is a list of [user_message, agent_response] pairs.
    """
    # Gradio history format needs to be converted for LlamaIndex if not using a direct chat engine
    # However, CondensePlusContextChatEngine handles internal history.
    # We just pass the new message to the chat_engine.
    try:
        response = chat_engine.chat(message)
        return str(response)
    except Exception as e:
        return f"An error occurred: {e}"

# Global variables to manage `check_retriever` output
check_retriever_history = []

# Check retrieved top document's index, filename and score
def check_retriever(chat_history):
    global check_retriever_history

    message = chat_history[-1][0]
    nodes = composite_retriever.retrieve(message)
    index_retrieved = nodes[0].metadata["retriever_pipeline_name"]
    file_retrieved = nodes[0].metadata["file_name"]
    score_retrieved = nodes[0].score

    # Check if the last entry in chat_history has a bot response (indicating a completed turn)
    if chat_history and chat_history[-1][1] is not None:
        index_retrieved_text = f"Index: {index_retrieved}"
        file_retrieved_text = f"File: {file_retrieved}"
        score_retrieved_text = f"Score: {score_retrieved}"
        check_retriever_history.append(f"{index_retrieved_text}")
        check_retriever_history.append(f"{file_retrieved_text}")
        check_retriever_history.append(f"{score_retrieved_text}\n==============================")

    return "\n".join(check_retriever_history)


print("[INFO] Launching Gradio interface...")

# Markdown text for chat interface
description = """
Hello! I'm your Smart Customer Support Triage Agent. I can answer questions about our product manuals, FAQs, and billing policies. Ask me anything!

Explore the documents in `./data` directory for sample knowledge base 📑
"""

# Markdown text for `./data` folder structure
knowledge_base_md = """
### 📁 Sample Knowledge Base
```
./data/
├── billing_policies_metadata.csv
├── faqs_general_metadata.csv
├── product_manuals_metadata.csv
├── product_manuals.pdf
├── task_automation_setup.pdf
├── collaboration_tools_overview.pdf
├── faqs_general.pdf
├── remote_work_best_practices_faq.pdf
├── sustainability_initiatives_info.pdf
├── billing_policies.pdf
├── multi_user_discount_guide.pdf
├── late_payment_policy.pdf
└── late_payment_policy_v2.pdf
```
"""

# Create a Gradio Blocks layout to structure the application
with gr.Blocks() as demo:
    # Create the Gradio ChatInterface at the top
    chat_interface = gr.ChatInterface(
        fn=chat_with_agent,
        title="Smart Customer Support Triage Agent",
        description=description,
        examples=[
            "Help! No response from the app, I can't do anything. What should I do? Who can I contact?",
            "Can we request a Zoom training on remote work? Do you have any guides available?",
            "I didn't pay the invoice. Outstanding 23 days. What's the late fee you are charging?",
            "Who is the author of the product manual and when is the last modified date?",
        ],
        cache_examples=False,
    )

    # DeepLinkButton for sharing current conversation
    gr.DeepLinkButton()

    # Privacy notice under the chat interface
    gr.Markdown(
        "_\*By using this chat, you agree that conversations may be recorded for improvement and evaluation. DO NOT disclose any privacy information in the conversation._"
    )

    # Row for the two side-by-side read-only text boxes
    with gr.Row():
        # Left column for the simple markdown text
        with gr.Column(scale=1):
            gr.Markdown(knowledge_base_md)

        # Right column for showing agentic retrieval and smart routing
        with gr.Column(scale=1):
            check_retriever_display = gr.Textbox(
                value="",  # Starts empty
                label="Top Retrieved Document",
                interactive=False,  # Make it read-only
                lines=12,  # Show 12 lines initially
                max_lines=12,  # Allow up to 12 lines before scrolling
                autoscroll=True,  # Automatically scroll to the bottom when new content is added
            )

    # Set up the event handler to update the counter
    # Pass the chatbot component itself as input to `check_retriever`
    chat_interface.chatbot.change(
        fn=check_retriever,
        inputs=[chat_interface.chatbot],  # Pass the chatbot history
        outputs=check_retriever_display,
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(show_error=True)