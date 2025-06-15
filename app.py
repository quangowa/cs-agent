import os
import gradio as gr

# Import necessary LlamaIndex components
from llama_index.indices.managed.llama_cloud import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)
from llama_index.core import Settings
from llama_index.llms.nebius import NebiusLLM
from llama_cloud.types import CompositeRetrievalMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)  # Import for type hinting agent response
from llama_index.core.schema import (
    NodeWithScore,
)  # Import for type hinting source_nodes

# Phoenix/OpenInference imports
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# --- Configuration ---
# Replace with your actual LlamaCloud Project Name
LLAMA_CLOUD_PROJECT_NAME = "CustomerSupportProject"

# Configure NebiusLLM
# Ensure NEBIUS_API_KEY is set in your environment variables
Settings.llm = NebiusLLM(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct", 
    temperature=0
)
print(f"[INFO] Configured LLM: {Settings.llm.model}")

# Configure LlamaTrace (Arize Phoenix)
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")

if PHOENIX_PROJECT_NAME and PHOENIX_API_KEY:
    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint="https://app.phoenix.arize.com/v1/traces",
        auto_instrument=True,
    )
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    print("[INFO] LlamaIndex tracing configured for LlamaTrace (Arize Phoenix).")
else:
    print(
        "[INFO] PHOENIX_PROJECT_NAME or PHOENIX_API_KEY not set. LlamaTrace (Arize Phoenix) not configured."
    )

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
    company_intro_slides_index = LlamaCloudIndex(
        name="CompanyIntroductionSlides",
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
    rerank_top_n=2,  # Rerank and return top 2 results from all queried indices
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
composite_retriever.add_index(
    company_intro_slides_index,
    description="Contains presentations that provide an overview of the company, its mission, leadership, and key information for new employees, partners, or investors.",
)
print("[INFO] Sub-indices added.")

# --- Create CondensePlusContextChatEngine ---
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
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
def initial_submit(message: str, history: list):
    """
    Handles the immediate UI update after user submits a message.
    Adds user message to history and shows a loading state for retriever info.
    """
    # Append user message in the 'messages' format
    history.append({"role": "user", "content": message})
    # Return updated history, clear input box, show loading for retriever info, and the original message
    # outputs=[chatbot, msg, retriever_output, user_message_state]
    return history, "", "Retrieving relevant information...", message

def get_agent_response_and_retriever_info(message_from_state: str, history: list):
    """
    Processes the LLM response and extracts retriever information.
    This function is called AFTER initial_submit, so history already contains user's message.
    """
    retriever_output_text = "Error: Could not retrieve information."

    try:
        # Call the chat engine to get the response
        response: AgentChatResponse = chat_engine.chat(message_from_state)

        # AgentChatResponse.response holds the actual LLM generated text: `response=str(response)`
        # Append the assistant's response in the 'messages' format
        history.append({"role": "assistant", "content": response.response})

        # Prepare the retriever information for the new textbox
        check_retriever_text = []

        # Safely attempt to get condensed_question
        condensed_question = "Condensed question not explicitly exposed by chat engine."
        # `chat` method returns `sources=[context_source]` within `AgentChatResponse`
        if hasattr(response, "sources") and response.sources is not None:
            context_source = response.sources[0]
            if (
                hasattr(context_source, "raw_input")
                and context_source.raw_input is not None
                and "message" in context_source.raw_input
            ):
                condensed_question = context_source.raw_input["message"]

        check_retriever_text.append(f"Condensed question: {condensed_question}")
        check_retriever_text.append("==============================")

        # Safely get source_nodes. Ensure it's iterable.
        nodes: list[NodeWithScore] = (
            response.source_nodes
            if hasattr(response, "source_nodes") and response.source_nodes is not None
            else []
        )

        if nodes:
            for i, node in enumerate(nodes):
                # Safely access node metadata and attributes
                metadata = (
                    node.metadata
                    if hasattr(node, "metadata") and node.metadata is not None
                    else {}
                )
                score = (
                    node.score
                    if hasattr(node, "score") and node.score is not None
                    else "N/A"
                )

                file_name = metadata.get("file_name", "N/A")
                page_info = ""
                # Add page number for .pptx files
                if file_name.lower().endswith(".pptx"):
                    page_label = metadata.get("page_label")
                    if page_label:
                        page_info = f" p.{page_label}"

                node_block = f"""\
[Node {i + 1}]
Index: {metadata.get("retriever_pipeline_name", "N/A")}
File: {file_name}{page_info}
Score: {score}
=============================="""
                check_retriever_text.append(node_block)
        else:
            check_retriever_text.append("No source nodes found for this query.")

        retriever_output_text = "\n".join(check_retriever_text)

        # Return updated history and the retriever text
        return history, retriever_output_text

    except Exception as e:
        # Log the full error for debugging
        import traceback

        print(f"Error in get_agent_response_and_retriever_info: {e}")
        traceback.print_exc()

        # Append a generic error message from the assistant
        history.append(
            {
                "role": "assistant",
                "content": "I'm sorry, I encountered an error while processing your request. Please try again.",
            }
        )

        # Only return the detailed error in the retriever info box
        retriever_output_text = f"Error generating retriever info: {e}"
        return history, retriever_output_text

# Markdown text for the application and chatbot welcoming message
description_text = """
Hello! I'm your Smart Customer Support Triage Agent. I can answer questions about our product manuals, FAQs, and billing policies. Ask me anything!

Explore the documents in `./data` directory for sample knowledge base 📑
"""

# Markdown text for `./data` folder structure
knowledge_base_md = """
### 📁 Sample Knowledge Base ([click to explore!](https://huggingface.co/spaces/karenwky/cs-agent/tree/main/data))
```
./data/
├── ProductManuals/
│   ├── product_manuals_metadata.csv
│   ├── product_manuals.pdf
│   ├── task_automation_setup.pdf
│   └── collaboration_tools_overview.pdf
├── FAQGeneralInfo/
│   ├── faqs_general_metadata.csv
│   ├── faqs_general.pdf
│   ├── remote_work_best_practices_faq.pdf
│   └── sustainability_initiatives_info.pdf
├── BillingPolicy/
│   ├── billing_policies_metadata.csv
│   ├── billing_policies.pdf
│   ├── multi_user_discount_guide.pdf
│   ├── late_payment_policy.pdf
│   └── late_payment_policy_v2.pdf
└── CompanyIntroductionSlides/
    ├── company_introduction_slides_metadata.csv
    └── TechSolve_Introduction.pptx
```
"""

# Create a Gradio `Blocks` layout to structure the application
print("[INFO] Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    # Custom CSS
    demo.css = """
    .monospace-font textarea {
        font-family: monospace; /* monospace font for better readability of structured text */
    }
    .center-title { /* centering the title */
        text-align: center;
    }
    """

    # `State` component to hold the user's message between chained function calls
    user_message_state = gr.State(value="")

    # Retriever info begin message
    retriever_info_begin_msg = "Retriever information will appear here after each query."

    # Center-aligned title
    gr.Markdown("# 💬 Smart Customer Support Triage Agent", elem_classes="center-title")
    gr.Markdown(description_text)

    with gr.Row():
        with gr.Column(scale=2):  # Main chat area
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_copy_button=True,
                resizable=True,
                avatar_images=(None, "./logo.png"),
                type="messages",
                value=[
                    {"role": "assistant", "content": description_text}
                ],  # Welcoming message
            )
            msg = gr.Textbox(  # User input
                placeholder="Type your message here...", lines=1, container=False
            )

            with gr.Row():
                send_button = gr.Button("Send")
                clear_button = gr.ClearButton([msg, chatbot])

        with gr.Column(scale=1):  # Retriever info area
            retriever_output = gr.Textbox(
                label="Agentic Retrieval & Smart Routing",
                interactive=False,
                lines=28,
                show_copy_button=True,
                autoscroll=False,
                elem_classes="monospace-font",
                value=retriever_info_begin_msg,
            )

    # New row for Examples and Sample Knowledge Base Tree Diagram
    with gr.Row():
        with gr.Column(scale=1):  # Examples column (left)
            gr.Markdown("### 🗣️ Example Questions")
            # Store the Examples component in a variable to access its `load_input_event`
            examples_component = gr.Examples(
                examples_per_page=8,
                examples=[
                    ["Help! No response from the app, I can't do anything. What should I do? Who can I contact?"],
                    ["I got an $200 invoice outstanding for 45 days. How much is the late charge?"],
                    ["Who is the author of the product manual and when is the last modified date?"],
                    ["Who are the founders of this company? What are their backgrounds?"],
                    ["Is your company environmentally friendly?"],
                    ["What are the procedures to set up task automation?"],
                    ["If I sign up for an annual 'Pro' subscription today and receive the 10% discount, but then decide to cancel after 20 days because the software isn't working for me, what exact amount would I be refunded, considering the 14-day refund policy for annual plans?"],
                    ["I have a question specifically about the 'Sustainable Software Design' aspect mentioned in your sustainability initiatives, which email address should I use for support, sustainability@techsolve.com or support@techsolve.com, and what kind of technical detail can I expect in a response?"],                  
                    ["How can your software help team collaboration?"],
                    ["What is your latest late payment policy?"],
                    ["If I enable auto-pay to avoid late payments, but my payment method on file expires, will TechSolve send me a notification before the payment fails and potentially incurs late fees?"], 
                    ["In shared workspaces, when multiple users are co-editing a document, how does the system handle concurrent edits to the exact same line of text by different users, and what mechanism is in place to prevent data loss or conflicts?"],
                    ["The refund policy states that annual subscriptions can be refunded within 14 days. Does this '14 days' refer to 14 calendar days or 14 business days from the purchase date?"],
                    ["Your multi-user discount guide states that discounts apply to accounts with 5+ users. If my team currently has 4 users and I add a 5th user mid-billing cycle, will the 10% discount be applied immediately to all 5 users, or only at the next billing cycle, and how would the prorated amount for the new user be calculated?"], 
                ],
                inputs=[msg],  # This tells examples to populate the 'msg' textbox
            )
        with gr.Column(scale=1):  # Knowledge Base column (right)
            gr.Markdown(knowledge_base_md)

    # Define the interaction for sending messages (via Enter key in textbox)
    submit_event = msg.submit(  # Step 1: Immediate UI update (updated history, clear input box, show loading for retriever info, and the original message)
        fn=initial_submit,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, retriever_output, user_message_state],
        queue=False,
    ).then(  # Step 2: Call LLM and update agent response and detailed retriever info
        fn=get_agent_response_and_retriever_info,
        inputs=[user_message_state, chatbot],
        outputs=[chatbot, retriever_output],
        queue=True,  # Allow queuing for potentially long LLM calls
    )

    # Define the interaction for send button click
    send_button.click(
        fn=initial_submit,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, retriever_output, user_message_state],
        queue=False,
    ).then(
        fn=get_agent_response_and_retriever_info,
        inputs=[user_message_state, chatbot],
        outputs=[chatbot, retriever_output],
        queue=True,
    )

    # Define the interaction for example questions click
    examples_component.load_input_event.then(
        fn=initial_submit,
        inputs=[msg, chatbot],  # 'msg' will have been populated by the example
        outputs=[chatbot, msg, retriever_output, user_message_state],
        queue=False,
    ).then(
        fn=get_agent_response_and_retriever_info,
        inputs=[user_message_state, chatbot],
        outputs=[chatbot, retriever_output],
        queue=True,
    )

    # Define the interaction for clearing all outputs
    clear_button.click(
        fn=lambda: (
            [],
            "",
            retriever_info_begin_msg,
            "",
        ),
        inputs=[],
        outputs=[chatbot, msg, retriever_output, user_message_state],
        queue=False,
    )

    # DeepLinkButton for sharing current conversation
    gr.DeepLinkButton()

    # Privacy notice at the bottom
    gr.Markdown(
        "_\*By using this chat, you agree that conversations may be recorded for improvement and evaluation. DO NOT disclose any privacy information in the conversation._"
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(show_error=True)