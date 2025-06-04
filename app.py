import gradio as gr

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex, LlamaCloudCompositeRetriever
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_cloud.types import CompositeRetrievalMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# --- Configuration ---
# Replace with your actual LlamaCloud Project Name
LLAMA_CLOUD_PROJECT_NAME = "CustomerSupportProject"

# Configure Anthropic LLM (Claude-3 Opus)
# Ensure ANTHROPIC_API_KEY is set in your environment variables
Settings.llm = Anthropic(model="claude-3-haiku-20240307", temperature=0)
print(f"Configured LLM: {Settings.llm.model}")

# --- Assume LlamaCloud Indices are pre-created ---
# In a real scenario, you would have uploaded your documents to these indices
# via LlamaCloud UI or API. Here, we connect to existing indices.
print("Connecting to LlamaCloud Indices...")

try:
    product_manuals_index = LlamaCloudIndex(
        name="ProductManuals",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
        # api_key=os.getenv("LLAMA_CLOUD_API_KEY") # API key can also be passed here
    )
    faq_general_info_index = LlamaCloudIndex(
        name="FAQGeneralInfo",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
    )
    billing_policy_index = LlamaCloudIndex(
        name="BillingPolicy",
        project_name=LLAMA_CLOUD_PROJECT_NAME,
    )
    print("Successfully connected to LlamaCloud Indices.")

except Exception as e:
    print(f"Error connecting to LlamaCloud Indices. Please ensure they exist and API key is correct: {e}")
    print("Exiting. Please create your indices on LlamaCloud and set environment variables.")
    exit() # Exit if indices cannot be connected, as the rest of the code depends on them

# --- Create LlamaCloudCompositeRetriever for Agentic Routing ---
print("Creating LlamaCloudCompositeRetriever...")
composite_retriever = LlamaCloudCompositeRetriever(
    name="Customer Support Retriever",
    project_name=LLAMA_CLOUD_PROJECT_NAME,
    create_if_not_exists=True,
    mode=CompositeRetrievalMode.ROUTING, # Enable intelligent routing
    rerank_top_n=5, # Rerank and return top 5 results from the chosen indices
    # api_key=os.getenv("LLAMA_CLOUD_API_KEY") # API key can also be passed here
)

# Add indices to the composite retriever with descriptive descriptions
# These descriptions are crucial for the agent's routing decisions.
print("Adding sub-indices to the composite retriever with descriptions...")
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
print("Sub-indices added.")

# --- Create ChatMemoryBuffer and CondensePlusContextChatEngine ---
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=composite_retriever, 
    memory=memory, 
    system_prompt=(
        """
        You are a Smart Customer Support Triage Agent. 
        Always be polite and friendly. 
        Provide accurate answers from product manuals, FAQs, and billing policies by intelligently routing queries to the most relevant knowledge base.
        """
    ), 
    verbose=True,
)
print("ChatEngine initialized.")

# --- Gradio Chat UI ---
def chat_with_agent(message, history):
    """
    Handles the chat interaction with the agent.
    `history` is a list of [user_message, agent_response] pairs.
    """
    # Gradio history format needs to be converted for LlamaIndex if not using a direct chat engine
    # However, CondenseQuestionChatEngine handles internal history.
    # We just pass the new message to the chat_engine.
    try:
        response = chat_engine.chat(message)
        return str(response)
    except Exception as e:
        return f"An error occurred: {e}"

print("Launching Gradio interface...")
iface = gr.ChatInterface(
    fn=chat_with_agent,
    title="Smart Customer Support Triage Agent",
    description=(
        "Hello! I'm your Smart Customer Support Triage Agent. "
        "I can answer questions about our product manuals, FAQs, and billing policies. "
        "Ask me anything!"
    ),
    examples=[
        "I can't login TechSolve.",
        "What is your refund policy?",
        "What services do your company offer?",
        "Can you tell me about the latest software update for Product Y?"
    ],
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    iface.launch()