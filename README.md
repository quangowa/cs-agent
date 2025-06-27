---
title: Smart Customer Support Agent
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: true
license: mit
short_description: Agentic retrieval & smart routing for customer support
tags:
- agent-demo-track
- llamaindex
- llamacloud
- rag
- agentic-retrieval
- smart-routing
- customer-service
- chatbot
- claude
- anthropic
- cohere-embed
---

# 💬 Smart Customer Support Triage Agent

An intelligent customer support system powered by LlamaIndex, Anthropic Claude and Gradio, featuring agentic routing and smart document retrieval across multiple knowledge bases.

> [!IMPORTANT]
> This project earned [Agent Track Honorable Mention](https://www.gradio.app/hackathon-winners) in “Gradio Agents & MCP Hackathon 2025”. Thank you so much! 🥹🙏❤️

## 🔮 Live Demo

Try the chatbot: **[https://huggingface.co/spaces/karenwky/cs-agent](https://huggingface.co/spaces/karenwky/cs-agent)**

_\* Llama-3.1-405B-Instruct supported by Nebius AI Studio is used as the LLM in live demo._ 

## 📹 Preview

<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/67d2db58176fdb283211e929/VZv-vBdGfYTsyD-3LuJYC.mp4"></video>

## 🚀 Technologies Used

- **LLM**: Anthropic Claude Sonnet 4.0 for natural language understanding and generation
- **RAG Framework**: LlamaIndex for agent framework and intelligent query routing
- **Cloud Platform**: LlamaCloud for document parsing & indexng, managed vector database
- **UI Framework**: Gradio for interactive web interface
- **Observability**: LlamaTrace (Arize Phoenix) for LLM tracing and monitoring
- **Language**: Python 3.12
- **Document Types**: PDF, PPTX, CSV support
- **Coding Assistance**: Gemini 2.5 Flash
- **Synthetic Data**: Grok Chat, Slidesgo for sample knowledge base generation

## 📁 Project Structure

```
cs-agent/
├── app.py                           # Main application logic
├── requirements.txt                 # Python dependencies
├── logo.png                         # Chatbot avatar image
└── data/                            # Sample knowledge base
    ├── ProductManuals/              # Product documentation
    ├── FAQGeneralInfo/              # FAQ and general information
    ├── BillingPolicy/               # Billing and payment policies
    └── CompanyIntroductionSlides/   # Company overview
```

## 🔄 Architecture & Workflow

<img src="https://cdn-uploads.huggingface.co/production/uploads/67d2db58176fdb283211e929/U3hKTIcnfrSE3ErjuFE8U.png"
    alt="Architecture & Workflow Diagram" width="1000" />

## ✨ Key Features

### 🎯 **Agentic Routing**
- Intelligent query classification and routing to appropriate knowledge bases
- Composite retrieval with automatic reranking
- Context-aware document selection based on query intent

### 🧠 **Metadata Awareness**
- Extracts and utilizes document metadata (e.g., author, creation date, modification date)
- File-specific information retrieval from document properties
- Enhanced context understanding through metadata integration

### 🔄 **Automatic Version Control**
- Tracks document versions automatically (e.g., `late_payment_policy_v2.pdf`)
- Maintains historical document access
- Version-aware responses for policy and procedure queries

### 💾 **Memory Management**
- Persistent chat memory with token limit
- Context preservation across conversation turns
- Condensed question generation for better retrieval

### 📊 **Observability**
- Real-time LLM tracing with Arize Phoenix
- Retrieval performance monitoring
- Source attribution and scoring visibility

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- LlamaCloud API key
- Anthropic API key
- (Optional) Arize Phoenix API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/karenwky/cs-agent
cd cs-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LLAMA_CLOUD_API_KEY="your-llamacloud-api-key"

# Optional: For Phoenix tracing
export PHOENIX_PROJECT_NAME="your-phoenix-project"
export PHOENIX_API_KEY="your-phoenix-api-key"
```

4. **Set up LlamaCloud indices**
   - Create a project named `CustomerSupportProject` in LlamaCloud
   - Upload documents to create these indices:
     - `ProductManuals`
     - `FAQGeneralInfo`
     - `BillingPolicy`
     - `CompanyIntroductionSlides`

5. **Launch the application**
```bash
python app.py
```

## 🗣️ Sample Questions

### 📚 **Product Manual Queries**
- "Help! No response from the app, I can't do anything. What should I do?"
- "What are the procedures to set up task automation?"
- "How can your software help team collaboration?"
- "Who is the author of the product manual and when is the last modified date?"

### ❓ **General FAQ & Company Info**
- "Who are the founders of this company? What are their backgrounds?"
- "Is your company environmentally friendly?"
- "What kind of technical detail can I expect for sustainability questions?"

### 💰 **Billing & Policy Questions**
- "I got a $200 invoice outstanding for 45 days. How much is the late charge?"
- "What is your latest late payment policy?"
- "If I enable auto-pay but my payment method expires, will I get notified?"
- "Does the 14-day refund policy refer to calendar days or business days?"

### 🔧 **Complex Multi-Domain Queries**
- "If I sign up for annual 'Pro' subscription with 10% discount but cancel after 20 days, what exact refund amount considering the 14-day policy?"
- "For multi-user discounts, if I add a 5th user mid-cycle, when does the 10% discount apply and how is prorating calculated?"

## 🔮 Potential Enhancements

### 🎯 **Advanced AI Capabilities**
- **Multi-modal Support**: Add image and video processing for visual troubleshooting guides
- **Voice Integration**: Implement speech-to-text and text-to-speech for accessibility
- **Sentiment Analysis**: Detect customer frustration levels and escalate appropriately
- **Predictive Analytics**: Anticipate customer needs based on interaction patterns

### 🔧 **Technical Improvements**
- **Streaming Responses**: Implement real-time response streaming for better UX
- **Caching Layer**: Add Redis/Memcached for frequently accessed information
- **API Gateway**: Create RESTful APIs for integration with existing support systems
- **Containerization**: Docker support for easy deployment and scaling

### 📊 **Analytics & Monitoring**
- **Custom Dashboards**: Build comprehensive analytics for support metrics
- **A/B Testing**: Framework for testing different response strategies
- **Performance Metrics**: Track response accuracy, user satisfaction, and resolution rates
- **Automated Quality Assurance**: Implement response quality scoring and feedback loops

### 🔐 **Security & Compliance**
- **Authentication System**: User management and role-based access control
- **Data Encryption**: End-to-end encryption for sensitive customer data
- **Audit Logging**: Comprehensive logging for compliance and security monitoring
- **GDPR Compliance**: Data privacy controls and user data management

### 🌐 **Integration & Scalability**
- **CRM Integration**: Connect with Salesforce, HubSpot, or other CRM systems
- **Ticketing System**: Integration with Zendesk, Jira Service Management
- **Multi-language Support**: Internationalization for global customer base
- **Load Balancing**: Horizontal scaling for high-traffic scenarios

### 🤖 **Advanced Agent Capabilities**
- **Tool Integration**: Connect with external APIs and services
- **Workflow Automation**: Automated ticket creation and routing
- **Escalation Logic**: Smart human handoff based on complexity scoring
- **Learning System**: Continuous improvement from customer interactions

## 🙏 Acknowledgments

- [RAG is dead, long live agentic retrieval — LlamaIndex](https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval)
- [LlamaCloudIndex + LlamaCloudRetriever - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/llama_cloud_index/)
- [Condense plus context - LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/chat_engines/condense_plus_context/)
- [LlamaIndex | Phoenix](https://arize.com/docs/phoenix/tracing/integrations-tracing/llamaindex#setup)

## 📝 License

This project is licensed under the MIT License. 