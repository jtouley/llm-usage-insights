# LLM Chat Insights PoC

## What Problem This Solves

This Proof-of-Concept analyzes your exported OpenAI ChatGPT conversations to extract insights about topic trends, conversation effectiveness, and interaction patterns. It helps understand your AI usage patterns, identifies your most productive conversation strategies, and clusters similar conversations to enhance your productivity with AI assistants.

## Quick Start

1. **Setup**
```bash
# Clone the repository
git clone https://github.com/jtouley/llm-usage-insights.git
cd chat-insights-poc

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
2. **Run the app**
```bash
streamlit run streamlit_app/Home.py
```
3. **Upload your data**
- Export your OpenAI conversations
- Upload the conversations.json file when prompted

## Architecture
This PoC uses a modular architecture with clear separation between data loading, preprocessing, analysis, and visualization. Sentence embeddings are used for semantic clustering, and effectiveness scoring identifies the most productive conversation patterns. The Streamlit interface provides interactive visualizations and insights.

## Future Enhancements
- Enhanced RAG workflow for contextual recommendations
- Custom fine-tuning configuration generation
- Usage pattern tracking and optimization
- Comparative analysis against aggregated (anonymized) benchmarks
- Dashboard export functionality for sharing insights

## Development
```bash
# Run tests
pytest

# Run linting
pre-commit run --all-files
```
