# FinScope: A Multi-Agent AI System for Financial Market Analysis

**FinScope** is an innovative Multi-Agent System (MAS) designed to provide superior financial market analysis by intelligently integrating **Quantitative (Technical)** forecasting with **Qualitative (Contextual)** news interpretation. It moves beyond traditional siloed approaches to deliver holistic, synthesized, and actionable investment recommendations.

## 🚀 Key Features and Innovations

FinScope's architecture introduces novel solutions to common financial analysis challenges:

### 1. The Decision Making Agent (Core Innovation)
A dedicated, intelligent agent responsible for synthesizing the distinct outputs of the forecasting (LSTM) and contextual (LLM) agents.
* **Conflict Resolution:** Autonomously weighs evidence and resolves conflicts between signals (e.g., technical "buy" signal vs. critical market news) to generate a single, unified recommendation.
* **Holistic Insight:** Ensures recommendations are grounded in both market mechanics and real-world context.

### 2. Modular Multi-Agent Architecture
The system uses a flexible MAS framework orchestrated by a **Coordinator Agent**.
* **Scalability:** Allows for easy extension and integration of new analytical tools (e.g., adding a different forecasting model or an arbitrage agent).
* **Future-Proofing:** Components like the LSTM Agent can be upgraded or replaced without re-engineering the entire pipeline.

### 3. Dynamic & Deep Context Integration
We use a Large Language Model (LLM) for advanced contextual analysis, moving beyond basic sentiment scoring.
* **Contextual Depth:** The LLM Agent processes raw news articles for nuanced interpretation, summarizing key risks, opportunities, and overall market sentiment.
* **Real-Time Data:** Dedicated fetching agents ensure dynamic acquisition of current market prices and breaking news.

---

## 🛠️ System Architecture Overview

The FinScope system operates through a five-stage process managed by a central **Coordinator Agent**:

| Agent | Responsibility | Key Output |
| :--- | :--- | :--- |
| **Price Fetcher Agent** | Retrieves historical time-series data and current market prices. | `Price Data` |
| **Data Fetcher Agent** | Scrapes and pre-processes financial news, social media data, or reports. | `News Corpus` |
| **LSTM Agent** | Executes time-series forecasting to predict future price movements (e.g., next 7 days). | `Quantitative Forecast` |
| **LLM Agent** | Analyzes the `News Corpus` for sentiment, risks, opportunities, and overall context. | `Qualitative Context Score` |
| **Decision Making Agent** | Combines the `Quantitative Forecast` and `Qualitative Context Score` to generate the final recommendation. | **Actionable Recommendation** |

---

## 💻 Getting Started

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yuvrajpradhan/finscope.git](https://github.com/yuvrajpradhan/finscope.git)
    cd finscope
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The system requires API keys for data fetching and the LLM service (e.g., OpenAI, Hugging Face, etc.).

1.  Create a file named `.env` in the root directory.
2.  Add your necessary environment variables:
    ```
    # Example for LLM service
    OPENAI_API_KEY="your_openai_key"

    # Example for market data (e.g., Alpha Vantage)
    MARKET_DATA_API_KEY="your_data_key"
    ```

### Usage

To run the main analysis for a specific stock ticker:

```bash
python main.py --ticker MSFT
