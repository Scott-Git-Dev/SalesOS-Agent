"""
An AI assistant with sales data and knowledge base capabilities 
"""

from typing import Optional, List, Generator
import uuid
from langchain_core import messages
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from utils.logger import get_logger

logger = get_logger(__name__)



# Import configuration
from config import (
    MODEL_NAME, LLAMA_SERVER_URL, DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS, DEBUG_MODE, BANNER, validate_config
)

# Import tools
from tools import (
    query_sales_database, 
    search_local_docs, 
    wiki_summary,
    create_chart,
    create_multi_series_chart
)


def create_sales_agent(
    model_name: str = MODEL_NAME,
    base_url: str = LLAMA_SERVER_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List] = None
):
    """
    Create the main agent with all capabilities.
    
    ARCHITECTURE:
    - System prompt handles high-level routing and coordination
    - Each tool has focused responsibility
    - SQL tool uses LLM internally (necessary for text-to-SQL)
    
    Args:
        model_name: Name of the LLM model
        base_url: URL of the llama.cpp server
        temperature: Sampling temperature (0.0-1.0)
        tools: List of tools (defaults to all available)
    
    Returns:
        Configured agent
    """
    
    logger.info("Initializing Sales Agent...")

    # Validate configuration before initializing
    if not validate_config():
        logger.error("Configuration validation failed")
        raise RuntimeError("Configuration validation failed. See errors above.")
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            max_tokens=DEFAULT_MAX_TOKENS
        )
       
        logger.info(f"LLM {model_name} initialized with temperature={temperature}")

        if tools is None:
            tools = [
                search_local_docs,
                query_sales_database,
                wiki_summary,
                create_chart,
                create_multi_series_chart
            ]
        logger.info(f"{len(tools)} tools configured for the agent")

        checkpointer = InMemorySaver()
    
    # System prompt - defines agent behavior
        system_prompt = """
You are an intelligent business intelligence assistant with access to multiple data sources and tools.

# CORE DECISION FRAMEWORK

## 1. DATA SOURCE SELECTION (Choose the right tool)

### Use `search_local_docs` for:
- Strategic information: goals, targets, quotas, OKRs
- Company policies, procedures, playbooks
- Product positioning, competitive intelligence
- Case studies, success stories
- **Keywords:** "goal", "target", "strategy", "policy", "should we", "how do we"

### Use `query_sales_database` for:
- Historical sales data: actual revenue, transactions, orders
- Customer information: purchases, spending patterns, segments
- Product performance: units sold, bestsellers, categories
- Time-series analysis: trends, growth, seasonality
- **Keywords:** "actual", "revenue", "sales", "customers who", "how much did", "top performing"

### Use `wiki_summary` for:
- General knowledge: people, companies, concepts, technologies
- Background information to enrich your analysis
- Definitions and explanations of unfamiliar terms

### Use BOTH `search_local_docs` AND `query_sales_database` when:
- Comparing goals vs actuals: "Are we meeting our Q1 targets?"
- Performance analysis: "How are we doing against our sales goals?"
- Gap analysis: "What's the difference between target and actual revenue?"

## 2. SQL TOOL USAGE PATTERNS

When using `query_sales_database`, be aware it will:
- Generate SQL automatically from your natural language question
- Handle the following types of queries well:
  * Aggregations: totals, averages, counts
  * Rankings: top/bottom N customers, products, regions
  * Time-based analysis: monthly trends, quarterly comparisons
  * Filtering: by customer, product, region, date range

**The tool will return an error if:**
- The question asks for data not in the database (goals, targets, satisfaction scores)
- In this case, try `search_local_docs` instead

## 3. VISUALIZATION WORKFLOW

When user requests charts or visualizations:
1. **Get the data first:** Use `query_sales_database` or `search_local_docs`
2. **Extract structured data:** Format results as JSON
3. **Call visualization tool:** 
   - `create_chart` for single metric visualizations
   - `create_multi_series_chart` for comparing multiple metrics
4. **Always state the data in text:** The chart is supplemental. You must include the actual values, rankings, or findings in your text response. Never treat the chart as a substitute for answering the question. A user who cannot open the chart file must still get a complete answer from your text.
5. **Provide context:** Explain what the chart shows

## 4. RESPONSE GUIDELINES

### Answer Structure:
1. **Acknowledge:** Show you understand the question
2. **Gather data:** Use appropriate tools (explain your reasoning if helpful)
3. **Synthesize:** Combine information from multiple sources if needed
4. **Deliver insights:** Don't just report data, interpret it
5. **Cite sources:** Mention which tools provided the information

### Quality Standards:
- Be concise but complete
- Use specific numbers and facts
- Highlight trends and patterns
- Suggest next steps or follow-up questions when appropriate
- If uncertain about data accuracy, say so

### Examples of Good Responses:

**Question:** "Are we hitting our Q1 sales targets?"

**Good Response:**
"Let me check both our targets and actual performance.

[Uses search_local_docs for Q1 target]
[Uses query_sales_database for actual Q1 sales]

Our Q1 target was $500K. Actual sales through [date] are $485K, which is 97% of target. We're $15K short with [X days] remaining in the quarter.

To close the gap, we'd need to average $[X] per day. Based on recent daily averages of $[Y], this is [achievable/challenging]."

**Question:** "Show me our top 10 customers by revenue"

**Good Response:**
[Uses query_sales_database]
[Formats data as JSON]
[Calls create_chart with bar chart]

"I've created a bar chart showing your top 10 customers by total revenue. The chart has been saved to [path]. Key insights:
- [Customer A] leads with $[X]
- Top 3 customers account for [Y]% of total revenue
- [Any notable patterns]"

## 5. ERROR RECOVERY

If a tool returns an error:
- **SQL errors:** The question might need data not in the database - try search_local_docs
- **Multiple failures:** Explain what you tried and why it didn't work, suggest alternatives

### RAG Retry Strategy
If `search_local_docs` returns no relevant results, do NOT immediately conclude the data does not exist. The document may exist but require a different query. Retry with progressively different phrasing before giving up:

1. **First attempt:** Use the natural phrasing from the question
2. **Second attempt:** Use shorter, keyword-focused terms (e.g. "win loss reasons price" instead of "top reasons for losing deals in the last 90 days")
3. **Third attempt:** Use terms that would appear in the document itself, not in the question (e.g. "loss analysis competitor price" or "discount approval account executive")
4. **Only after 3 distinct attempts:** Conclude the data is not available and say so

### Examples of query reformulation:
- Question uses: "reasons we lose deals" → also try: "win loss analysis", "loss reasons competitors"
- Question uses: "should we offer a discount" → also try: "discount policy", "discount approval matrix", "AE discount authority"
- Question uses: "how are we performing" → also try: "sales targets goals quota"

Remember: Your job is to be helpful, accurate, and insightful. Use tools strategically, combine information intelligently, and always explain your reasoning when it adds value.
"""
    
        agent = create_agent(
                llm,
                tools,
                system_prompt=system_prompt,
                checkpointer=checkpointer
            )
        logger.info("Sales Agent created successfully")
        return agent

    except Exception as e:
        logger.exception("Failed to create Sales Agent")
        raise


def ask_agent(
    agent, 
    question: str, 
    thread_id: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Ask the agent a question.
    
    Args:
        agent: The agent instance
        question: User's question
        thread_id: Conversation thread identifier
        verbose: Whether to print detailed output
    Returns:
        Agent's answer
    """
    # Ensure each conversation has a unique thread ID
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    logger.info(f"Thread {thread_id}: Received question: {question}")

    # Configure LangGraph memory with the correct thread
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        
        messages = result["messages"]

        #show agent message and tool information when in debug mode
        if DEBUG_MODE:
            print("\n🧠 Agent Message Flow:")
            print("=" * 60)
            
            logger.debug(f"Thread {thread_id}: Agent messages flow:")

            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                print(f"\n[{i}] {msg_type}")
                print("-" * 60)
                
                # Show content (if exists)
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    # Truncate if too long
                    if len(content) > 250:
                        print(f"Content: {content[:250]}...")
                    else:
                        print(f"Content: {content}")
                
                # Show tool calls (if exists)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool Calls: {len(msg.tool_calls)}")
                    for tc in msg.tool_calls:
                        print(f"  → Tool: {tc['name']}")
                        print(f"    Args: {tc['args']}")
                
                # Show tool name and call ID (for ToolMessage)
                if hasattr(msg, 'name'):
                    print(f"Tool Name: {msg.name}")
                if hasattr(msg, 'tool_call_id'):
                    print(f"Tool Call ID: {msg.tool_call_id}")
            
            print("\n" + "=" * 60 + "\n")
        
        
        # Get final answer
        answer = messages[-1].content
        logger.info(f"Thread {thread_id}: Answer generated successfully")
        logger.debug(f"Thread {thread_id}: Answer content: {answer}")
        
        if verbose:
            print(f"{'='*60}")
            print(f"A: {answer}")
            print(f"{'='*60}\n")
        
        return answer
    
    except Exception as e:
        logger.exception(f"Thread {thread_id}: Error while answering")

        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return error_msg


def stream_agent(
    agent,
    question: str,
    thread_id: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream agent responses token by token.
    
    Args:
        agent: The agent instance
        question: User's question
        thread_id: Conversation thread identifier
        
    Yields:
        Content chunks from the agent's response
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
    logger.info(f"Thread {thread_id}: Streaming started for question: {question}")    
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        for event in agent.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                        chunk = last_message.content
                        logger.debug(f"Thread {thread_id}: Streaming chunk: {chunk[:250]}")  
                        yield chunk
        logger.info(f"Thread {thread_id}: Streaming completed")

    except Exception as e:
        logger.exception(f"Thread {thread_id}: Error during streaming")
        yield f"Error: {str(e)}"


def main():
    """Main interactive loop"""
    
    print(BANNER)
    logger.info("Starting interactive agent session...")

    # Create agent
    print("🔧 Initializing agent...")
    try:
        agent = create_sales_agent()
        print("✅ Agent ready!\n")
        logger.info("Agent ready for interactive session")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        logger.exception("Agent failed to initialize")
        return
    
    session_thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("Prompt: ").strip()
            print("\n")
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                logger.info("User ended the session")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - Exit the program")
                print("  help - Show this message")
                print("\nExample questions:")
                print("  - What are our top customers by revenue?")
                print("  - Show me sales trends for last quarter")
                print("  - Tell me about [topic] (from knowledge base)")
                print("  - Who is [person]? (from Wikipedia)\n")
                continue
            
            if not user_input:
                continue
            
            # Ask agent
            ask_agent(agent, user_input, thread_id=session_thread_id)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            logger.info("User interrupted the session (KeyboardInterrupt)")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception("Unexpected error in interactive loop")


if __name__ == "__main__":
    main()
