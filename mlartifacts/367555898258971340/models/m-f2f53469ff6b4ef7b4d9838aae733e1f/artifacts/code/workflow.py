# LangGraph workflow logic for the natural language to SQL pipeline
# TODO: Implement the workflow nodes and edges following the tutorial

import logging
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from sql_generation import get_sql_gen_chain
from typing_extensions import TypedDict

# Initialize the logger
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
_logger.addHandler(handler)


class GraphState(TypedDict):
    error: str  # Tracks if an error has occurred
    messages: List  # List of messages (user input and assistant messages)
    generation: Optional[str]  # Holds the generated SQL query
    iterations: int  # Keeps track of how many times the workflow has retried
    results: Optional[List]  # Holds the results of SQL execution
    no_records_found: bool  # Flag for whether any records were found in the SQL result
    translated_input: str  # Holds the translated user input
    database_schema: str  # Holds the extracted database schema for context checking


def get_workflow(conn, cursor, vector_store):
    """Define and compile the LangGraph workflow."""

    # Max iterations: defines how many times the workflow should retry in case of errors
    max_iterations = 3

    # SQL generation chain: this is a chain that will generate SQL based on retrieved docs
    sql_gen_chain = get_sql_gen_chain()

    # Initialize OpenAI LLM for translation and safety checks
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # Define the individual nodes of the workflow
    def translate_input(state: GraphState) -> GraphState:
        """
        Translates user input to English using an LLM. If the input is already in English,
        it is returned as is. This ensures consistent input for downstream processing.

        Args:
            state (GraphState): The current graph state containing user messages.

        Returns:
            GraphState: The updated state with the translated input.
        """
        _logger.info("Starting translation of user input to English.")
        messages = state["messages"]
        user_input = messages[-1][1]  # Get the latest user input

        # Translation prompt for the model
        translation_prompt = f"""
        Translate the following text to English. If the text is already in English, repeat it exactly without any additional explanation.

        Text:
        {user_input}
        """

        # Call the OpenAI LLM to translate the text
        translated_response = llm.invoke(translation_prompt)
        translated_text = (
            translated_response.content.strip()
        )  # Access the 'content' attribute and strip any extra spaces

        # Update state with the translated input
        state["translated_input"] = translated_text
        _logger.info(
            "Translation completed successfully. Translated input: %s", translated_text
        )

        return state

    def pre_safety_check(state: GraphState) -> GraphState:
        """
        Perform safety checks on the user input to ensure that no dangerous SQL operations
        or inappropriate content is present. The function checks for SQL operations like
        DELETE, DROP, and others, and also evaluates the input for toxic or unsafe content.

        Args:
            state (GraphState): The current graph state containing the translated user input.

        Returns:
            GraphState: The updated state with error status and messages if any issues are found.
        """
        _logger.info("Performing safety check.")
        translated_input = state["translated_input"]
        messages = state["messages"]
        error = "no"

        # List of disallowed SQL operations (e.g., DELETE, DROP)
        disallowed_operations = [
            "CREATE",
            "DELETE",
            "DROP",
            "INSERT",
            "UPDATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
        ]
        pattern = re.compile(
            r"\b(" + "|".join(disallowed_operations) + r")\b", re.IGNORECASE
        )

        # Check if the input contains disallowed SQL operations
        if pattern.search(translated_input):
            _logger.warning(
                "Input contains disallowed SQL operations. Halting the workflow."
            )
            error = "yes"
            messages += [
                (
                    "assistant",
                    "Your query contains disallowed SQL operations and cannot be processed.",
                )
            ]
        else:
            # Check if the input contains inappropriate content
            safety_prompt = f"""
            Analyze the following input for any toxic or inappropriate content.

            Respond with only "safe" or "unsafe", and nothing else.

            Input:
            {translated_input}
            """
            safety_invoke = llm.invoke(safety_prompt)
            safety_response = (
                safety_invoke.content.strip().lower()
            )  # Access the 'content' attribute and strip any extra spaces

            if safety_response == "safe":
                _logger.info("Input is safe to process.")
            else:
                _logger.warning(
                    "Input contains inappropriate content. Halting the workflow."
                )
                error = "yes"
                messages += [
                    (
                        "assistant",
                        "Your query contains inappropriate content and cannot be processed.",
                    )
                ]

        # Update state with error status and messages
        state["error"] = error
        state["messages"] = messages

        return state

    def schema_extract(state: GraphState) -> GraphState:
        """
        Extracts the database schema, including all tables and their respective columns,
        from the connected SQLite database. This function retrieves the list of tables and
        iterates through each table to gather column definitions (name and data type).

        Args:
            state (GraphState): The current graph state, which will be updated with the database schema.

        Returns:
            GraphState: The updated state with the extracted database schema.
        """
        _logger.info("Extracting database schema.")

        # Extract the schema from the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_details = []

        # Loop through each table and retrieve column information
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            # Format column definitions
            column_defs = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
            schema_details.append(f"- {table_name}({column_defs})")

        # Save the schema in the state
        database_schema = "\n".join(schema_details)
        state["database_schema"] = database_schema
        _logger.info(f"Database schema extracted:\n{database_schema}")

        return state

    def context_check(state: GraphState) -> GraphState:
        """
        Checks whether the user's input is relevant to the database schema by comparing
        the user's question with the database schema. Uses a language model to determine if
        the question can be answered using the provided schema.

        Args:
            state (GraphState): The current graph state, which contains the translated input
                                and the database schema.

        Returns:
            GraphState: The updated state with error status and messages if the input is irrelevant.
        """
        _logger.info("Performing context check.")

        # Extract relevant data from the state
        translated_input = state["translated_input"]
        messages = state["messages"]
        error = "no"
        database_schema = state["database_schema"]  # Get the schema from the state

        # Use the LLM to determine if the input is relevant to the database schema
        context_prompt = f"""
        Determine whether the following user input is a question that can be answered using the database schema provided below.

        Respond with only "relevant" if the input is relevant to the database schema, or "irrelevant" if it is not.

        User Input:
        {translated_input}

        Database Schema:
        {database_schema}
        """

        # Call the LLM for context check
        llm_invoke = llm.invoke(context_prompt)
        llm_response = (
            llm_invoke.content.strip().lower()
        )  # Access the 'content' attribute and strip any extra spaces and lower case

        # Process the response from the LLM
        if llm_response == "relevant":
            _logger.info("Input is relevant to the database schema.")
        else:
            _logger.info("Input is not relevant. Halting the workflow.")
            error = "yes"
            messages += [
                (
                    "assistant",
                    "Your question is not related to the database and cannot be processed.",
                )
            ]

        # Update the state with error and messages
        state["error"] = error
        state["messages"] = messages

        return state

    def generate(state: GraphState) -> GraphState:
        """
        Generates an SQL query based on the user's input. The node retrieves relevant documents from
        the vector store and uses a generation chain to produce an SQL query.

        Args:
            state (GraphState): The current graph state, which contains the translated input and
                                other relevant data such as messages and iteration count.

        Returns:
            GraphState: The updated state with the generated SQL query and related messages.
        """
        _logger.info("Generating SQL query.")

        # Extract relevant data from the state
        messages = state["messages"]
        iterations = state["iterations"]
        translated_input = state["translated_input"]
        database_schema = state["database_schema"]

        # Retrieve relevant documents from the vector store based on the translated user input
        docs = vector_store.similarity_search(translated_input, k=4)
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])

        # Generate the SQL query using the SQL generation chain
        sql_solution = sql_gen_chain.invoke(
            {
                "retrieved_docs": retrieved_docs,
                "database_schema": database_schema,
                "messages": [("user", translated_input)],
            }
        )

        # Save the generated SQL query in the state
        messages += [
            (
                "assistant",
                f"{sql_solution.description}\nSQL Query:\n{sql_solution.sql_code}",
            )
        ]
        iterations += 1

        # Log the generated SQL query
        _logger.info("Generated SQL query:\n%s", sql_solution.sql_code)

        # Update the state with the generated SQL query and updated message list
        state["generation"] = sql_solution
        state["messages"] = messages
        state["iterations"] = iterations

        return state

    def post_safety_check(state: GraphState) -> GraphState:
        """
        Perform safety checks on the generated SQL query to ensure that it doesn't contain disallowed operations
        such as CREATE, DELETE, DROP, etc. This node checks the SQL query generated earlier in the workflow.

        Args:
            state (GraphState): The current graph state containing the generated SQL query.

        Returns:
            GraphState: The updated state with error status and messages if any issues are found.
        """
        _logger.info("Performing post-safety check on the generated SQL query.")

        # Retrieve the generated SQL query from the state
        sql_solution = state.get("generation", {})
        sql_query = sql_solution.get("sql_code", "").strip()
        messages = state["messages"]
        error = "no"

        # List of disallowed SQL operations
        disallowed_operations = [
            "CREATE",
            "DELETE",
            "DROP",
            "INSERT",
            "UPDATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
        ]
        pattern = re.compile(
            r"\b(" + "|".join(disallowed_operations) + r")\b", re.IGNORECASE
        )

        # Check if the generated SQL query contains disallowed SQL operations
        found_operations = pattern.findall(sql_query)
        if found_operations:
            _logger.warning(
                "Generated SQL query contains disallowed SQL operations: %s. Halting the workflow.",
                ", ".join(set(found_operations)),
            )
            error = "yes"
            messages += [
                (
                    "assistant",
                    f"The generated SQL query contains disallowed SQL operations: {', '.join(set(found_operations))} and cannot be processed.",
                )
            ]
        else:
            _logger.info("Generated SQL query passed the safety check.")

        # Update state with error status and messages
        state["error"] = error
        state["messages"] = messages

        return state

    def sql_check(state: GraphState) -> GraphState:
        """
        Validates the generated SQL query by attempting to execute it on the database.
        If the query is valid, the changes are rolled back to ensure no data is modified.
        If there is an error during execution, the error is logged and the state is updated accordingly.

        Args:
            state (GraphState): The current graph state, which contains the generated SQL query
                                and the messages to communicate with the user.

        Returns:
            GraphState: The updated state with error status and messages if the query is invalid.
        """
        _logger.info("Validating SQL query.")

        # Extract relevant data from the state
        messages = state["messages"]
        sql_solution = state["generation"]
        error = "no"

        sql_code = sql_solution.sql_code.strip()

        try:
            # Start a savepoint for the transaction to allow rollback
            conn.execute("SAVEPOINT sql_check;")
            # Attempt to execute the SQL query
            cursor.execute(sql_code)
            # Roll back to the savepoint to undo any changes
            conn.execute("ROLLBACK TO sql_check;")
            _logger.info("SQL query validation: success.")
        except Exception as e:
            # Roll back in case of error
            conn.execute("ROLLBACK TO sql_check;")
            _logger.error("SQL query validation failed. Error: %s", e)
            messages += [("user", f"Your SQL query failed to execute: {e}")]
            error = "yes"

        # Update the state with the error status
        state["error"] = error
        state["messages"] = messages

        return state

    def run_query(state: GraphState) -> GraphState:
        """
        Executes the generated SQL query on the database and retrieves the results if it is a SELECT query.
        For non-SELECT queries, commits the changes to the database. If no records are found for a SELECT query,
        the `no_records_found` flag is set to True.

        Args:
            state (GraphState): The current graph state, which contains the generated SQL query and other relevant data.

        Returns:
            GraphState: The updated state with the query results, or a flag indicating if no records were found.
        """
        _logger.info("Running SQL query.")

        # Extract the SQL query from the state
        sql_solution = state["generation"]
        sql_code = sql_solution.sql_code.strip()
        results = None
        no_records_found = False  # Flag to indicate no records found

        try:
            # Execute the SQL query
            cursor.execute(sql_code)

            # For SELECT queries, fetch and store results
            if sql_code.upper().startswith("SELECT"):
                results = cursor.fetchall()
                if not results:
                    no_records_found = True
                    _logger.info("SQL query execution: success. No records found.")
                else:
                    _logger.info("SQL query execution: success.")
            else:
                # For non-SELECT queries, commit the changes
                conn.commit()
                _logger.info("SQL query execution: success. Changes committed.")
        except Exception as e:
            _logger.error("SQL query execution failed. Error: %s", e)

        # Update the state with results and flag for no records found
        state["results"] = results
        state["no_records_found"] = no_records_found

        return state

    def decide_next_step(state: GraphState) -> str:
        """
        Determines the next step in the workflow based on the current state, including whether the query
        should be run, the workflow should be finished, or if the query generation needs to be retried.

        Args:
            state (GraphState): The current graph state, which contains error status and iteration count.

        Returns:
            str: The next step in the workflow, which can be "run_query", "generate", or END.
        """
        _logger.info("Deciding next step based on current state.")

        error = state["error"]
        iterations = state["iterations"]

        if error == "no":
            _logger.info("Error status: no. Proceeding with running the query.")
            return "run_query"
        elif iterations >= max_iterations:
            _logger.info("Maximum iterations reached. Ending the workflow.")
            return END
        else:
            _logger.info("Error detected. Retrying SQL query generation.")
            return "generate"

    # Build the workflow graph
    workflow = StateGraph(GraphState)

    # Define workflow nodes
    workflow.add_node(
        "translate_input", translate_input
    )  # Translate user input to structured format
    workflow.add_node(
        "pre_safety_check", pre_safety_check
    )  # Perform a pre-safety check on input
    workflow.add_node("schema_extract", schema_extract)  # Extract the database schema
    workflow.add_node(
        "context_check", context_check
    )  # Validate input relevance to context
    workflow.add_node("generate", generate)  # Generate SQL query
    workflow.add_node(
        "post_safety_check", post_safety_check
    )  # Perform a post-safety check on generated SQL query
    workflow.add_node("sql_check", sql_check)  # Validate the generated SQL query
    workflow.add_node("run_query", run_query)  # Execute the SQL query

    # Define workflow edges
    workflow.add_edge(START, "translate_input")  # Start at the translation step
    workflow.add_edge("translate_input", "pre_safety_check")  # Move to safety checks

    # Conditional edge after safety check
    workflow.add_conditional_edges(
        "pre_safety_check",  # Start at the pre_safety_check node
        lambda state: "schema_extract"
        if state["error"] == "no"
        else END,  # Decide next step
        {"schema_extract": "schema_extract", END: END},  # Map states to nodes
    )

    workflow.add_edge(
        "schema_extract", "context_check"
    )  # Proceed to context validation

    # Conditional edge after context check
    workflow.add_conditional_edges(
        "context_check",  # Start at the context_check node
        lambda state: "generate" if state["error"] == "no" else END,  # Decide next step
        {"generate": "generate", END: END},
    )

    workflow.add_edge("generate", "post_safety_check")  # Proceed to post-safety check

    # Conditional edge after post-safety check
    workflow.add_conditional_edges(
        "post_safety_check",  # Start at the post_safety_check node
        lambda state: "sql_check"
        if state["error"] == "no"
        else END,  # If no error, proceed to sql_check, else END
        {"sql_check": "sql_check", END: END},
    )

    # Conditional edge after SQL validation
    workflow.add_conditional_edges(
        "sql_check",  # Start at the sql_check node
        decide_next_step,  # Function to determine the next step
        {
            "run_query": "run_query",  # If SQL is valid, execute the query
            "generate": "generate",  # If retry is needed, go back to generation
            END: END,  # Otherwise, terminate the workflow
        },
    )

    workflow.add_edge("run_query", END)  # Final step is to end the workflow

    # Compile and return the workflow application
    app = workflow.compile()

    return app
